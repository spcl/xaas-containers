# IR Containers

IR containers ship LLVM bitcode (`.bc` files) instead of native binaries. The bitcode is produced from the application source during the build pipeline and lowered to native code at deployment time on the target machine.

## Pipeline Steps

Build step and the corresponding result:

```
  1. xaas ir buildgen   → buildgen.yml
  2. xaas ir analyze    → build_analyze.yml
  3. xaas ir preprocess → preprocess.yml
  4. xaas ir cpu-tuning → cpu_tuning.yml
  5. xaas ir irs        → ir_compilation.yml + irs/ directory with the list of IRs
  6. xaas ir container  → Generic Docker image {name}-ir
  7. xaas ir deploy     → Specialized Docker image {name}-deploy-{features}
```

### Step 1 — Build (`xaas ir buildgen <config>`)

We enumerate all feature combinations and run CMake configuration in Docker to produce `compile_commands.json` for each combination.

**Docker image:** `spcleth/xaas:builder-19-dev` (or a derived image when `layers_deps` is set — the dep layers are copied in and a temporary Dockerfile is generated)

**Process:**
- Computes the Cartesian product of `features_boolean` flags (2^n combinations) and `features_select` options.
- For each combination, creates `{working_directory}/build/build_{name}/`.
- Runs CMake configuration inside Docker, with source mounted at `/source`, and build directory at `/build`. This guarantees a consistent environment between configurations that captures the exact compile commands.

**Outputs:** `{working_directory}/buildgen.yml` - containing all build result.

### Step 2 — Analyze (`xaas ir analyze <config>`)

Compare `compile_commands.json` across all builds to identify which source files have different compile flags between configurations.

**Inputs:** `{working_directory}/buildgen.yml` from the previous step.

**Process:**
- Parses every compile command in every build directory
- Supported compiler frontends: `clang++`, `clang`, `cc`, `c++` (Clang), `nvcc` (CUDA's nvcc), and `icpx` (not fully supported yet).
- CUDA-specific: we handle extra flags like `--options-file` (compressed compilation flags), `-ccbin` (host compiler) and `-gencode`/`--generate-code` (distinguishes SASS `sm_XX` and PTX `compute_XX` targets for different compute capabilities).
- Classifies divergences per different translation unit into specific categories.

**Outputs:** `{working_directory}/build_analyze.yml` - contains mappings from file (translation unit) to default command and divergences between different configurations.

### Step 3 — Preprocess (`xaas ir preprocess run <config>`)

We run `clang -E -P` on each divergent source file inside the build containers to produce preprocessed `.i` files. Then, we compute the MD5-hash of preprocessing result to determine whether files are truly different after macro expansion.
By default, it also runs `/tools/openmp-finder/omp-finder` inside the container to detect if the file contains OpenMP constructs like pragmas.
By default, we run a batch of 128 files to limit the disk usage.
If a file's hash matches the baseline and its only divergence is in compile-time definitions and include paths, then we can mark it as identical - we just verified that the different macros and includes do not affect the actual code after preprocessing. If a file is truly different, then we have to keep it as a separate IR file.

Run `xaas ir preprocess summary <config>` to print preprocessing results

**Inputs:** `{working_directory}/build_analyze.yml` from the previous step.

**Outputs:** `{working_directory}/preprocess.yml` — per-target file entries containing the MD5 hash of preprocessed file and OpenMP detection results>

### Step 4 — CPU Tuning (`xaas ir cpu-tuning run <config>`)

For source files whose only divergence is CPU tuning flags, e.g., `-mavx` or `-march=`, we extract the exact LLVM target attributes (`target-cpu`, `target-features`, `tune-cpu`) that would be applied.
These are stored so the IR container can re-apply the correct ones at deployment. This allows us to create a single IR file if the only difference is in CPU tuning flags, since the same IR can be optimized with different target attributes at deploy time.
If the source file depends on the specific architecture, e.g., contains different code paths that include vectorization intrinsics, we will catch this during preprocessing as preprocessed files will be different between build configurations.

LLVM will use different target attributes. We do not hardcode a mapping from compiler flags to attributes; instead, we run the LLVM `opt` tool with a custom pass `replace-target-features` that extracts the attributes from the IR generated with the original compile command. This ensures we capture the exact attributes that LLVM would apply, including any implicit ones.

Run `xaas ir cpu-tuning summary <config>` to print CPU tuning results.

**Inputs:** `{working_directory}/preprocess.yml` from the previous step.

**Outputs:** `{working_directory}/cpu_tuning.yml` — updated preprocessing result with additional entries storing target flags.

### Step 5 — IR Generation (`xaas ir irs run <config>`)

We compile each unique source file to LLVM bitcode (`.bc`) using the original CMake compile command modified to add `-emit-llvm`. We deduplicate identical IR files by MD5 hash, and store only the unique subset needed to support all build configurations. For files that only differ in CPU tuning flags, we generate a single IR and re-apply the correct target attributes at deployment time.
Files sharing the same preprocessed hash and compatible divergence profile (same CPU tuning, compiler, optimizations, OpenMP) share a single LLVM IR file.

When CPU tuning flags are non-empty for a specific translation unikt, we append `-mllvm -disable-llvm-optzns` to defer LLVM optimization passes until deployment. If we optimize now and later try to re-optimize with different target attributes, we might end up with suboptimal code.

For CUDA, we remove all existing `-gencode` and `-arch` flags and replace them them with standardized gencode for all 14 supported SASS targets plus PTX for the highest (compatibility with future GPU devices).
This way, we can run on all GPUs even if the project does not specify a portable build step.
Supported CUDA compute capabilities: `50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90`

Run `xaas ir irs summary <config>` to print IR build results.

**Inputs:** `{working_directory}/cpu_tuning.yml` from the previous step.

**Outputs:**
- `{working_directory}/ir_compilation.yml` — updated result with paths to created IR files.
- `{working_directory}/irs/<cmake-target>/<hash>/<id>/<file>.bc` — bitcode files

### Step 6 — Container Build (`xaas ir container <config>`)

Package all IR bitcode files, configured build directories, and the source tree into a Docker image. Generate per-build `build.sh` scripts that replay compilation using IR inputs.
You can use the `--docker-repository` option to specify a custom repository for the output image; otherwise, it defaults to `spcleth/xaas`.

**Inputs:** `{working_directory}/ir_compilation.yml` from the previous step.

**Outputs:** Docker image with the name in the format: `{docker_repository}:{project_name}-ir`

### Step 7 — Deploy (`xaas ir deploy <config>`)

We build a deployment-ready Docker image from the IR container by composing hardware-specific feature layers, JIT-compiling all bitcode to native objects with GNU `parallel`, and packaging the result into a minimal runtime image.
Each `build.sh` builds all translation units but from an IR file, and applies LLVM's `opt` with new target attributes when CPU tuning is needed.
All compilation results are placed where CMake expects `.o` files. After that, we run the linking steps with original CMake scripts, which finalizes the build process.

**Inputs:** deployment configuration YAML (see examples in `configs`).

**Outputs:** deployed Docker image with the name in the format: `{docker_repository}:{image_name}-deploy-{feature_combo}`.
