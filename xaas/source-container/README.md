## Running the Tool

The tool supports three modes: **user**, **test**, and **automated**, each with different levels of control.

### Basic Usage

- **User mode** (interactive selection):
  ```bash
  python3 run.py <project_name>
  ```

- **Test mode** (manual option selection using flags):
  ```bash
  python3 run.py <project_name> --mode test --test-options "vectorization_flags=sse4.1 gpu_backends=CUDA parallel_libraries=OpenMP"
  ```

- **Automated mode** (Gemini selects the build flags from the avaliable build options):
  ```bash
  python3 run.py <project_name> --mode automated
  ```

### Example with Specialization Options

- Basic test mode example:
  ```bash
  python3 run.py gromacs --mode test --test-options "vectorization_flags=sse4.1 gpu_backends=CUDA parallel_libraries=OpenMP"
  ```

- Complex test mode with multiple flags:
  ```bash
  python3 run.py gromacs --mode test --test-options "vectorization_flags=sse4.1 gpu_backends=CUDA parallel_libraries=OpenMP,openmpi fft_libraries=\"mkl (CPU),MKL (GPU)\" linear_algebra_libraries=None"
  ```

### Optional Flags

- `--debug`: Enable debug output for internal states
  ```bash
  python3 run.py gromacs --debug
  ```

- `--milc_app`: Specify the MILC application target (only for `milc` in test mode)
  ```bash
  python3 run.py milc --mode test --milc_app su3_rmd
  ```

- `--base-image`: Override the base Docker image
  ```bash
  python3 run.py gromacs --base-image intel/oneapi-hpckit:latest
  ```

---

## Build Flags Reference

When using `--test-options`, you can pass any combination of these flags (comma-separated):

- `vectorization_flags`: e.g., `sse4.1`, `avx2`, `avx512`, `neon`
- `gpu_backends`: e.g., `CUDA`, `ROCm`, `OpenCL`, `None`
- `parallel_libraries`: e.g., `OpenMP`, `openmpi`, `intel-oneapi-mpi`
- `fft_libraries`: e.g., `FFTW`, `mkl (CPU)`, `MKL (GPU)`, `rocfft`
- `linear_algebra_libraries`: e.g., `OpenBLAS`, `MKL`, `cuBLAS`, `ScaLAPACK`, `None`
- `optimization_build_flags`: project-specific tuning flags

---

## Setup

Before running the tool, set up the environment:

```bash
cd source-container
bash setup.sh
```

**Important**: You must export your Gemini API key before running the tool:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Alternatively, you can edit `setup.sh` and hardcode the API key there.

---

## Directory Structure

Ensure that the following project directories exist at the **top level** of the repository, next to the `source-container` directory:

```text
.
├── gromacs-2025.0
├── milc_qcd-7.8.1
├── dwarf-p-cloudsc
├── source-container
│   ├── gemini_interface.py
│   ├── checker.py
│   ├── run.py
│   └── ...
```

---