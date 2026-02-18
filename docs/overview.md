# XaaS Containers: Overview

## What Is XaaS Containers?

XaaS (Acceleration as a Service) Containers is a framework for building performance-portable containers for HPC.
The central problem we solve: standard containers are built once in a portable manner to support many different types of machines, but HPC systems vary widely in CPU micro-architectures, GPU generations, and available libraries.
Recompiling inside a container at deployment is slow and fragile.

Instead, XaaS defers the architecture-specific compilation step to the moment the target hardware is known, while still shipping a self-contained artifact that carries everything needed to produce an optimized binary.

## Two Approaches

Both approaches produce a final Docker image that runs the application optimized for the target system. The IR path compiles once to IR and lowers many times; the source path ships source and compiles once per target.
The same set of feature-layer Docker images are shared by both paths.

### IR Containers

The application source is compiled to LLVM intermediate representation (`.bc` bitcode files) rather than native code. The IR container ships the bitcode alongside the build configurations. At deployment time, the IR is lowered to native code by running the LLVM backend with the exact CPU flags and GPU targets of the destination machine. When the application can use an LLVM-based compiler (like Clang or Flang), we can optimize deployment by generating IRs covering all supported build configurations and minimize the deployment time.

See [ir-containers.md](ir-containers.md) for the full pipeline.

### Source Containers

The application source tree and build toolchain are packaged together. At deployment time the system capabilities are detected (CPU SIMD extensions, available GPUs, MPI libraries, FFT/BLAS libraries), the best-matching build configuration is selected, and the application is compiled natively.

Use it when your application cannot be supported by IR containers, e.g., the build system is complex or the project requires non-LLVM compilers (e.g. GCC or vendor tools).

## Installation

Requirements: Python 3.10+, Docker.

```bash
pip install .          # runtime only
pip install ".[dev]"   # adds pytest, ruff, black, isort, mypy
```

After installation, the `xaas` CLI is available. The two primary workflow entry points are `xaas ir` and `xaas source`.

Global defaults live in `xaas/config/system.yaml`:

| Field | Explanation | Default |
|---|---|---|
| `docker_repository` | Main Docker repository used to store images | `spcleth/xaas` |
| `runner_image` | Default image for running applications | `runner-19` |
| `ir_type` | Type of IR; currently only LLVM IR supported | `llvm-ir` |
| `parallelism_level` | Number of parallel jobs run during compilation and processing | `4` |

## Known Limitations

**Overall** both source and IR containers use a different configuration interface to specialize applications; this is a legacy of concurrent development of both approaches. We should merge the configuration interfaces so that both pipelines can leverage the same hardware detection and feature selection logic.

**IR pipeline:**
- Only CMake is implemented for build generation; Autotools builds are not supported yet. We could try [Bear](https://github.com/rizsotto/Bear/) but it does not support dry-runs, i.e., you have to run full compilation [to get the compilation database.](https://github.com/rizsotto/Bear/issues/284) Alternatively, we can look into [compiledb](https://github.com/nickdiego/compiledb) (might be abandoned) or into [compiledb's fork](https://github.com/fcying/compiledb-go).
- Fortran is not yet supported.
- `icpx` compile commands are parsed but raise `NotImplementedError` in the analyzer; we have not integrated the oneAPI compiler into the IR pipeline yet.
- FFTW3 dependency layers are x86_64 only.

**Source containers:**
- `automated` deployment mode requires `GOOGLE_API_KEY`. We should properly warn that this requires an LLM call.
- ICON (`icon-model`) and CloudSC (`cloudsc`) applications are not yet supported.
- `arm64` source base image exist and have been used on GH200 system, but some of the feature layer ecosystem is not yet available for arm64.

