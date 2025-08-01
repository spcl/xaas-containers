import json
import re

from xaas.config import Language


class Checker:
    def __init__(self, specialization_points, system_features):
        self.specialization_points = specialization_points
        self.system_features = system_features

    # comeplete (sort of)
    def get_vectorization_flags(self):
        # TODO: returns AVX_512_knl on non_knl archs -- need to be fixed!
        # Output: {'sse': '-mfpmath=sse', 'avx': '-DAVX', 'avx_512': '-DAVX512'}

        system_vectorizations = {
            vec.lower()
            for vec in self.system_features.get("CPU Info", {}).get("Supported Vectorizations", [])
        }
        json_simd = {
            k.lower(): v.get("build_flag", "")
            for k, v in self.specialization_points.get("simd_vectorization", {}).items()
        }

        # Enhanced comprehension for optimization flags, including special cases
        json_optimization = {
            (
                "avx_512"
                if re.search(r"-davx512", flag, re.IGNORECASE)
                else re.sub(r"-d", "", flag, flags=re.IGNORECASE).lower()
            ): flag
            for flag in self.specialization_points.get("optimization_build_flags", [])
        }

        combined_flags = {**json_simd, **json_optimization}

        # Include special flags AUTO, NONE, Portable from simd_vectorization or optimization_build_flags
        for special in ["auto", "none", "portable"]:
            if special in self.specialization_points.get("simd_vectorization", {}):
                combined_flags[special] = self.specialization_points["simd_vectorization"][
                    special
                ].get("build_flag", special)
            elif any(
                flag.lower() == f"-d{special}"
                for flag in self.specialization_points.get("optimization_build_flags", [])
            ):
                combined_flags[special] = f"-D{special.upper()}"

        system_map = {
            "sse": "SSE",
            "sse2": "SSE2",
            "sse4_1": "SSE4.1",
            "sse4_2": "SSE4.2",
            "avx": ["AVX", "AVX_128_FMA", "AVX_256"],
            "avx2": ["AVX2", "AVX2_128", "AVX2_256"],
            "avx512": "AVX_512",
            "avx_512": "AVX_512",
            "avx512f": "AVX_512",
            "avx512dq": "AVX_512",
            "avx512cd": "AVX_512",
            "avx512bw": "AVX_512",
            "avx512vl": "AVX_512",
        }

        def matches_keyword(vectorization, keywords):
            return any(re.search(keyword, vectorization, re.IGNORECASE) for keyword in keywords)

        detected_vectorizations = set()
        for sys_vec, mapped_vecs in system_map.items():
            if sys_vec in system_vectorizations:
                detected_vectorizations.update(
                    mapped_vecs if isinstance(mapped_vecs, list) else [mapped_vecs]
                )

        # Add special flags if supported
        detected_vectorizations.update(["auto", "none", "portable"])

        # Return vectorizations coupled with their build flags as a clean dictionary
        supported_vectorizations = {
            flag: combined_flags[flag]
            for flag in combined_flags.keys()
            & {mapped.lower() for mapped in detected_vectorizations}
        }

        return supported_vectorizations

    # comeplete
    def get_gpu_backend(self):
        if not self.specialization_points.get("gpu_build", {}).get("value", False):
            return {}

        # Collect GPU backends from specialization points
        specialization_backends = self.specialization_points.get("gpu_backends", {})

        # Collect GPU backends from system features
        system_backends = self.system_features.get("GPU Backends", {})
        loaded_modules = self.system_features.get("Loaded Modules", {})
        gpu_backends = {"cuda", "opencl", "hip", "sycl", "rocm", "openacc"}

        # Create refined output mapping backend names to build flags and versions
        backend_mapping = {}
        for backend, specs in specialization_backends.items():
            system_spec = system_backends.get(backend, {})
            libraries = system_spec.get("libraries", [])
            version = system_spec.get("version")

            # Check for backend in loaded modules
            module_version = loaded_modules.get(backend.lower())

            # Only include backends with non-empty libraries, non-null versions, or found in loaded modules
            if (libraries and version is not None) or module_version:
                backend_mapping[backend] = {
                    "build_flag": specs.get("build_flag", "None"),
                    "version": version or module_version,
                }

        return backend_mapping

    # compelete
    def get_parallel_libraries(self):
        # always returns openmp despite its used_by_default value
        # For mpi, regardless of its used_by_default value, it checks if one of the variants in on the system and return that
        # returns any other option that is not openmp and mpi
        specialization_libs = self.specialization_points.get("parallel_programming_libraries", {})
        system_libs = self.system_features.get("Parallel Libraries", {})
        known_mpi_variants = {"mpi", "mpich", "intel-oneapi-mpi", "openmpi", "roc-ompi/4.0.6rc4"}

        parallel_mapping = {}

        for lib_name, specs in specialization_libs.items():
            if lib_name.lower() == "openmp":
                parallel_mapping[lib_name] = {
                    "build_flag": specs.get("build_flag", "N/A"),
                    "version": system_libs.get(lib_name, {}).get("version", "Unknown"),
                    "used_as_default": specs.get("used_as_default", False),
                }
            elif lib_name.lower() == "mpi":
                installed_mpi_variant = next(
                    (mod for mod in known_mpi_variants if mod in system_libs), None
                )
                if installed_mpi_variant:
                    parallel_mapping[installed_mpi_variant] = {
                        "build_flag": specs.get("build_flag", "N/A"),
                        "used_as_default": specs.get("used_as_default", False),
                        "library_name": installed_mpi_variant,
                    }
            else:
                parallel_mapping[lib_name] = {
                    "build_flag": specs.get("build_flag", "N/A"),
                    "version": system_libs.get(lib_name, {}).get("version", "Unknown"),
                    "used_as_default": specs.get("used_as_default", False),
                }

        return parallel_mapping

    # complete
    def normalize_library_name(self, lib_name):
        normalized_name = re.sub(r"\[.*?\]|\s+", "", lib_name).lower()
        mapping = {
            "mkl": ["mkl", "MKL"],
            "mkl/onemkl": ["MKL", "oneMKL", "mkl", "onemkl"],
            "blas/lapack": ["BLAS", "LAPACK"],
            "openblas": ["OpenBLAS"],
            "cublas": ["cuBLAS"],
            "rocblas": ["rocBLAS"],
        }
        if normalized_name in mapping:
            return mapping[normalized_name]
        elif re.search(r"internal|built-in", normalized_name, re.IGNORECASE):
            return ["internal"]
        return [lib_name]

    def find_fft_libraries(self):
        # if intel-oneapi-mkl is on the system, list all MKL and oneAPI options as oneAPI is backward comptable
        # if intel-mkl is on the system, consider MKL options only
        # oneMath is not oneAPI (we don't check for oneMath)
        # if CUDA backend is found, that mean cuFFT exists

        build_fft_libs = self.specialization_points.get("FFT_libraries", {})
        system_fft_libs = self.system_features.get("FFT Libraries", {})
        # loaded_modules = self.system_features.get("Loaded Modules", {})

        # Normalize system FFT libraries
        normalized_system_libs = {}
        for k, v in system_fft_libs.items():
            for norm_name in self.normalize_library_name(k):
                normalized_system_libs[norm_name] = v

        # Detect installed MKL/oneMKL
        # oneapi_installed = "intel-oneapi-mkl" in loaded_modules
        # intel_mkl_installed = "intel-mkl" in loaded_modules

        fft_results = {}
        for lib_name, build_info in build_fft_libs.items():
            normalized_lib_names = self.normalize_library_name(lib_name)

            for normalized_lib_name in normalized_lib_names:
                if normalized_lib_name in normalized_system_libs or build_info.get(
                    "used_as_default", False
                ):
                    system_info = normalized_system_libs.get(normalized_lib_name, {})
                    version = system_info.get("version", "Unknown")
                    fft_results[lib_name] = {
                        "build_flag": build_info.get("build_flag", "N/A"),
                        "version": version,
                        "used_as_default": build_info.get("used_as_default", False),
                    }
                elif build_info.get("built-in", False):
                    fft_results[lib_name] = {
                        "build_flag": build_info.get("build_flag", "N/A"),
                        "version": "Built-in",
                        "used_as_default": build_info.get("used_as_default", False),
                    }

        # Check for CUDA in GPU backends and add cuFFT if present
        # Implicit inference of the existence
        gpu_backends = self.get_gpu_backend()
        if "CUDA" in gpu_backends:
            cufft_config = build_fft_libs.get("cuFFT", {})
            if cufft_config:
                fft_results["cuFFT"] = {
                    "build_flag": cufft_config.get("build_flag", "N/A"),
                    "version": cufft_config.get("version", "Unknown"),
                    "used_as_default": cufft_config.get("used_as_default", False),
                }

        return fft_results

    # complete
    def find_linear_algebra_libraries(self):
        build_la_libs = self.specialization_points.get("linear_algebra_libraries", {})
        system_la_libs = self.system_features.get("linear algebra Libraries", {})
        loaded_modules = self.system_features.get("Loaded Modules", {})
        gpu_backends = self.system_features.get("GPU Backends", {})

        la_results = {}

        for lib_name, build_info in build_la_libs.items():
            normalized_lib_names = self.normalize_library_name(lib_name)

            for normalized_lib_name in normalized_lib_names:
                # Detect MKL from module
                if normalized_lib_name in ["MKL", "oneMKL"]:
                    version = loaded_modules.get("intel-oneapi-mkl") or loaded_modules.get(
                        "intel-mkl"
                    )
                    if version:
                        la_results["MKL"] = {
                            "build_flag": build_info.get("build_flag", "N/A"),
                            "version": version,
                            "used_as_default": build_info.get("used_as_default", False),
                        }

                    for sys_lib_name, system_info in system_la_libs.items():
                        if sys_lib_name.lower() == "mkl":
                            la_results["MKL"] = {
                                "build_flag": build_info.get("build_flag", "N/A"),
                                "version": version,
                                "used_as_default": build_info.get("used_as_default", False),
                            }

                # Detect cuBLAS from CUDA backend info
                elif normalized_lib_name.lower() == "cublas":
                    cuda_info = gpu_backends.get("CUDA", {})
                    if cuda_info and cuda_info.get("version"):
                        la_results["cuBLAS"] = {
                            "build_flag": build_info.get("build_flag", "N/A"),
                            "version": cuda_info["version"],
                            "used_as_default": build_info.get("used_as_default", False),
                        }

                # Detect OpenBLAS from system features
                elif normalized_lib_name.lower() == "openblas":
                    for sys_lib_name, system_info in system_la_libs.items():
                        if sys_lib_name.lower() == "openblas":
                            la_results["OpenBLAS"] = {
                                "build_flag": build_info.get("build_flag", "N/A"),
                                "version": system_info.get("version", "Unknown"),
                                "used_as_default": build_info.get("used_as_default", False),
                            }

                    # Fallback: Check loaded modules for OpenBLAS if not already added
                    if "OpenBLAS" not in la_results and "openblas" in loaded_modules:
                        la_results["OpenBLAS"] = {
                            "build_flag": build_info.get("build_flag", "N/A"),
                            "version": loaded_modules["openblas"],
                            "used_as_default": build_info.get("used_as_default", False),
                        }

                # Detect other known libraries
                elif normalized_lib_name in ["BLAS", "NVPL", "BLIS", "LAPACK"]:
                    for sys_lib_name, system_info in system_la_libs.items():
                        if sys_lib_name.lower() == normalized_lib_name.lower():
                            la_results[sys_lib_name] = {
                                "build_flag": build_info.get("build_flag", "N/A"),
                                "version": system_info.get("version", "Unknown"),
                                "used_as_default": build_info.get("used_as_default", False),
                            }

                # Internal fallback
                elif normalized_lib_name == "internal":
                    la_results[lib_name] = {
                        "build_flag": build_info.get("build_flag", "N/A"),
                        "version": "Built-in",
                        "used_as_default": build_info.get("used_as_default", False),
                    }

        return la_results

    # complete
    def perform_check(self, app_language: Language):
        return {
            "vectorization_flags": self.get_vectorization_flags(),
            "gpu_backends": self.get_gpu_backend(),
            "parallel_libraries": self.get_parallel_libraries(),
            "fft_libraries": self.find_fft_libraries(),
            "linear_algebra_libraries": self.find_linear_algebra_libraries(),
            "compiler": self.map_compiler(app_language),
        }

    def map_compiler(self, app_language: Language) -> dict:
        # special handling for compilers
        # these are not specified by the app
        system_compilers = self.system_features["Compilers"][app_language.value]

        # currently, all compilers are available in PATH - no need to find paths
        # we map compiler name to itself
        # if we ever need to support custom paths, then we need to change it here
        return {
            key: {
                "build_flag": None,
                "version": "default",
                "used_as_default": "False",
                "language": app_language.value,
            }
            for key, _ in system_compilers.items()
        }

    def find_mandatory_installations(self):
        # returns all the libraries where used_as_default is True
        external_libs = self.specialization_points.get("other_external_libraries", {})
        mandatory_libs = {
            lib_name: details
            for lib_name, details in external_libs.items()
            if details.get("used_as_default", False)
        }
        return mandatory_libs

    def get_optimization_build_flags(self):
        return self.specialization_points.get("optimization_build_flags", [])
