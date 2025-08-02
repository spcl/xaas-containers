import os
import logging
import inspect

from xaas.config import SourceDeploymentConfigBaseImage


def install_mkl() -> str:
    return inspect.cleandoc("""
    # Update package list and install dependencies
    RUN apt update && apt install -y gpg-agent wget
    # Add Intel oneAPI GPG key
    RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
    # Add Intel oneAPI repository
    RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list
    # Update package list again
    RUN apt update
    # Install Intel MKL
    RUN apt install -y intel-oneapi-mkl intel-oneapi-mkl-devel
    # mkl has to be sourced to be recognized by gromacs
    RUN source /opt/intel/oneapi/mkl/latest/env/vars.sh
    """)


def install_cuda(cuda_version: str, repo_url: str) -> str:
    return inspect.cleandoc(f"""
    # Add CUDA repository key
    RUN apt-key adv --fetch-keys {repo_url}3bf863cc.pub \\
        && echo "deb {repo_url} /" > /etc/apt/sources.list.d/cuda.list \\
        && apt-get update \\
        && apt-get install -y --no-install-recommends \\
            cuda-toolkit-{cuda_version}

    # Set CUDA environment variables
    ENV PATH="/usr/local/cuda-{cuda_version}/bin:$PATH"
    ENV LD_LIBRARY_PATH="/usr/local/cuda-{cuda_version}/lib64:$LD_LIBRARY_PATH"

    # NVIDIA runtime environment variables
    ENV NVIDIA_VISIBLE_DEVICES all
    ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
    ENV NVIDIA_REQUIRE_CUDA "cuda>={cuda_version}"
    """)


def install_rocm(rocm_version: str) -> str:
    return inspect.cleandoc(f"""
    # Install ROCm stack
    RUN apt update && apt install -y wget gnupg && \\
        wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \\
        echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/{rocm_version}/ ubuntu main" > /etc/apt/sources.list.d/rocm.list && \\
        apt update && \\
        apt install -y rocm-dev rocblas rocfft rocm-llvm rocprim-dev && \\
    rm -rf /var/lib/apt/lists/*
    """)


def install_rocfft(rocm_version: str) -> str:
    return inspect.cleandoc("""
    # Install rocFFT and dependencies
    RUN apt update && \\
        apt install -y rocfft && \\
        rm -rf /var/lib/apt/lists/*
    """)


def install_openblas() -> str:
    return inspect.cleandoc("""
    RUN apt update
    RUN apt install -y libopenblas-dev
    """)


def install_scalapack() -> str:
    return inspect.cleandoc("""
    RUN apt update
    RUN apt install -y libscalapack-mpi-dev
    """)


class DockerfileCreator:
    def __init__(self, project_name: str, working_directory: str, cpu_architecture: str):
        self.project_name = project_name
        self.working_directory = working_directory
        self.dockerfile_content = []

        self.architecture = cpu_architecture

    def create_source_dockerfile(self, project_directory: str) -> str:
        # source container
        self.add_base_image()
        self.copy_project_directory(project_directory)
        self.dockerfile_content.append("")

        os.makedirs(self.working_directory, exist_ok=True)
        output_file = os.path.join(self.working_directory, "Dockerfile.source")

        with open(output_file, "w") as file:
            file.write("\n".join(self.dockerfile_content))

        logging.info(f"Dockerfile created at: {output_file}")

        return output_file

    @staticmethod
    def extract_vectorization_flags(system_features: dict) -> list:
        return system_features.get("CPU Info", {}).get("Supported Vectorizations", [])

    def add_base_image(self):
        if self.architecture == "x86_64":
            base_image = "spcleth/xaas:source-base-x86-24.04"
        elif self.architecture == "aarch64":
            # FIXME: change
            base_image = "ealnuaimi/xaas:ubuntu22.04-mpich3.4-arm-gnu13"
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        base_image_content = f"""
        FROM {base_image} AS SOURCE 
        """
        self.dockerfile_content.append(base_image_content.strip())

    def add_multistage_phase(self, source_image: str, base_image: str | None):
        base_image_content = f"""
        FROM {source_image} AS SOURCE
        """
        self.dockerfile_content.append(inspect.cleandoc(base_image_content.strip()))

        if base_image is not None:
            base_image_content = f"""
            FROM {base_image} AS DEPLOYMENT 
            # Use Bash as the default shell for all RUN commands
            SHELL ["/bin/bash", "-c"]
            """
            self.dockerfile_content.append(inspect.cleandoc(base_image_content))

            """
                We always install CMake since many images have an old version.
                We cannot control the necessary cmake version for every application.

                FIXME: this could potentially be another feature provided by the base image.
            """
            self.dockerfile_content.append(
                """
ARG CMAKE_VERSION=3.31.5
RUN apt update && apt install -y wget tar && \\
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \\
tar -xzf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz --strip-components=1 -C /usr/local && \\
rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz
"""
            )

        else:
            self.dockerfile_content.append('SHELL ["/bin/bash", "-c"]')

    def process_specializations(
        self, selected_specializations: dict, system_features: dict, provided_features: list[str]
    ):
        if selected_specializations.get("gpu_backends"):
            self.install_gpu_backend(selected_specializations, provided_features)
        if selected_specializations.get("linear_algebra_libraries"):
            self.install_linear_algebra_lib(selected_specializations, provided_features)
        if selected_specializations.get("fft_libraries"):
            self.install_fft_lib(selected_specializations, system_features, provided_features)

    def install_gpu_backend(self, selected_specializations: dict, provided_features: list[str]):
        # FIXME: move this into separate dependencies
        gpu_backends = selected_specializations.get("gpu_backends", {})

        for backend, config in gpu_backends.items():
            if backend.lower() == "cuda":  # Case-insensitive check for CUDA
                cuda_version = config.get("version")

                if not cuda_version:
                    raise ValueError(
                        "CUDA version is required but missing in selected_specializations."
                    )

                # Architecture mapping
                arch_map = {
                    "x86_64": "x86_64",
                    "aarch64": "sbsa",
                    "arm64": "sbsa",
                    "ppc64le": "ppc64le",
                }
                arch_key = arch_map.get(self.architecture, "x86_64")  # Default to x86_64 if unknown
                repo_url = f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/{arch_key}/"

                self.dockerfile_content.append(install_cuda(cuda_version, repo_url).strip())

            elif backend.lower() in ["hip", "rocm"]:
                rocm_version = config.get("version")
                if not rocm_version:
                    raise ValueError(
                        "ROCm version is required but missing in selected_specializations."
                    )

                self.dockerfile_content.append(install_rocm(rocm_version).strip())
            elif backend.lower() == "sycl":
                if backend.lower() in provided_features:
                    logging.debug("Using existing SYCL installation.")
                    continue

                raise NotImplementedError("SYCL installation is not yet supported.")

    def install_fft_lib(
        self, selected_specializations: dict, system_features: dict, provided: list[str]
    ):
        # FIXME: move this into separate dependencies

        fft_libraries = selected_specializations.get("fft_libraries", {})

        # Prioritize MKL if present
        for fft_lib in fft_libraries:
            if fft_lib in ["MKL", "mkl"]:
                if "mkl" not in provided:
                    self.dockerfile_content.append(install_mkl().strip())
                # Stop here, don't install FFTW if MKL is chosen
                return

        # Install rocFFT if explicitly selected
        for fft_lib in fft_libraries:
            if fft_lib.lower() == "rocfft":
                self.dockerfile_content.append(install_rocfft.strip())
                # Only one FFT library should be installed
                return

        # If MKL is not selected, check for FFTW
        for fft_lib, config in fft_libraries.items():
            if fft_lib.lower() in ["fftw", "fftw3"]:
                if config.get("used_as_default", False):
                    logging.info(f"Skipping installation of {fft_lib} (used as default).")
                    return

                # Download and extract FFTW
                fftw_install_commands = """
                # Download and extract FFTW
                RUN wget http://www.fftw.org/fftw-3.3.10.tar.gz \\
                    && tar -xvzf fftw-3.3.10.tar.gz \\
                    && cd fftw-3.3.10
                """

                # Base configure command
                configure_command = """
                RUN cd fftw-3.3.10 \\
                    && ./configure --prefix=/usr/local \\
                        --enable-float \\
                        --enable-shared
                """

                # Add architecture-specific flags
                if self.architecture.startswith("x86"):
                    configure_command += (
                        " \\\n                    --enable-sse2 --enable-avx --enable-avx2"
                    )
                    if "avx512" in self.extract_vectorization_flags(system_features):
                        configure_command += " --enable-avx512"
                elif self.architecture in ["aarch64", "arm64"]:
                    configure_command += " \\\n                    --enable-neon"
                elif self.architecture.startswith("ppc"):
                    configure_command += " \\\n                    --enable-vsx"

                # Build and install FFTW
                build_commands = """
                RUN cd fftw-3.3.10 \\
                    && make -j$(nproc) \\
                    && make install
                """

                # Append all commands to Dockerfile
                self.dockerfile_content.append(fftw_install_commands.strip())
                self.dockerfile_content.append(configure_command.strip())
                self.dockerfile_content.append(build_commands.strip())
                return  # Stop after installing FFTW (ensures only one FFT library)

    def install_linear_algebra_lib(
        self, selected_specializations: dict, provided_features: list[str]
    ):
        # FIXME: move this into separate dependencies
        linear_algebra_libs = selected_specializations.get("linear_algebra_libraries", {})
        fft_libraries = selected_specializations.get("fft_libraries", {})

        # True if MKL (GPU) or mkl (CPU) present in FFT OR Linear Algebra libs
        mkl_already_installed = any(lib in ["MKL", "mkl"] for lib in list(fft_libraries.keys()))

        # Only install MKL if requested, but NOT already installed by FFT/other
        if "MKL" in linear_algebra_libs and not mkl_already_installed:
            self.dockerfile_content.append(install_mkl().strip())
            return

        # Otherwise, install other linear algebra libraries as usual
        for lib in linear_algebra_libs:
            if lib == "OpenBLAS":
                self.dockerfile_content.append(install_openblas().strip())
            elif lib == "ScaLAPACK":
                self.dockerfile_content.append(install_scalapack().strip())

    def copy_project_directory(self, project_directory):
        content = inspect.cleandoc(
            f"""
            COPY {project_directory} /source
            WORKDIR /source
        """
        )
        self.dockerfile_content.append(content.strip())

    def copy_from_first_build_stage(self, has_different_build_image: bool):
        if has_different_build_image:
            content = inspect.cleandoc("""
            COPY --link --from=SOURCE /source /source
            WORKDIR /source
            """)
            self.dockerfile_content.append(content.strip())

    def application_build_command(self, build_command: str):
        self.dockerfile_content.append(build_command.strip())

    def add_default_command(self):
        self.dockerfile_content.append("# Default command (modify if needed)")
        self.dockerfile_content.append('CMD ["/bin/bash"]')

    def add_build_args(self):
        self.dockerfile_content.append("ARG nproc")

    def create_deployment_dockerfile(
        self,
        selected_specializations,
        system_features,
        build_command: str,
        source_image: str,
        dockerfile_name: str,
        deployment_base_image: SourceDeploymentConfigBaseImage | None,
    ):
        self.add_multistage_phase(
            source_image, deployment_base_image.name if deployment_base_image else None
        )
        self.add_build_args()
        self.process_specializations(
            selected_specializations,
            system_features,
            deployment_base_image.provided_features if deployment_base_image else None,
        )
        self.copy_from_first_build_stage(deployment_base_image is not None)
        self.application_build_command(build_command)
        self.add_default_command()

        os.makedirs(self.working_directory, exist_ok=True)

        with open(dockerfile_name, "w") as file:
            file.write("\n".join(self.dockerfile_content))

        logging.info(f"Dockerfile created at: {dockerfile_name}")
