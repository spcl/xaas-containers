import os


class DockerfileCreator:
    def __init__(
        self,
        project_directory,
        selected_specializations,
        system_features,
        build_command,
        base_image=None,
        output_file="Dockerfile",
    ):

        self.project_directory = project_directory
        self.selected_specializations = selected_specializations
        self.system_features = system_features
        self.build_command = build_command
        self.output_file = output_file
        self.dockerfile_content = []
        self.base_image = base_image

        self.architecture = self.extract_architecture()
        self.vectorization_flags = self.extract_vectorization_flags()

    def extract_architecture(self):
        return self.system_features.get("CPU Info", {}).get("Architecture", "unknown")

    def extract_vectorization_flags(self):
        return self.system_features.get("CPU Info", {}).get("Supported Vectorizations", [])

    def add_base_image(self):

        if self.architecture == "x86_64":
            base_image = "ealnuaimi/xaas:ubuntu20.04-mpich3.1.4-v1.1"
        elif self.architecture == "aarch64":
            base_image = "ealnuaimi/xaas:ubuntu22.04-mpich3.4-arm-gnu13"
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        base_image_content = f"""
        FROM {base_image} AS SOURCE 
        # Use Bash as the default shell for all RUN commands
        SHELL ["/bin/bash", "-c"]
        """
        self.dockerfile_content.append(base_image_content.strip())

    def add_multistage_phase(self):

        if self.base_image is not None:
            base_image_content = f"""
            FROM {self.base_image} AS DEPLOYMENT 
            # Use Bash as the default shell for all RUN commands
            SHELL ["/bin/bash", "-c"]
            """
            self.dockerfile_content.append(base_image_content.strip())

    def process_specializations(self):
        if self.selected_specializations.get("gpu_backends"):
            self.install_gpu_backend()
        if self.selected_specializations.get("linear_algebra_libraries"):
            self.install_linear_algebra_lib()
        if self.selected_specializations.get("fft_libraries"):
            self.install_fft_lib()

    def install_gpu_backend(self):

        gpu_backends = self.selected_specializations.get("gpu_backends", {})

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

                # CUDA installation commands (full version)
                cuda_install_commands = f"""
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
                """

                self.dockerfile_content.append(cuda_install_commands.strip())

            elif backend.lower in ["hip", "rocm"]:
                rocm_version = config.get("version")
                if not rocm_version:
                    raise ValueError(
                        "ROCm version is required but missing in selected_specializations."
                    )

                # ROCm installation commands
                rocm_install_commands = f"""
                # Install ROCm stack
                RUN apt update && apt install -y wget gnupg && \\
                    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \\
                    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/{rocm_version}/ ubuntu main" > /etc/apt/sources.list.d/rocm.list && \\
                    apt update && \\
                    apt install -y rocm-dev rocblas rocfft rocm-llvm rocprim-dev && \\
                rm -rf /var/lib/apt/lists/*
            """

                self.dockerfile_content.append(rocm_install_commands.strip())

    def install_fft_lib(self):

        # mkl source command might be incomplete

        fft_libraries = self.selected_specializations.get("fft_libraries", {})

        # Prioritize MKL if present
        for fft_lib in fft_libraries.keys():
            if fft_lib in ["MKL (GPU)", "mkl (CPU)"]:
                mkl_install_commands = """
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
                """
                self.dockerfile_content.append(mkl_install_commands.strip())
                return  # Stop here, don't install FFTW if MKL is chosen

        # Install rocFFT if explicitly selected
        for fft_lib in fft_libraries.keys():
            if fft_lib.lower() == "rocfft":
                rocfft_install_commands = """
                # Install rocFFT and dependencies
                RUN apt update && \\
                    apt install -y rocfft && \\
                    rm -rf /var/lib/apt/lists/*
                """
                self.dockerfile_content.append(rocfft_install_commands.strip())
                return  # Only one FFT library should be installed

        # If MKL is not selected, check for FFTW
        for fft_lib, config in fft_libraries.items():
            if fft_lib in ["FFTW", "FFTW3"]:
                if config.get("used_as_default", False):
                    print(f"Skipping installation of {fft_lib} (used as default).")
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
                    if "avx512" in self.vectorization_flags:
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

    def install_linear_algebra_lib(self):
        linear_algebra_libs = self.selected_specializations.get("linear_algebra_libraries", {})

        for lib in linear_algebra_libs.keys():
            if lib == "OpenBLAS":
                openblas_install_commands = """
                RUN apt update
                RUN apt install -y libopenblas-dev
                """
                self.dockerfile_content.append(openblas_install_commands.strip())
            elif lib == "ScaLAPACK":
                scalapack_install_commands = """
                RUN apt update
                RUN apt install -y libscalapack-mpi-dev
                """
                self.dockerfile_content.append(scalapack_install_commands.strip())

    def copy_project_directory(self):
        app_name = os.path.basename(os.path.normpath(self.project_directory))
        content = f"""
        COPY {self.project_directory} /{app_name}
        WORKDIR /{app_name}
        """
        self.dockerfile_content.append(content.strip())

    def copy_from_first_build_stage(self):
        app_name = os.path.basename(os.path.normpath(self.project_directory))

        content = f"""
        COPY --from=SOURCE /{app_name} /{app_name}
        WORKDIR /{app_name}
        """
        self.dockerfile_content.append(content.strip())

    def application_build_command(self):
        self.dockerfile_content.append(self.build_command.strip())

    def add_default_command(self):
        self.dockerfile_content.append("# Default command (modify if needed)")
        self.dockerfile_content.append('CMD ["/bin/bash"]')

    def create_dockerfile(self):

        # source container
        self.add_base_image()
        self.copy_project_directory()

        # multi-stage build (deployment))

        self.add_multistage_phase()
        self.process_specializations()
        self.copy_from_first_build_stage()
        self.application_build_command()
        self.add_default_command()

        # with open(self.output_file, "w") as file:
        # file.write("\n".join(self.dockerfile_content))

        # print(f"Dockerfile created at: {self.output_file}")
        print("Dockerfile content:")
        print("\n".join(self.dockerfile_content))
