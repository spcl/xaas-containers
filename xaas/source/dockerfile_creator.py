import os
import logging
import inspect

from xaas.config import SourceDeploymentConfigBaseImage
from xaas.config import FeatureType, CPUArchitecture
from xaas.config import XaaSConfig


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
    def __init__(
        self, project_name: str, working_directory: str, cpu_architecture: CPUArchitecture
    ):
        self.project_name = project_name
        self.working_directory = working_directory
        self.init_dockerfile_content = []
        self.dockerfile_content = []

        self.architecture = cpu_architecture

        self.env_configurations = []
        self.env_values = []

        # all features enabled by the user.
        self.all_features: list[FeatureType] = []

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
            base_image = "docker.io/spcleth/xaas:source-base-x86-24.04"
        elif self.architecture == "arm64":
            base_image = "docker.io/spcleth/xaas:source-base-arm-24.04"
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        base_image_content = f"""
        FROM {base_image} AS SOURCE 
        """
        self.dockerfile_content.append(base_image_content.strip())

    def add_multistage_phase(self, source_image: str, base_image: str | None):
        base_image_content = f"""
        FROM docker.io/{source_image} AS SOURCE
        """
        self.dockerfile_content.append(inspect.cleandoc(base_image_content.strip()))

        if base_image is not None:
            base_image_content = f"""
            FROM docker.io/{base_image} AS DEPLOYMENT
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
        self,
        selected_specializations: dict,
        system_features: dict,
        provided_features: list[FeatureType],
    ) -> list[tuple[FeatureType, str]]:
        features_requested: list[tuple[FeatureType, str]] = []

        if selected_specializations.get("gpu_backends"):
            features, req_features = self.install_gpu_backend(
                selected_specializations, provided_features
            )
            self.all_features.extend(features)
            features_requested.extend(req_features)
        if selected_specializations.get("linear_algebra_libraries"):
            features, req_features = self.install_linear_algebra_lib(
                selected_specializations, provided_features
            )
            self.all_features.extend(features)
            features_requested.extend(req_features)
        if selected_specializations.get("fft_libraries"):
            features, req_features = self.install_fft_lib(
                selected_specializations, system_features, provided_features
            )
            self.all_features.extend(features)
            features_requested.extend(req_features)

        return features_requested

    def install_gpu_backend(
        self, selected_specializations: dict, provided_features: list[FeatureType]
    ) -> tuple[list[FeatureType], list[tuple[FeatureType, str]]]:
        gpu_backends = selected_specializations.get("gpu_backends", {})

        for backend, config in gpu_backends.items():
            if backend.lower() == "cuda":
                cuda_version = config.get("version")

                if not cuda_version:
                    raise ValueError(
                        "CUDA version is required but missing in selected_specializations."
                    )

                if (
                    cuda_version
                    in XaaSConfig().layers.layers[self.architecture][FeatureType.CUDA].versions
                ):
                    logging.debug(f"Using Docker layer for CUDA version {cuda_version}.")

                    return ([FeatureType.CUDA], [(FeatureType.CUDA, cuda_version)])
                else:
                    logging.debug(f"No Docker layer for CUDA version {cuda_version}, install.")

                    # Architecture mapping
                    arch_map = {
                        "x86_64": "x86_64",
                        "aarch64": "sbsa",
                        "arm64": "sbsa",
                        "ppc64le": "ppc64le",
                    }
                    arch_key = arch_map.get(self.architecture, "x86_64")
                    repo_url = f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/{arch_key}/"

                    self.dockerfile_content.append(install_cuda(cuda_version, repo_url).strip())

                    return ([FeatureType.CUDA], [])

            elif backend.lower() in ["hip", "rocm"]:
                rocm_version = config.get("version")
                if not rocm_version:
                    raise ValueError(
                        "ROCm version is required but missing in selected_specializations."
                    )

                self.dockerfile_content.append(install_rocm(rocm_version).strip())
                return ([FeatureType.ROCM], [])

            elif backend.lower() == "sycl":
                if FeatureType.SYCL in provided_features:
                    logging.debug("Using existing SYCL installation.")
                    continue

                raise NotImplementedError(
                    "SYCL installation is not supported - it must be requested explicitly."
                )

        return ([], [])

    def install_fft_lib(
        self,
        selected_specializations: dict,
        system_features: dict,
        provided_features: list[FeatureType],
    ) -> tuple[list[FeatureType], list[tuple[FeatureType, str]]]:
        fft_libraries = selected_specializations.get("fft_libraries", {})

        for fft_lib in fft_libraries:
            # support both CPU and GPU versions
            if fft_lib in ["MKL", "mkl"]:
                if (
                    FeatureType.ONEAPI in self.all_features
                    or FeatureType.ONEAPI in provided_features
                ):
                    logging.debug("Using existing oneAPI installation for MKL.")
                    self.env_configurations.append(". /opt/intel/oneapi/setvars.sh --force")
                    return ([], [])

                mkl_version = fft_libraries["MKL"]["version"]

                if (
                    mkl_version
                    in XaaSConfig().layers.layers[self.architecture][FeatureType.ONEAPI].versions
                ):
                    logging.debug(f"Using Docker layer for oneAPI-MKL version {mkl_version}.")

                    self.env_configurations.append(". /opt/intel/oneapi/setvars.sh --force")
                    return ([FeatureType.ONEAPI], [(FeatureType.ONEAPI, mkl_version)])
                else:
                    logging.debug(f"Installing oneAPI-MKL version {mkl_version}.")
                    self.dockerfile_content.append(install_mkl().strip())
                    self.env_configurations.append(". /opt/intel/oneapi/setvars.sh --force")
                    return ([FeatureType.ONEAPI], [])

        # Install rocFFT if explicitly selected
        for fft_lib in fft_libraries:
            if fft_lib.lower() == "rocfft":
                self.dockerfile_content.append(install_rocfft.strip())
                return ([FeatureType.ROCFFT], [])

        # If MKL is not selected, check for FFTW
        for fft_lib, config in fft_libraries.items():
            if fft_lib.lower() in ["fftw", "fftw3"]:
                if config.get("used_as_default", False):
                    logging.info(f"Skipping installation of {fft_lib} (used as default).")
                    return ([FeatureType.FFTW3], [])

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
                    configure_command += " \\\n --enable-sse2 --enable-avx --enable-avx2"
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

                self.dockerfile_content.append(fftw_install_commands.strip())
                self.dockerfile_content.append(configure_command.strip())
                self.dockerfile_content.append(build_commands.strip())
                return ([FeatureType.FFTW3], [])

        return ([], [])

    def install_linear_algebra_lib(
        self, selected_specializations: dict, provided_features: list[str]
    ) -> tuple[list[FeatureType], list[tuple[FeatureType, str]]]:
        linear_algebra_libs = selected_specializations.get("linear_algebra_libraries", {})

        # Only install MKL if requested, but NOT already installed by FFT/other
        if "MKL" in linear_algebra_libs:
            if FeatureType.ONEAPI in self.all_features or FeatureType.ONEAPI in provided_features:
                logging.debug("Using existing oneAPI installation for MKL.")
                self.env_values.append('XAAS_BLAS_PATH="${MKLROOT}"')
                self.env_configurations.append(". /opt/intel/oneapi/setvars.sh --force")
                return ([], [])

            mkl_version = linear_algebra_libs["MKL"]["version"]
            if (
                mkl_version
                in XaaSConfig().layers.layers[self.architecture][FeatureType.ONEAPI].versions
            ):
                logging.debug(f"Using Docker layer for oneAPI-MKL version {mkl_version}.")

                self.env_configurations.append(". /opt/intel/oneapi/setvars.sh --force")
                self.env_values.append('XAAS_BLAS_PATH="${MKLROOT}"')
                return ([FeatureType.ONEAPI], [(FeatureType.ONEAPI, mkl_version)])
            else:
                logging.debug(f"Installing oneAPI-MKL version {mkl_version}.")
                self.dockerfile_content.append(install_mkl().strip())
                self.env_configurations.append(". /opt/intel/oneapi/setvars.sh --force")
                self.env_values.append('XAAS_BLAS_PATH="${MKLROOT}"')
                return ([FeatureType.ONEAPI], [])

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
            COPY --from=SOURCE /source /source
            WORKDIR /source
            """)
            self.dockerfile_content.append(content.strip())

    def application_build_command(self, build_command: str):
        # define all envs before we process
        cmd = build_command.strip()

        if len(self.env_values) > 0:
            for env in self.env_values:
                cmd = f" export {env} && {cmd}"

        if len(self.env_configurations) > 0:
            for env in self.env_configurations:
                cmd = f"{env} && {cmd}"

        cmd = f"RUN {cmd}"

        self.dockerfile_content.append(cmd)

    def add_default_command(self):
        self.dockerfile_content.append("")
        self.dockerfile_content.append('CMD ["/bin/bash"]')
        self.dockerfile_content.append("")

    def add_build_args(self):
        self.dockerfile_content.append("")
        self.dockerfile_content.append("ARG nproc")
        self.dockerfile_content.append("")

    def copy_from_layers(self, features_requested: list[tuple[FeatureType, str]]):
        envs = []
        for layer_type, version in features_requested:
            layer = XaaSConfig().layers.layers[self.architecture][layer_type]

            layer_name = layer.name.replace(f"${{{layer.version_arg}}}", version)
            layer_build_location = layer.build_location.replace(
                f"${{{layer.version_arg}}}", version
            )

            self.init_dockerfile_content.append(
                f"FROM docker.io/{XaaSConfig().docker_repository}:{layer_name} as {layer_name}-layer"
            )
            self.dockerfile_content.append(
                f"COPY --from={layer_name}-layer {layer_build_location} {layer_build_location}"
            )
            for name, env in layer.envs.items():
                env = env.replace(f"${{{layer.version_arg}}}", version)
                envs.append(f'ENV {name}="{env}:${name}"')

        self.dockerfile_content.append("")
        self.dockerfile_content.extend(envs)

    def handle_gpu_hooks(self):
        # Container hooks for NVIDIA GPUs require that we define
        # certain environment variables. Otherwise, the hook might not be enabled,
        # and GPUs won't be available in the container.
        #
        # We tried to enable it at runtime by adding those envs to srun on CSCS Alps,
        # but this failed.
        if FeatureType.CUDA in self.all_features:
            self.dockerfile_content.append("ENV NVIDIA_VISIBLE_DEVICES all")
            self.dockerfile_content.append("ENV NVIDIA_DRIVER_CAPABILITIES compute,utility")

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
        features_requested = self.process_specializations(
            selected_specializations,
            system_features,
            deployment_base_image.provided_features if deployment_base_image else [],
        )
        self.copy_from_first_build_stage(deployment_base_image is not None)
        self.copy_from_layers(features_requested)
        self.application_build_command(build_command)
        self.add_default_command()
        self.handle_gpu_hooks()

        # mkl has to be sourced to be recognized by gromacs
        # RUN source /opt/intel/oneapi/mkl/latest/env/vars.sh
        os.makedirs(self.working_directory, exist_ok=True)

        with open(dockerfile_name, "w") as file:
            file.write("\n".join(self.init_dockerfile_content))
            file.write("\n")
            file.write("\n".join(self.dockerfile_content))

        logging.info(f"Dockerfile created at: {dockerfile_name}")
