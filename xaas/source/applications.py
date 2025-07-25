import logging
from enum import Enum
from typing import Callable

from xaas.source.gemini_interface import GeminiInterface
import xaas.source.utils as utils


class Application(str, Enum):

    GROMACS = "gromacs"
    MILC = "milc"
    OPENQCD = "openqcd"
    Q_E = "q-e"
    VPIC_KOKKOS = "vpic-kokkos"
    LLAMMA_CPP = "llama.cpp"
    ICON = "icon-model"
    CLOUDSC = "cloudsc"


class ApplicationSpecialization:

    def __init__(self, system_features: dict, gemini_interface: GeminiInterface | None):

        self._gemini_interface = gemini_interface
        self._system_features = system_features

    def gromacs(self, selected_specializations: dict, specialization_points: dict) -> str:
        release_build = "-DCMAKE_BUILD_TYPE=Release "
        build_flags_string = utils.extract_build_flags(
            selected_specializations, specialization_points
        )
        build_flags_string = release_build + build_flags_string

        if (
            "fft_libraries" in selected_specializations
            and "fftw3" in selected_specializations["fft_libraries"]
        ):
            build_flags_string = f" {build_flags_string} -DGMX_BUILD_OWN_FFTW=ON"

        return f"""
            RUN mkdir build \\
                && cd build \\
                && cmake .. {build_flags_string} \\
                && make -j$(nproc) \\
                && make check \\
                && sudo make install \\
                && source /usr/local/gromacs/bin/GMXRC \\
                && cd ../
            """

    def milc(self, selected_specializations: dict, specialization_points: dict) -> str:
        assert self._gemini_interface is not None, "Gemini interface is not initialized."
        # FIXME: this steep needs to be added to the container build. we are missing source dir
        self._gemini_interface.edit_makefile(selected_specializations, "")
        # FIXME: Make this configurable
        milc_application_name = "su3_rmd"
        return f"""
        RUN cd {milc_application_name} \\
        && $cp ../Makefile . \\ 
        && make {milc_application_name}
        """

    def openqcd(self, selected_specializations: dict, specialization_points: dict) -> str:
        assert self._gemini_interface is not None, "Gemini interface is not initialized."
        # FIXME: this steep needs to be added to the container build. we are missing source dir
        self._gemini_interface.edit_makefile(selected_specializations, "")
        return """
        ENV MPI_INCLUDE=/usr/local/mpich/include
        ENV MPI_HOME=/usr/local/mpich/lib 
        RUN cd main && make
        """

    def q_e(self, selected_specializations: dict, specialization_points: dict) -> str:
        build_flags_string = utils.extract_build_flags(
            selected_specializations, specialization_points
        )
        return f"""
            RUN mkdir build \\
                && cd build \\
                && cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 {build_flags_string} \\
                && make -j$(nproc) \\
                && cd ../
            """

    def vpic_kokkos(self, selected_specializations: dict, specialization_points: dict) -> str:
        cpu_arch, gpu_arch = utils.get_kokkos_arch(self._system_features)
        release_build = '-DCMAKE_BUILD_TYPE=Release  -DCMAKE_CXX_FLAGS="-rdynamic" '
        build_flags_string = utils.extract_build_flags(
            selected_specializations, specialization_points
        )

        if "CUDA" in selected_specializations.get("gpu_backends", {}):
            build_flags_string += " -DCMAKE_CXX_COMPILER=/vpic-kokkos/kokkos/bin/nvcc_wrapper "
        if cpu_arch:
            build_flags_string += f" {cpu_arch} "
        if gpu_arch:
            build_flags_string += f" {gpu_arch} "

        return f"""
            RUN mkdir build \\
                && cd build \\
                && cmake .. {release_build + build_flags_string} \\
                && make -j$(nproc) \\
                && make install \\
                && cd ../
            """

    def llamma_cpp(self, selected_specializations: dict, specialization_points: dict) -> str:
        # have to specify the blas vendor -DGGML_BLAS_VENDOR=OpenBLAS

        # for perfomance portability
        default_flag = "-DGGML_NATIVE=OFF "

        build_flags_string = utils.extract_build_flags(
            selected_specializations, specialization_points
        )
        logging.debug(f"Generated CMake build flags: {build_flags_string}")

        if "CUDA" in selected_specializations.get("gpu_backends", {}):
            cuda_arch_flag = utils.get_cuda_architecture_flag(self._system_features)
            build_flags_string += " -DGGML_BACKEND_DL=OFF "
            build_flags_string += f"{cuda_arch_flag}"

        if "MKL" in selected_specializations.get("linear_algebra_libraries", {}):
            build_flags_string += "  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx  -DGGML_BLAS_VENDOR=Intel10_64lp"
        elif "OpenBLAS" in selected_specializations.get("linear_algebra_libraries", {}):
            build_flags_string += "  -DGGML_BLAS_VENDOR=OpenBLAS"
        elif "cuBLAS" in selected_specializations.get("linear_algebra_libraries", {}):
            pass

        return f"""
            RUN cmake -B build {build_flags_string} \\
                && cmake --build build --config Release -j $(nproc)
            """

    def icon(self, selected_specializations: dict, specialization_points: dict) -> str:
        # FIXME: This is not functional
        # Not fully tested
        build_flags_string = utils.extract_build_flags(
            selected_specializations, specialization_points
        )
        return f"""
            RUN ./configure {build_flags_string}
            """

    def cloudsc(self, selected_specializations: dict, specialization_points: dict) -> str:
        # FIXME: This is not functional
        # Not fully tested
        build_flags_string = utils.extract_build_flags(
            selected_specializations, specialization_points
        )
        return f"""
            RUN ./cloudsc-bundle create \\
                && ./cloudsc-bundle build --build-type release --cloudsc-fortran=ON --cloudsc-c=ON --with-serialbox {build_flags_string}
            """


class ApplicationSpecializationBuilder:

    APPLICATIONS = {
        Application.GROMACS: ApplicationSpecialization.gromacs,
        Application.MILC: ApplicationSpecialization.milc,
        Application.CLOUDSC: ApplicationSpecialization.cloudsc,
        Application.ICON: ApplicationSpecialization.icon,
        Application.LLAMMA_CPP: ApplicationSpecialization.llamma_cpp,
        Application.OPENQCD: ApplicationSpecialization.openqcd,
        Application.Q_E: ApplicationSpecialization.q_e,
        Application.VPIC_KOKKOS: ApplicationSpecialization.vpic_kokkos,
    }

    @staticmethod
    def application_configurer(
        name: Application,
    ) -> Callable[[ApplicationSpecialization, dict, dict], str]:
        try:
            return ApplicationSpecializationBuilder.APPLICATIONS[Application(name)]
        except KeyError:
            raise ValueError(f"Unsupported application: {name}")
