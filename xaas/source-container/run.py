import argparse
import json


from system_discovery import discover_system
from checker import Checker
from dockerfile_creator import DockerfileCreator
from gemini_interface import GeminiInterface
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Specialize project builds based on system capabilities.")
    parser.add_argument("project_name", choices=["gromacs", "milc", "openqcd", "q-e", "vpic-kokkos", "llama.cpp", "cloudsc", "icon-model"],
                        help="Name of the project to specialize.")
    parser.add_argument("--mode", choices=["user", "test", "automated"], default="user",
                    help="Mode of execution: 'user' for interactive mode, 'test' for manual flags, 'automated' for Gemini-driven selection.")
    parser.add_argument("--test-options", type=str, default="",
                        help="Comma-separated list of specialization options for test mode (e.g., 'vectorization_flags=avx2,gpu_backends=openCL')")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode to print internal states.")
    parser.add_argument("--milc_app", type=str, default=None,
                        help="Name of the MILC application to build (required in test mode if project_name is 'milc').")
    parser.add_argument("--base-image", type=str, default=None,
                    help="Base Docker image to use. If not specified, the image is selected based on architecture.")
    return parser.parse_args()


def main():
    args = parse_args()
    debug_enabled = args.debug

    gemini_interface = GeminiInterface()

    specialization_points = utils.load_specialization_points(args.project_name)
    utils.debug_print(f"Loaded specialization points: {specialization_points}", debug_enabled)

    project_name_mapping = {
        "milc": "milc_qcd-7.8.1",
        "gromacs": "gromacs-2025.0",
        "openqcd": "openQCD-2.4.2",
        "cloudsc": "dwarf-p-cloudsc"
    }

    project_directory_name = project_name_mapping.get(args.project_name, args.project_name)

    system_features = discover_system()
    utils.debug_print(f"Discovered system features: {system_features}", debug_enabled)

    checker = Checker(specialization_points, system_features)
    options = checker.perform_check()
    utils.debug_print(f"Available specialization options: {options}", debug_enabled)


    if args.mode == "automated" and args.project_name == "gromacs":
        selected_specializations = gemini_interface.select_options(options, args.project_name)
    elif args.mode == "automated": 
        selected_specializations = gemini_interface.select_options(options, None)
    elif args.mode in ["user", "test"]:
        selected_specializations = utils.get_user_choices(
            checker,
            options,
            args.project_name,
            system_features,
            mode=args.mode,
            test_options_str=args.test_options
        )
        utils.debug_print(f"Selected specializations: {selected_specializations}", debug_enabled)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


    build_command = ""

    if args.project_name == "gromacs":
        release_build = "-DCMAKE_BUILD_TYPE=Release "
        build_flags_string = utils.extract_build_flags(selected_specializations, specialization_points)
        build_flags_string = release_build + build_flags_string

        build_command = f"""
            RUN mkdir build \\
                && cd build \\
                && cmake .. {build_flags_string} \\
                && make -j$(nproc) \\
                && make check \\
                && sudo make install \\
                && source /usr/local/gromacs/bin/GMXRC \\
                && cd ../
            """

        utils.debug_print(f"Generated CMake build flags: {build_flags_string}", debug_enabled)

    elif args.project_name == "milc":
        gemini_interface.edit_makefile(selected_specializations, project_directory_name)
        milc_application_name = args.milc_app if args.mode == "test" and args.milc_app else input("Please enter the name of the MILC application to build: ").strip()
        build_command = f"""
        RUN cd {milc_application_name} \\
        && $cp ../Makefile . \\ 
        && make {milc_application_name}
        """

    elif args.project_name == "openqcd":
        gemini_interface.edit_makefile(selected_specializations, project_directory_name)
        build_command = f"""
        ENV MPI_INCLUDE=/usr/local/mpich/include
        ENV MPI_HOME=/usr/local/mpich/lib 
        RUN cd main && make
        """

    elif args.project_name == "q-e":
        build_flags_string = utils.extract_build_flags(selected_specializations, specialization_points)
        build_command = f"""
            RUN mkdir build \\
                && cd build \\
                && cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 {build_flags_string} \\
                && make -j$(nproc) \\
                && cd ../
            """
        utils.debug_print(f"Generated CMake build flags: {build_flags_string}", debug_enabled)

    elif args.project_name == "vpic-kokkos":
        cpu_arch, gpu_arch = utils.get_kokkos_arch(system_features)
        release_build = "-DCMAKE_BUILD_TYPE=Release  -DCMAKE_CXX_FLAGS=\"-rdynamic\" "
        build_flags_string = utils.extract_build_flags(selected_specializations, specialization_points)

        if 'CUDA' in selected_specializations.get('gpu_backends', {}):
            build_flags_string += " -DCMAKE_CXX_COMPILER=/vpic-kokkos/kokkos/bin/nvcc_wrapper "
        if cpu_arch:
            build_flags_string += f" {cpu_arch} "
        if gpu_arch:
            build_flags_string += f" {gpu_arch} "

        build_command = f"""
            RUN mkdir build \\
                && cd build \\
                && cmake .. {release_build + build_flags_string} \\
                && make -j$(nproc) \\
                && make install \\
                && cd ../
            """
        print(f"Generated CMake build command:\n{build_command}")
        utils.debug_print(f"Generated CMake build flags: {build_flags_string}", debug_enabled)

    elif args.project_name == "llama.cpp":
        # have to specify the blas vendor -DGGML_BLAS_VENDOR=OpenBLAS

        # for perfomance portability
        default_flag = "-DGGML_NATIVE=OFF "

        build_flags_string = utils.extract_build_flags(selected_specializations, specialization_points)
        utils.debug_print(f"Generated CMake build flags: {build_flags_string}", debug_enabled)

        if 'CUDA' in selected_specializations.get('gpu_backends', {}):
            cuda_arch_flag = utils.get_cuda_architecture_flag(system_features)
            build_flags_string += " -DGGML_BACKEND_DL=OFF "
            build_flags_string += f"{cuda_arch_flag}"

        if 'MKL' in selected_specializations.get('linear_algebra_libraries', {}):
            build_flags_string += "  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx  -DGGML_BLAS_VENDOR=Intel10_64lp"
        elif 'OpenBLAS' in selected_specializations.get('linear_algebra_libraries', {}):
            build_flags_string += "  -DGGML_BLAS_VENDOR=OpenBLAS"
        elif 'cuBLAS' in selected_specializations.get('linear_algebra_libraries', {}):
            pass

        build_command = f"""
            RUN cmake -B build {build_flags_string} \\
                && cmake --build build --config Release -j $(nproc)
            """

    elif args.project_name == "icon-model":
        #Not fully tested 
        build_flags_string = utils.extract_build_flags(selected_specializations, specialization_points)
        build_command = f"""
            RUN ./configure {build_flags_string}
            """
        utils.debug_print(f"Generated CMake build flags: {build_flags_string}", debug_enabled)

    elif args.project_name == "cloudsc":
        # Not fully tested
        build_flags_string = utils.extract_build_flags(selected_specializations, specialization_points)
        build_command = f"""
            RUN ./cloudsc-bundle create \\
                && ./cloudsc-bundle build --build-type release --cloudsc-fortran=ON --cloudsc-c=ON --with-serialbox {build_flags_string}
            """
        utils.debug_print(f"Generated CMake build flags: {build_flags_string}", debug_enabled)

    #mandatory_libraries = checker.find_mandatory_installations()
    #utils.debug_print(f"Mandatory libraries to install: {mandatory_libraries}", debug_enabled)

    dockerfile_creator = DockerfileCreator(project_directory_name, selected_specializations, system_features, build_command, base_image=args.base_image)
    dockerfile_creator.create_dockerfile()


if __name__ == "__main__":
    main()
