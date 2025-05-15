import os
import json 
from pathlib import Path
import re

# All the helper functions that are used in run.py are here

def get_kokkos_arch(system_features):
    
    # Define known mappings for Kokkos architectures
    kokkos_cpu_arch_map = {
        "Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz": "-DKokkos_ARCH_SKX=ON",
        "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz": "-DKokkos_ARCH_SKX=ON",
        "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz": "-DKokkos_ARCH_ICX=ON",
        "AMD EPYC 7742 64-Core Processor": "-DKokkos_ARCH_ZEN2=ON",
        "AMD EPYC 7H12 64-Core Processor": "-DKokkos_ARCH_ZEN2=ON",
        "AMD EPYC 9V33 64-Core Processor": "-DKokkos_ARCH_ZEN3=ON",
        "Intel(R) Xeon(R) E5-2699 v4 @ 2.20GHz": "-DKokkos_ARCH_BDW=ON",
    }

    kokkos_gpu_arch_map = {
        "Tesla V100": "-DKokkos_ARCH_VOLTA70=ON",
        "Tesla A100": "-DKokkos_ARCH_AMPERE80=ON",
        "NVIDIA A100": "-DKokkos_ARCH_AMPERE80=ON",
        "Tesla P100": "-DKokkos_ARCH_PASCAL61=ON",
        "Tesla P40": "-DKokkos_ARCH_PASCAL60=ON",
        "RTX 3090": "-DKokkos_ARCH_AMPERE86=ON",
        "RTX 4090": "-DKokkos_ARCH_ADA89=ON",
        "H100": "-DKokkos_ARCH_HOPPER90=ON",
        "RTX 2080 Ti": "-DKokkos_ARCH_TURING75=ON",
        "Tesla T4": "-DKokkos_ARCH_TURING75=ON",
    }

    # CPU
    cpu_model = system_features.get("CPU Info", {}).get("Model Name", "")
    cpu_arch = kokkos_cpu_arch_map.get(cpu_model, None)

    # GPU
    # VPIC-Kokkos supports CUDA accelerators only for now, hence, checking nvidia GPUs only  
    gpu_info_list = system_features.get("Accelerators", {}).get("Accelerators", {}).get("nvidia", [])
    gpu_arch = None
    for gpu_info in gpu_info_list:
        gpu_name = gpu_info.get("gpu", "")
        for known_name in kokkos_gpu_arch_map:
            if known_name in gpu_name:
                gpu_arch = kokkos_gpu_arch_map[known_name]
                break
        if gpu_arch:
            break

    return cpu_arch, gpu_arch


# This is for llama.cpp 
def get_cuda_architecture_flag(system_features):
   
    gpu_info_list = system_features.get("Accelerators", {}).get("Accelerators", {}).get("nvidia", [])
    compute_caps = set()

    for gpu in gpu_info_list:
        cc = gpu.get("compute_cap", None)
        if cc:
            major_minor = cc.replace(".", "")  
            compute_caps.add(major_minor)

    if compute_caps:
        sorted_caps = sorted(compute_caps)
        return f'-DCMAKE_CUDA_ARCHITECTURES="{",".join(sorted_caps)}"'
    else:
        return ""


def debug_print(message, debug_enabled):
    """Prints debug messages if debug mode is enabled."""
    if debug_enabled:
        print(f"[DEBUG] {message}")
    
def load_specialization_points(project_name):
    file_path = Path(f"projects_specialization_points/{project_name}.json")
    if not file_path.exists():
        raise FileNotFoundError(f"Specialization points file for {project_name} not found.")
    with open(file_path, 'r') as f:
        return json.load(f)
    
def display_options(options, project_name, checker):

    print("\n=== Available Specialization Options ===")
    
    for category, choices in options.items():
        choice_list = list(choices.keys())  # Convert dict keys to list
        
        # Ensure "None" is added only if it's not already present
        if category in ["gpu_backends", "fft_libraries", "linear_algebra_libraries"] and "None" not in choice_list:
            choice_list.append("None")

        print(f"{category.replace('_', ' ').capitalize()}: {', '.join(choice_list)}")

    # Only show optimization flags if the project supports them
    if project_name in ["vpic-kokkos", "llama.cpp"]:
        optimization_flags = checker.get_optimization_build_flags()
        if optimization_flags:
            print(f"Optimization build flags: {', '.join(optimization_flags)}")

    print("\n========================================\n")

def parse_test_options(test_options_str, available_options, project_name, checker):
    """Parses and validates test mode options from a string."""
    test_options = {category: {} for category in available_options.keys()}

    if test_options_str:
        pattern = re.findall(r'(\S+?)="([^"]+)"|(\S+?)=(\S+)', test_options_str)

        for match in pattern:
            key = match[0] if match[0] else match[2]  # Extract key
            values_str = match[1] if match[1] else match[3]  # Extract values
            values = [v.strip() for v in values_str.split(",")]  # Trim spaces

            if key in available_options:
                for value in values:
                    normalized_value = value.strip()

                    # Handle "None" as a valid selection
                    if normalized_value.lower() == "none":
                        test_options[key] = {}  # Empty dictionary means no selection
                        break

                    for valid_option in available_options[key]:
                        if normalized_value == valid_option:
                            if key == "fft_libraries":
                                test_options[key][valid_option] = available_options[key][valid_option]
                            else:
                                test_options[key] = {valid_option: available_options[key][valid_option]}
                            break
                    else:
                        print(f"Warning: '{value}' is not a valid option for {key}, ignoring.")

            elif key == "optimization_build_flags" and project_name in ["vpic-kokkos", "llama.cpp"]:
                optimization_flags = checker.get_optimization_build_flags()
                test_options["optimization_build_flags"] = [flag for flag in values if flag in optimization_flags]

    return test_options

def select_option(category, choices, selected_options, allow_multiple=False, max_choices=1):
    """Handles user selection interactively."""
    option_list = list(choices.keys())

    # Ensure "None" is only added once
    if category in ["gpu_backends", "fft_libraries", "linear_algebra_libraries"] and "None" not in option_list:
        option_list.append("None")

    print(f"\nSelect an option for {category}:")
    for idx, option in enumerate(option_list, 1):
        print(f"{idx}. {option}")

    if allow_multiple:
        print(f"You can choose up to {max_choices} options.")

    selected_indices = []
    while len(selected_indices) < max_choices:
        selected = input("Enter the number(s) of your selection, separated by commas: ").strip()
        if not selected:
            if selected_indices:
                break
            print("You must select at least one option.")
            continue

        indices = [int(idx.strip()) for idx in selected.split(',') if idx.strip().isdigit() and 1 <= int(idx.strip()) <= len(option_list)]
        selected_indices.extend([idx for idx in indices if idx not in selected_indices])

    for idx in selected_indices:
        choice = option_list[idx - 1]
        if choice == "None":
            selected_options[category] = {}  # Empty dictionary means no selection
        else:
            selected_options[category][choice] = choices.get(choice, {})

# if user select MKL/oneAPI MKL as FFT, then MKL is chosen as default linear algebra library 
# For vpic-kokkos and llama.cpp, let user choose fine-tuning options from optimization_build_flags
def get_user_choices(checker, options, project_name, system_features, mode="user", test_options_str=""):
    if mode == "test":
        display_options(options, project_name, checker)
        return parse_test_options(test_options_str, options, project_name, checker)

    selected_options = {
        'vectorization_flags': [] if project_name in ["milc", "openqcd"] else {},
        'gpu_backends': {},
        'parallel_libraries': {},
        'fft_libraries': {},
        'linear_algebra_libraries': {},
        'optimization_build_flags': []
    }

    # **Sort vectorization flags in order before presenting them**
    vectorization_order = ["none", "sse2", "sse4.1", "avx_128_fma", "avx_256", "avx2_128", "avx2_256", "auto"]
    
    if project_name in ["milc", "openqcd"]:
        available_vectorization_flags = system_features.get("CPU Info", {}).get("Supported Vectorizations", [])
        available_vectorization_flags = sorted(available_vectorization_flags, key=lambda x: vectorization_order.index(x) if x in vectorization_order else len(vectorization_order))

        if available_vectorization_flags:
            print("\nSelect one vectorization flag (from lower to higher):")
            for idx, flag in enumerate(available_vectorization_flags, 1):
                print(f"{idx}. {flag}")

            selected = None
            while selected is None:
                user_input = input("Enter the number of your selection: ").strip()
                if user_input.isdigit():
                    idx = int(user_input)
                    if 1 <= idx <= len(available_vectorization_flags):
                        selected = available_vectorization_flags[idx - 1]
                    else:
                        print("Invalid selection. Please enter a valid number.")
                else:
                    print("Invalid input. Please enter a number.")

            selected_options['vectorization_flags'] = [selected]
            print(f"\nSelected vectorization flag for {project_name}: {selected_options['vectorization_flags']}")

    else:
        if "vectorization_flags" in options and options["vectorization_flags"]:
            sorted_flags = sorted(options["vectorization_flags"].keys(), key=lambda x: vectorization_order.index(x) if x in vectorization_order else len(vectorization_order))
            sorted_options = {flag: options["vectorization_flags"][flag] for flag in sorted_flags}
            select_option("vectorization_flags", sorted_options, selected_options, allow_multiple=False)

    # Ensure OpenMP is always selected explicitly
    if "OpenMP" in options.get("parallel_libraries", {}):
        selected_options["parallel_libraries"]["OpenMP"] = options["parallel_libraries"]["OpenMP"]
        print("\nAutomatically selected OpenMP for parallel libraries.")

    mpi_choices = {k: v for k, v in options.get("parallel_libraries", {}).items() if k != "OpenMP"}

    for category, choices in options.items():
        if choices and category != "vectorization_flags":
            if category == "parallel_libraries":
                print("You can choose one additional MPI variant. OpenMP is always selected.")
                if mpi_choices:
                    select_option(category, mpi_choices, selected_options)

            elif category in ["gpu_backends", "fft_libraries", "linear_algebra_libraries"]:
                # Ensure "None" is always an option
                choices_with_none = {**choices, "None": {}}

                allow_multiple = category == "fft_libraries"  # Allow multiple FFT selections
                select_option(category, choices_with_none, selected_options, allow_multiple=allow_multiple)

                # If "None" was selected, clear the category
                if "None" in selected_options[category]:
                    selected_options[category] = {}

            elif category == "linear_algebra_libraries":
                selected_fft_libs = [lib.lower() for lib in selected_options.get("fft_libraries", {})]
                if any("mkl" in lib for lib in selected_fft_libs):
                    for lib_name, lib_info in choices.items():
                        if "mkl" in lib_name.lower():
                            selected_options["linear_algebra_libraries"] = {lib_name: lib_info}
                            print(f"\nAutomatically selected {lib_name} for linear algebra libraries because MKL FFT was chosen.")
                            break
                    continue  # Skip manual selection if MKL was auto-selected
                select_option(category, choices, selected_options)

            else:
                select_option(category, choices, selected_options)

    # Handle optimization build flags for vpic-kokkos and llama.cpp
    if project_name in ["vpic-kokkos", "llama.cpp"]:
        optimization_flags = checker.get_optimization_build_flags()
        if optimization_flags:
            print("\nSelect optimization build flags (you can choose multiple, separated by commas):")
            for idx, flag in enumerate(optimization_flags, 1):
                print(f"{idx}. {flag}")
            
            selected = input("Enter the number(s) of your selection, separated by commas: ").strip()
            selected_flags = []
            for idx in selected.split(','):
                if idx.strip().isdigit() and 1 <= int(idx.strip()) <= len(optimization_flags):
                    flag = optimization_flags[int(idx.strip()) - 1]
                    if flag == "-DGGML_CUDA_PEER_MAX_BATCH_SIZE":
                        flag += "=128"
                    elif "=" not in flag:
                        flag += "=ON"
                    selected_flags.append(flag)
        
            selected_options["optimization_build_flags"] = selected_flags
    
    return selected_options



# For cmake files
def extract_build_flags(selected_specializations, specialization_points):
    """Extracts build flags from the selected specializations."""
    build_flags = []
    
    # Mapping user-friendly names back to their actual JSON keys
    normalization_map = {
        "mkl (CPU)": "mkl",
        "MKL (GPU)": "MKL"
    }

    internal_library = specialization_points.get("internal_build", {}).get("library_name", None)
    internal_build_flag = specialization_points.get("internal_build", {}).get("build_flag", None)

    # Ensure Kokkos is handled first
    if internal_library == "Kokkos" and internal_build_flag:
        if "=" not in internal_build_flag:
            internal_build_flag += "=ON"
        build_flags.insert(0, internal_build_flag)

    cpu_fft_flag = None  # Stores -DGMX_FFT_LIBRARY flag
    gpu_fft_flag = None  # Stores -DGMX_GPU_FFT_LIBRARY flag

    for category_name, category in selected_specializations.items():
        if isinstance(category, dict):  
            for key, value in category.items():
                json_key = normalization_map.get(key, key)  # Normalize keys
                
                if json_key in specialization_points and isinstance(specialization_points[json_key], dict):
                    flag = specialization_points[json_key].get("build_flag", None)
                    if flag:
                        # Assign CPU vs GPU FFT flags properly
                        if key == "mkl (CPU)":
                            cpu_fft_flag = "-DGMX_FFT_LIBRARY=mkl"
                        elif key == "MKL (GPU)":
                            gpu_fft_flag = "-DGMX_GPU_FFT_LIBRARY=MKL"
                        else:
                            if "=" not in flag:
                                flag += "=ON"
                            build_flags.append(flag)

                elif isinstance(value, dict) and 'build_flag' in value and value['build_flag']:
                    flag = value['build_flag']
                    if "=" not in flag:
                        flag += "=ON"
                    build_flags.append(flag)
                elif isinstance(value, str):
                    flag = value
                    if "=" not in flag:
                        flag += "=ON"
                    build_flags.append(flag)

        elif isinstance(category, list):  
            for flag in category:
                if "=" not in flag:
                    flag += "=ON"
                build_flags.append(flag)

    # Ensure internal library flags are correctly appended
    if internal_library and any(internal_library in selected_specializations.get(cat, {}) for cat in selected_specializations):
        if internal_build_flag and internal_library != "Kokkos":
            if "=" not in internal_build_flag:
                internal_build_flag += "=ON"
            build_flags.append(internal_build_flag)

    # Append FFT flags **at the end** to prevent overwriting
    if cpu_fft_flag:
        build_flags.append(cpu_fft_flag)
    if gpu_fft_flag:
        build_flags.append(gpu_fft_flag)

    return " ".join(build_flags)
