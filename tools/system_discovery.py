import subprocess
from collections import defaultdict
import re
import json
import os
import shutil


# verified
# cutomized for the format of the output of "module list" on ault - might break of other systems (not sure)
def get_loaded_modules():
    result = subprocess.run(
        "module list 2>&1",  # Redirect stderr to stdout to capture all output
        shell=True,
        text=True,
        capture_output=True,
    )

    output = result.stdout
    modules = {}

    # Parse the output for module names and versions
    lines = output.split("\n")
    for line in lines:
        matches = re.findall(r"\)\s+([\w\-\.]+)\/([\w\.\-]+)", line)
        for match in matches:
            module_name, module_version = match
            modules[module_name] = module_version

    return modules


# verified
def get_cpu_info():
    """Run the lscpu command and return its output as a string."""
    try:
        output = subprocess.check_output("lscpu", shell=True, text=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error running lscpu: {e}")
        return None


# verified
def parse_lscpu_output(output):
    """Parse the lscpu output to extract the architecture, CPU model name, and flags."""
    architecture = None
    model_name = None
    flags = []
    for line in output.splitlines():
        if line.startswith("Architecture"):
            architecture = line.split(":")[1].strip()
        elif line.startswith("Model name"):
            model_name = line.split(":")[1].strip()
        elif line.startswith("Flags") or line.startswith("Features"):
            flags = line.split(":")[1].strip().split()
    return architecture, model_name, flags


# verified
def determine_vectorization(architecture, flags):
    """Determine all supported vectorization flags for the given architecture."""

    vectorization_flags = {
        "x86_64": [
            "avx-512",
            "avx512f",
            "avx512dq",
            "avx512cd",
            "avx512bw",
            "avx512vl",
            "avx2",
            "avx",
            "sse4_2",
            "sse4_1",
            "ssse3",
            "sse3",
            "sse2",
            "sse",
        ],
        "aarch64": ["asimd", "sve"],
        "ppc64le": ["vsx"],
        "power8": ["vsx"],
        "fujitsu": ["sve"],
    }

    if architecture not in vectorization_flags:
        return []

    supported_flags = [flag for flag in vectorization_flags[architecture] if flag in flags]
    return supported_flags


# verified
# not tested on fpga or tpu
def get_accelerators():
    """Detect accelerators on the system (e.g., GPUs, FPGAs, TPUs) and fetch driver versions if possible."""
    accelerators = defaultdict(list)
    driver_versions = {}
    nvidia_detected = False
    amd_detected = False

    if shutil.which("lspci"):
        try:
            output = subprocess.check_output("lspci", shell=True, text=True)
            for line in output.splitlines():
                if any(keyword in line.lower() for keyword in ["nvidia", "amd", "fpga", "tpu"]):
                    accelerators["lspci"].append(line.strip())
                    if "nvidia" in line.lower():
                        nvidia_detected = True
                    if "amd" in line.lower():
                        amd_detected = True
        except subprocess.CalledProcessError as e:
            print(f"Error running lspci: {e}")

    # Fallback method when lspci is not available
    if shutil.which("nvidia-smi"):
        try:
            smi_output = subprocess.check_output(
                "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader",
                shell=True,
                text=True,
            )
            nv_gpus = []
            gpus = smi_output.splitlines()
            for gpu in gpus:
                data = [x.strip() for x in gpu.split(",")]
                nv_gpus.append({"gpu": data[0], "compute_cap": data[1], "driver": data[2]})
            nvidia_detected = bool(gpus)
            accelerators["nvidia"] = nv_gpus
        except subprocess.CalledProcessError:
            pass

    if shutil.which("rocminfo"):
        try:
            rocminfo_output = subprocess.check_output("rocminfo", shell=True, text=True)
            amd_gpus = []


            if "Agent" in rocminfo_output:

                runtime_version_match = re.search(r'Runtime Version:\s+(.*)', rocminfo_output)
                runtime_version = runtime_version_match.group(1).strip() if runtime_version_match else None

                agent_sections = re.split(r'\*{7,}\s*\nAgent \d+\s*\n\*{7,}\s*\n', rocminfo_output)
                for section in agent_sections[1:]:
                    agent_info = {}
                    device_type_match = re.search(r'Device Type:\s+(.*)', section)
                    if device_type_match and device_type_match.group(1).strip() == 'GPU':
                        name_match = re.search(r'Name:\s+(.*)', section)
                        marketing_name_match = re.search(r'Marketing Name:\s+(.*)', section)

                        if name_match:
                            agent_info['name'] = name_match.group(1).strip()
                        if marketing_name_match:
                            agent_info['marketing_name'] = marketing_name_match.group(1).strip()

                        if agent_info:
                            agent_info['runtime_version'] = runtime_version
                            amd_gpus.append(agent_info)
                    else:
                        name_match = re.search(r'Name:\s+(.*)', section).group(1).strip()
                        print("Ignore non-GPU device", name_match)

                amd_detected = True
            accelerators["amd"] = amd_gpus
        except subprocess.CalledProcessError:
            pass

    # Simple check for FPGA/TPU nodes
    if os.path.exists("/dev/fpga0"):
        accelerators.append("FPGA: /dev/fpga0 detected")
    if os.path.exists("/dev/accel0"):
        accelerators.append("TPU: /dev/accel0 detected")

    # NVIDIA driver version
    if nvidia_detected:
        try:
            smi_output = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
            match = re.search(r"Driver Version:\s*([\d.]+)", smi_output)
            if match:
                driver_versions["NVIDIA"] = match.group(1)
        except Exception as e:
            print(f"Error retrieving NVIDIA driver version: {e}")

    # AMD driver version
    if amd_detected:
        try:
            amd_output = subprocess.run(
                ["modinfo", "amdgpu"], capture_output=True, text=True
            ).stdout
            match = re.search(r"version:\s*([\d.]+)", amd_output)
            if match:
                driver_versions["AMD"] = match.group(1)
        except Exception as e:
            print(f"Error retrieving AMD driver version: {e}")

    driver_version_set = list(driver_versions.values())

    return {
        "Accelerators": accelerators if accelerators else ["No accelerators detected"],
        "Driver Versions": driver_version_set,
    }


# verified
def find_shared_library(lib_name):
    """Find the locations of a shared library."""
    try:
        output = subprocess.check_output(f"ldconfig -p | grep {lib_name}", shell=True, text=True)
        return [line.split()[-1] for line in output.splitlines()]
    except subprocess.CalledProcessError:
        return []  # Return an empty list if the library is not found


# verified
# not tested for mpich, mpi-cray
def get_parallel_libraries():
    # Detect the installed MPI libraries installed in the system by looking at the LD_LIBRARY_PATH
    # Search the paths in LD_LIBRARY_PATH for the shared libraries (libmpi.so.*, libmpifort.so.*, libmpicxx.so.*)
    # to be used in OCI hooks automation

    found_libraries = {}
    mpi_libraries = ["libmpi.so", "libmpifort.so", "libmpicxx.so"]

    if "LD_LIBRARY_PATH" in os.environ:
        ld_library_paths = os.environ["LD_LIBRARY_PATH"].split(":")
        for path in ld_library_paths:
            if not os.path.isdir(path):
                continue
            for root, _, files in os.walk(path):
                for file in files:
                    for lib_name in mpi_libraries:
                        if re.fullmatch(f"{lib_name}\\.\\d+\\.\\d+\\.\\d+", file):
                            full_path = os.path.join(root, file)
                            version = file.split(".so.")[-1]
                            if "intel-oneapi-mpi" in path.lower():
                                mpi_impl = "Intel MPI"
                            elif "openmpi" in path.lower():
                                mpi_impl = "OpenMPI"
                            elif "mpich" in path.lower():
                                mpi_impl = "MPICH"
                            elif "cray" in path.lower():
                                mpi_impl = "Cray MPI"
                            elif any(
                                keyword in path.lower()
                                for keyword in ["rom-openmpi", "roc-ompi", "ompi"]
                            ):
                                mpi_impl = "ROM-OpenMPI"
                            else:
                                mpi_impl = "Unknown MPI"

                            found_libraries.setdefault(mpi_impl, {})[file] = full_path

    if not found_libraries:
        found_libraries["No MPI Found"] = {
            "None": "No MPI shared libraries detected in LD_LIBRARY_PATH"
        }

    return found_libraries


# verified
def get_cufft_versions():
    """Retrieve full versioned cuFFT and cuFFTW shared libraries from CUDA_HOME."""
    cuda_home = os.environ.get("CUDA_HOME")
    cufft_versions = {}
    if cuda_home:
        cufft_path = os.path.join(cuda_home, "lib64")
        if os.path.exists(cufft_path):
            for lib_name in ["libcufft", "libcufftw"]:
                lib_files = [
                    f for f in os.listdir(cufft_path) if f.startswith(lib_name) and ".so." in f
                ]
                if lib_files:
                    # Find fully versioned shared library (e.g., libcufft.so.11.0.2.54)
                    full_versioned_lib = sorted(
                        lib_files, key=lambda x: list(map(int, re.findall(r"\d+", x)))
                    )[-1]
                    version_match = re.search(r"\.so\.(\d+\.\d+\.\d+\.\d+)", full_versioned_lib)
                    version = version_match.group(1) if version_match else "Unknown Version"
                    cufft_versions[lib_name] = {
                        "path": os.path.join(cufft_path, full_versioned_lib),
                        "version": version,
                    }
    return cufft_versions


# verified: works for MKL, cufft, cufftw.
# todo: vkftt, clfft, rocfft
def find_fft_libraries():
    fft_libraries = {}
    fft_libs = {
        "FFTW": "libfftw",
        "FFTPACK": "libfftpack",
        "MKL": "libmkl_core",
        "VkFFT": "libvkfft",
        "clFFT": "libclFFT",
        "rocFFT": "librocfft",
    }

    ld_library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")

    for fft_lib_name, shared_lib in fft_libs.items():
        lib_paths = find_shared_library(shared_lib)

        if fft_lib_name == "MKL":
            for path in ld_library_paths:
                if os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for file in files:
                            if re.match(r"libmkl_core\.so", file):
                                lib_paths.append(os.path.join(root, file))

        if lib_paths:
            if fft_lib_name == "MKL":
                version_match = re.search(r"intel-mkl-(\d+\.\d+\.\d+)", lib_paths[0])
                version = version_match.group(1) if version_match else "Unknown Version"
            else:
                filename_version = re.search(r"\.so\.(\d+\.\d+\.\d+)", lib_paths[0])
                if filename_version:
                    version = filename_version.group(1)
                else:
                    try:
                        version_output = subprocess.check_output(
                            f"strings {lib_paths[0]} | grep -Ei 'version|release'",
                            shell=True,
                            text=True,
                        )
                        version_match = re.search(r"\d+\.\d+(\.\d+)?", version_output)
                        version = version_match.group(0) if version_match else "Unknown Version"
                    except subprocess.CalledProcessError:
                        version = "Unknown Version"

            fft_libraries[fft_lib_name] = {
                "shared_lib": shared_lib,  # Keep the full shared library name
                "path": lib_paths[0],
                "version": version,
            }

    cufft_versions = get_cufft_versions()

    # Rename CUDA-based FFT library keys to match user preference but keep shared_lib as full name
    for lib, details in cufft_versions.items():
        new_key = lib.replace("lib", "")  # e.g., libcufft -> cufft, libcufftw -> cufftw
        fft_libraries[new_key] = {
            "shared_lib": lib,  # Keep full library name like "libcufft"
            "path": details["path"],
            "version": details["version"],
        }

    return fft_libraries


# verified
# can't detect rocblas yet
# returns installation path and not shared libraries
def find_linear_libraries():
    linear_libraries = {}

    linear_libs = {
        "MKL": "mkl",
        "BLAS": "blas",
        "LAPACK": "lapack",
        "SCALAPACK": "scalapack",
        "rocBLAS": "rocblas",
        "cuBLAS": "cublas",
        "OpenBLAS": "openblas",
    }

    ld_library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")

    for lib_name, lib_identifier in linear_libs.items():
        install_path = None
        version = None

        for path in ld_library_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    if lib_identifier.lower() in root.lower():
                        install_path = root
                        version_match = re.search(r"\d+\.\d+\.\d+", root)
                        if version_match:
                            version = version_match.group(0)
                        break
                if install_path:
                    break

        if install_path and version:
            linear_libraries[lib_name] = {"installation_path": install_path, "version": version}

    return linear_libraries


# verified
def check_version(command, pattern=None):
    """Check the version of a library or tool using a shell command and extract using a pattern."""
    try:
        output = subprocess.check_output(command, shell=True, text=True).strip()
        if pattern:
            match = re.search(pattern, output)
            if match:
                return match.group(1)
        return output
    except subprocess.CalledProcessError:
        return "Version not found"


# verified
def get_rocm_version():
    """Attempt to find the ROCm version using multiple methods."""
    # Method 1: Check /opt directory for versioned folders
    try:
        for entry in os.listdir("/opt"):
            if entry.startswith("rocm-"):
                return entry.replace("rocm-", "")
    except Exception as e:
        print(f"Error accessing /opt directory: {e}")

    # Method 2: Check version file if it exists
    try:
        with open("/opt/rocm/version", "r") as version_file:
            return version_file.read().strip()
    except FileNotFoundError:
        pass

    return "Version not found"


# verified
def get_opencl_version():
    """Attempt to find the OpenCL version by inspecting libraries."""
    try:
        output = subprocess.check_output(
            "strings /lib64/libOpenCL.so.1 | grep -i 'OpenCL'", shell=True, text=True
        )
        match = re.search(r"OpenCL (\d+\.\d+)", output)
        if match:
            return match.group(1)
    except subprocess.CalledProcessError:
        pass
    return "Version not found"


# verified
def get_version_from_ld_library_path():
    """Attempt to find versions from LD_LIBRARY_PATH."""
    ld_library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    versions = {}

    for path in ld_library_paths:
        if "hip" in path.lower():
            match = re.search(r"hip-(\d+\.\d+\.\d+)", path)
            if match:
                versions["HIP"] = match.group(1)

        if "rocm" in path.lower():
            match = re.search(r"rocm-(\d+\.\d+\.\d+)", path)
            if match:
                versions["ROCm"] = match.group(1)

        if "cuda" in path.lower():
            match = re.search(r"cuda-(\d+\.\d+)", path)
            if match:
                versions["CUDA"] = match.group(1)

    return versions


# verified
def get_cuda_version():
    """Attempt to find the CUDA version using nvidia-smi."""
    try:
        output = subprocess.check_output("nvidia-smi", shell=True, text=True)
        match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        if match:
            return match.group(1)
    except subprocess.CalledProcessError:
        return "Version not found"
    return "Version not found"


# verified
# To be tested: the shared libraries it returns- I'm not sure if they are correct or not or if we need them
# To be tested: SYCL
# Note: openCL's version finding might be wrong (needs a check)
def get_gpu_backends():
    """Detect GPU backends and their library locations and versions."""
    gpu_backends = {
        "CUDA": {"libraries": [], "version": None},
        "OpenCL": {"libraries": [], "version": None},
        "SYCL": {"libraries": [], "version": None},
        "HIP": {"libraries": [], "version": None},
        "ROCm": {"libraries": [], "version": None},
    }

    # CUDA Detection

    cuda_libs = find_shared_library("libcuda")
    if cuda_libs:
        gpu_backends["CUDA"]["libraries"] = cuda_libs

    # Always try to get CUDA version
    if shutil.which("nvcc"):
        gpu_backends["CUDA"]["version"] = check_version("nvcc --version", r"release (\d+\.\d+)")
    else:
        gpu_backends["CUDA"]["version"] = get_cuda_version()

    # OpenCL Detection
    opencl_libs = find_shared_library("libOpenCL")
    if opencl_libs:
        gpu_backends["OpenCL"]["libraries"] = opencl_libs
        gpu_backends["OpenCL"]["version"] = get_opencl_version()

    # HIP Detection
    hip_libs = find_shared_library("libhip")
    if hip_libs:
        gpu_backends["HIP"]["libraries"] = hip_libs
        gpu_backends["HIP"]["version"] = check_version(
            "hipcc --version", r"HIP version: (\d+\.\d+\.\d+)"
        )

    # ROCm Detection: Force detection if rocm-* is present in /opt
    try:
        opt_entries = os.listdir("/opt")
        rocm_dirs = [entry for entry in opt_entries if entry.startswith("rocm-")]

        if rocm_dirs:
            # print(f"Detected ROCm directories in /opt: {rocm_dirs}")
            gpu_backends["ROCm"]["version"] = get_rocm_version()
        else:
            print("No ROCm directories found in /opt.")
    except Exception as e:
        print(f"Error accessing /opt directory: {e}")

    # Fallback to LD_LIBRARY_PATH if versions are not found
    ld_versions = get_version_from_ld_library_path()
    for backend in ld_versions:
        if (
            gpu_backends[backend]["version"] == "Version not found"
            or gpu_backends[backend]["version"] is None
        ):
            gpu_backends[backend]["version"] = ld_versions[backend]

    return gpu_backends


"""
def get_compilers():
    # Detect installed compilers on the system 
    compilers = {}
    for compiler in ["gcc", "g++", "clang", "icc", "ifort", "mpicc", "mpicxx", "mpifort"]:
        try:
            version = subprocess.check_output(f"{compiler} --version", shell=True, text=True).splitlines()[0]
            compilers[compiler] = version
        except subprocess.CalledProcessError:
            compilers[compiler] = "Not found"
    return compilers
"""


def discover_system():
    results = {}

    # CPU Info
    cpu_output = get_cpu_info()
    if cpu_output is not None:
        architecture, model_name, flags = parse_lscpu_output(cpu_output)
        vectorization_flags = determine_vectorization(architecture, flags)
        results["CPU Info"] = {
            "Architecture": architecture,
            "Model Name": model_name,
            "Supported Vectorizations": vectorization_flags,
        }
    else:
        results["CPU Info"] = "Unable to determine CPU information."

    # Accelerators
    accelerators_info = get_accelerators()
    results["Accelerators"] = accelerators_info

    # Parallel Libraries
    parallel_libraries = get_parallel_libraries()
    results["Parallel Libraries"] = parallel_libraries

    # FFT Libraries
    fft_libraries = find_fft_libraries()
    results["FFT Libraries"] = fft_libraries

    # Math Libraries
    linear_libraries = find_linear_libraries()
    results["linear algebra Libraries"] = linear_libraries

    # GPU Backends
    gpu_backends = get_gpu_backends()
    results["GPU Backends"] = gpu_backends

    loaded_modules = get_loaded_modules()
    results["Loaded Modules"] = loaded_modules

    """
    # Compilers
    compilers = get_compilers()
    results["Compilers"] = compilers
    """

    # return an dictionary of the results
    return results


# For testing only
if __name__ == "__main__":
    system_info = discover_system()
    # print("System Information:")
    # for key, value in system_info.items():
    #    print(f"{key}: {value}")
    with open("system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)
