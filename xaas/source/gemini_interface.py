import google.generativeai as genai
import os
import json
import re
import subprocess


class GeminiInterface:

    PROMPTS = {
        "default": """

    I will share a build file, and I would like you to identify all the specialization points for an HPC program and the associated build flags used to enable those features during the build process. Please pay close attention to:

    - Comments and messages within the build file, as they often reveal the necessary flags.
    - Functions like `gmx_option_multichoice`, which specify build flags and options for libraries.
    - Ensure libraries are correctly matched to their corresponding build flags based on these functions.
    - Option Commands: In some projects, build flags are provided in `option` commands. Look at these commands to extract the build flags correctly.
    - Full Build Flags Extraction: Ensure that the full build flags are extracted, not just partial representations. For instance, if a flag is defined as `-DQE_ENABLE_CUDA=ON`, extract the entire flag with its value.
    - Distinguish Between Build Flags and Preprocessor Definitions: Do not confuse preprocessor definitions (e.g., `__CUDA`, `__MPI`) with actual build flags (e.g., `-DQE_ENABLE_CUDA`, `-DQE_ENABLE_MPI`). Extract only the build flags that are explicitly defined in the build configuration.
    - Portability Frameworks: Some build systems use portability frameworks like Kokkos. Pay attention to build flags like `-DKokkos_ENABLE_OPENMP`, `-DKokkos_ENABLE_PTHREAD`, and `-DKokkos_ENABLE_CUDA`.
    - Vectorization Libraries: Some projects use external vectorization libraries like V4. Look for build flags such as `-DUSE_V4_ALTIVEC`, `-DUSE_V4_PORTABLE`, and `-DUSE_V4_SSE`.

    Key Instructions:
    1. Analyze Functions for Build Flags:
    - Look for functions such as `gmx_option_multichoice`, `gmx_dependent_option`, and `option` commands that define build flags and their corresponding options.
    - For example, the flag `-DGMX_FFT_LIBRARY` has options like `fftw3`, `mkl`, and `fftpack[built-in]`.
    - Another example is `-DGMX_GPU_FFT_LIBRARY` with options like `cuFFT`, `VkFFT`, `clFFT`, `rocFFT`, and `MKL`. Match the library names with the build flags from these function calls.
    - Additionally, the flag `-DGMX_GPU` has options like `CUDA`, `OpenCL`, `SYCL`, and `HIP`. Ensure these GPU backends are matched correctly to their corresponding flags.
    - For Kokkos, look for flags like `-DKokkos_ENABLE_OPENMP`, `-DKokkos_ENABLE_PTHREAD`, and `-DKokkos_ENABLE_CUDA`.

    2. Match Libraries to Flags:
    - Libraries should be matched to their respective build flags based on these function definitions.
    - For example:
        - If `GMX_FFT_LIBRARY` is set to `fftw3`, the build flag is `-DGMX_FFT_LIBRARY=fftw3`.
        - If `GMX_GPU_FFT_LIBRARY` is set to `cuFFT`, the build flag is `-DGMX_GPU_FFT_LIBRARY=cuFFT`.
        - For vectorization, look for flags like `-DUSE_V4_ALTIVEC`, `-DUSE_V4_PORTABLE`, and `-DUSE_V4_SSE`.

    3. Match GPU Backends to GMX_GPU:
    - Ensure that GPU backends (CUDA, OpenCL, SYCL, HIP, METAL) are matched to the `GMX_GPU` flag based on the `gmx_option_multichoice` function.
    - For example:
        - If `GMX_GPU` is set to `CUDA`, the build flag is `-DGMX_GPU=CUDA`.
        - If `GMX_GPU` is set to `SYCL`, the build flag is `-DGMX_GPU=SYCL`.
    - For Quantum ESPRESSO: Ensure that GPU backends like CUDA are matched to their corresponding build flags, such as `-DQE_ENABLE_CUDA`, instead of preprocessor definitions like `__CUDA`.

    4. Consider Default Values and Dependencies:
    - Identify the default libraries and how they are conditionally set. For example:
        - `GMX_FFT_LIBRARY_DEFAULT` is `mkl` if `GMX_INTEL_LLVM` is set, otherwise `fftw3`.
        - The GPU FFT library defaults vary based on the GPU backend (e.g., `cuFFT` for CUDA, `VkFFT` for OpenCL).

    5. Special Attention to FFT Libraries:
    - Look for all flags related to FFT libraries like:
        - `-DGMX_FFT_LIBRARY`
        - `-DGMX_FFT_LIBRARY_DEFAULT`
        - `-DGMX_GPU_FFT_LIBRARY`
    - Extract not only the flag but also the corresponding library it enables (e.g., `fftw3`, `mkl`, `cuFFT`).

    6. Include Relevant Build Flags:
    - Do not include preprocessor definitions generated internally. Only include build flags explicitly defined in the file.
    - Ensure that each build flag is extracted with its full definition, including any assigned values.

    Specifically, identify the following:

    - Does the build system support GPU builds? (For example, the presence of a flag like BUILD_GPU indicates GPU support.)
    - What GPU backends does it support (e.g. CUDA, HIP, SYCL, OpenCL)? Are these backends enabled or disabled by default? What is their minimum version, if specified?
    - What parallel programming libraries (e.g. MPI, OpenMP, Pthread, thread-MPI, OpenACC) are supported, and are they enabled or disabled by default? What is their minimum version, if specified?
    - What linear algebra libraries (e.g. BLAS, LAPACK, SCALAPACK, MKL/oneMKL) does the build system use, and under which conditions? What are the default libraries used in the build process?
    - What Fast Fourier Transform libraries (e.g. FFTW, fftpack, MKL/oneMKL, cuFFT, vkFFT, clFFT, rocFFT) does the build system use? What library is built-in? Are there specific dependencies for the library to be used (for example, they must be used with a certain GPU backend or parallel library)? Are they enabled or disabled by default? For the build-flags, look for flags defined via `gmx_option_multichoice` such as `-DGMX_FFT_LIBRARY`, `-DGMX_FFT_LIBRARY_DEFAULT`, `-DGMX_GPU_FFT_LIBRARY`.
    - What other external libraries are used, what versions are specified, and what are their dependencies? List all external libraries and the conditions for their use.
    - What other compiler flags are supported?
    - Are there build flags used to optimize the performance of the program? (e.g., auto-tuning, team reduction, hierarchical parallelism, accumulators, qunatization, batch size, force use of custom matrix multiplications)
    - Which compilers are supported, and what are the minimum versions required?
    - What architectures does the system support?
    - Does it support SIMD vectorization, and what vectorization levels are supported? find the build flag for each supported vectorization level.
    - What is the minimum version required for the build system? Is it a CMake or Make build system?
    - Are there any libraries that require internal builds? If so, name them and provide the build flags (e.g. `-DGMX_BUILD_OWN_FFTW`, `DBUILD_INTERNAL_KOKKOS`).

    The answer should be provided as a JSON structure adhering to the specified schema, with keys including `gpu_build`, `gpu_backends`, `parallel_programming_libraries`, `linear_algebra_libraries`, `fft_libraries`, `other_external_libraries`, `optimization_build_flags`, `compiler_flags`, `compilers`, `architectures`, `simd_vectorization`, and `build_system`, `internal_build`. The `build_flag` value for each feature should be the flag itself (e.g., `-DGMX_SIMD`, `-DGMX_GPU`, `-DQE_ENABLE_CUDA`, `-DQE_ENABLE_MPI`, `-DKokkos_ENABLE_OPENMP`, `-DUSE_V4_ALTIVEC`) without any surrounding text. Do not include any preprocessor definitions that are generated internally. The response must be a valid JSON structure; do not include any introductory or explanatory text.

    Here is the build file:
    {file_content}

    JSON output schema. Use this JSON schema to format your response but do not include it in the output:
    {schema}
    """,
        "bundle": """
   I will provide you with a `bundle.yml` file, which is used to install packages, configure, and compile CMake-based HPC projects. Your task is to identify all specialization points relevant to the HPC program and extract the associated build flags that enable those features during the build process.

    Key Instructions:
    1. Understand the Build Configuration Structure:
    - The `bundle.yml` file contains an `options` section that defines optional build configurations for the project.
    - Unlike traditional CMake-based build systems, the build flags are not standard CMake variables (e.g., `ENABLE_MPI`, `ENABLE_CUDA`).
    - Instead, the build flags are represented by option names such as `with-mpi` and `with-cuda`.
    
    2. Extract Build Flags Correctly:
    - The option name itself is the relevant build flag, NOT the CMake command found under it.
    - Example:
        - Instead of extracting `ENABLE_CUDA=ON`, extract `with-cuda` as the build flag.
        - Instead of `ENABLE_MPI=ON`, extract `with-mpi` as the build flag.

    3. Leverage the `help` Section for Context:
    - The `help` section associated with each option provides additional insights into the purpose and relevance of the build flag.
    - Ensure you capture this context in your extraction.

    4. Ensure Comprehensive Extraction of Specialization Points:
    - Identify all available GPU backends (e.g., `with-cuda`, `with-hip`, `with-sycl`).
    - Recognize parallel programming configurations (`with-mpi`, `without-openmp`).
    - Detect precision settings (`single-precision` vs. `double-precision`).
    - Include architecture-specific optimizations or vectorization flags if present.
    - Extract external library dependencies (e.g., `with-serialbox`, `with-atlas`).


    Specifically, identify the following:

        - Does the build system support GPU builds? (For example, the presence of a flag like BUILD_GPU indicates GPU support.)
        - What GPU backends does it support (e.g. CUDA, HIP, SYCL, OpenCL)? Are these backends enabled or disabled by default? What is their minimum version, if specified?
        - What parallel programming libraries (e.g. MPI, OpenMP, Pthread, thread-MPI, OpenACC) are supported, and are they enabled or disabled by default? What is their minimum version, if specified?
        - What linear algebra libraries (e.g. BLAS, LAPACK, SCALAPACK, MKL/oneMKL) does the build system use, and under which conditions? What are the default libraries used in the build process?
        - What Fast Fourier Transform libraries (e.g. FFTW, fftpack, MKL/oneMKL, cuFFT, vkFFT, clFFT, rocFFT) does the build system use? What library is built-in? Are there specific dependencies for the library to be used (for example, they must be used with a certain GPU backend or parallel library)? Are they enabled or disabled by default? 
        - What other external libraries are used, what versions are specified, and what are their dependencies? List all external libraries and the conditions for their use.
        - What other compiler flags are supported?
        - Are there build flags used to optimize the performance of the program? (e.g., auto-tuning, team reduction, hierarchical parallelism, accumulators)
        - Which compilers are supported, and what are the minimum versions required?
        - What architectures does the system support?
        - Does it support SIMD vectorization, and what vectorization levels are supported? find the build flag for each supported vectorization level.
        - What is the minimum version required for the build system? Is it a CMake or Make build system?
        - Are there any libraries that require internal builds? If so, name them and provide the build flags.

        The answer should be provided as a JSON structure adhering to the specified schema, with keys including `gpu_build`, `gpu_backends`, `parallel_programming_libraries`, `linear_algebra_libraries`, `fft_libraries`, `other_external_libraries`, `optimization_build_flags`, `compiler_flags`, `compilers`, `architectures`, `simd_vectorization`, and `build_system`, `internal_build`. The `build_flag` value for each feature should be the flag itself without any surrounding text. Do not include any preprocessor definitions that are generated internally. The response must be a valid JSON structure; do not include any introductory or explanatory text.

        Here is the build file:
        {file_content}

        JSON output schema. Use this JSON schema to format your response but do not include it in the output:
        {schema}

    """,
        "edit_makefile": """

    You are an expert in HPC application build systems. Your task is to modify an existing Makefile by updating only specific build flags while keeping the rest of the file unchanged. Follow these instructions strictly:

	1.	Preserve all other content: Do not remove or modify any part of the Makefile except for the specified build flags.
	2.	Modify only the requested variables: Update only the flags that I specify. If a flag does not exist, add it to the appropriate variable assignment.
	3.	Retain variable formatting: Keep the structure of the Makefile intact, including variable assignments, spacing, and comments.
	4.	No hallucinated changes: Do not introduce new or made-up flags, variables, or build configurations unless explicitly instructed.
	5.	Ensure correctness: If a flag is already present but needs modification, update it in place.
	6.	Output only the modified Makefile: Return the full Makefile with the requested modifications applied, ensuring clarity in changes.


    Modify the provided Makefile according to the following rules while preserving formatting, comments, and all other content:

    1. Preserve File Integrity
    - Do not rewrite or remove unrelated sections of the Makefile.  
    - Maintain formatting, spacing, and comments.  

    2. Enable Required Build Flags Based on Specialization Points
    - Use the dictionary of `selected_specializations` to determine which build options should be enabled.  
    - Modify only existing variables and flags—do not introduce new ones unless explicitly required.  

    3. Reading Context from Comments
    - If a required build flag is missing, read the comments in the Makefile to determine the correct way to enable it.  
    - Do not assume the provided flags in `selected_specializations` are correct—verify them against the Makefile’s existing structure.  

    4. Mandatory Build Rules 
    - Enable Parallel Execution: Set `MPP = true`.  
    - Enable OpenMP: Set `OMP = true`.  
    - Maintain GNU as the Default Compiler: Ensure `COMPILER = gnu`, unless explicitly overridden.  
    - Enable CUDA if Requested: Instead of adding `-DHAVE_QUDA`, enable the following variables:  
        ```
        WANTQUDA = true
        WANT_CL_BCG_GPU = true
        WANT_FN_CG_GPU = true
        WANT_FL_GPU = true
        WANT_FF_GPU = true
        WANT_GF_GPU = true
        ```
    - Enable FFTW if Selected: Set `WANTFFTW = true`.  
    
    - Set Vectorization Flags in the Correct Location:  
     - For MILC Project:  
       - Vectorization flags must be added only to `OCFLAGS`.  
       - Do not use OpenQCD-specific flags like `-std=c89 -O -mfpmath=sse -mno-avx -Dx64 -DPM`.  
       - Instead, apply vectorization flags from `selected_specializations['vectorization_flags']`:  
         - If `sse2` is selected, append `-msse2` to `OCFLAGS`.  
         - If `avx512f` is selected, append `-mavx512f` to `OCFLAGS`.  
     - For OpenQCD Project:  
       - Modify only `CFLAGS`, not `OCFLAGS`.  
       - Use OpenQCD-specific flags:  
         - If SSE vectorization is selected, apply:  
           ```
           -std=c89 -O -mfpmath=sse -mno-avx -Dx64 -DPM
           ```
         - **If AVX512 is selected**, apply:  
           ```
           -std=c89 -O -DAVX512 -DAVX -DPM -DFMA3
           ```

   - Context Awareness:  
     - Determine if the Makefile is for MILC or OpenQCD based on its variables and comments.  
     - Avoid applying OpenQCD-specific flags to MILC or vice versa.
    

    5. Ensure MPI is Enabled if Required  
    - If `openmpi` is selected as the default, verify MPI-related settings and ensure they are correctly configured.  

    6. Apply Additional Optimization Flags
    - If `optimization_build_flags` are specified, append them to the appropriate location (`CFLAGS`, `OCFLAGS`, etc.), ensuring they do not conflict with existing flags.  

    7. Output Requirements
    - Return the full modified Makefile with only the necessary changes applied.  
    - Do not introduce any new speculative flags.  
    - Ensure correctness by modifying flags in place without duplication.

    Here is the Makefile:
    {makefile_content}

    Here is the selected specialization points that you should enable:
    {selected_specializations}

    """,
        "automated_selection": """
    You are an HPC systems expert. I will provide a JSON object listing available build-time specialization options for a scientific application.

    Your task is to analyze the available options and select the best combination of flags that will yield the highest possible performance on the current system. These options cover GPU backends, parallel programming libraries, FFT and BLAS libraries, vectorization flags, and performance tuning flags.

    Please return your response in the form of a JSON object named `selected_specializations` that follows this schema:

    {{
    "vectorization_flags": {{}},
    "gpu_backends": {{}},
    "parallel_libraries": {{}},
    "fft_libraries": {{}},
    "linear_algebra_libraries": {{}},
    "optimization_build_flags": []
    }}

    - Do not include Markdown formatting or extra text.
    - Return only a valid JSON object in this exact format.
    - Use the provided options only; do not invent new fields or values.
    - For each selected option (e.g., CUDA, OpenMP, MKL), return its full dictionary entry from the provided options, including fields like `build_flag`, `version`, and `used_as_default` if present.
    - Do not return string values such as `"CUDA"` — always return the full key-value pair.

    Here are the available options:
    {options}
    """,
        "gromacs_automated_selection": """
    You are an expert in high-performance computing (HPC) software optimization.

    I will provide you with:
    1. Official documentation from the GROMACS project.
    2. A JSON object listing all available build-time specialization options.

    Your task is to select the best combination of build options for maximizing performance on a modern HPC system using this documentation as a guide.

    Return a JSON object in the following format (do not include any other text):

    {{
    "vectorization_flags": {{}},
    "gpu_backends": {{}},
    "parallel_libraries": {{}},
    "fft_libraries": {{}},
    "linear_algebra_libraries": {{}},
    "optimization_build_flags": []
}}

    Guidelines:
    - Prefer SIMD flags like `AVX_512` if supported and mentioned as optimized in the documentation.
    - Use GPU acceleration (e.g., `CUDA`, `SYCL`) if supported and recommended.
    - Choose parallelization options like `MPI`, `OpenMP`, or `Thread-MPI` based on GROMACS guidance.
    - Select FFT and BLAS libraries known for performance (e.g., `fftw3`, `MKL`).
    - Include any GROMACS-specific performance optimization flags mentioned in the docs.
    - Do not invent flags. Only use what's listed in the available options.
    - For each selected option (e.g., CUDA, OpenMP, MKL), return its full dictionary entry from the provided options, including fields like `build_flag`, `version`, and `used_as_default` if present.
    - Do not return string values such as `"CUDA"` — always return the full key-value pair.

    GROMACS Documentation:
    {docs}

    Available Specialization Options:
    {options}
    """,
    }

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Error: GOOGLE_API_KEY environment variable not set.")

        self.schema = self.load_json_schema()

    def load_json_schema(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(script_dir, "json_schema.json")

        try:
            with open(schema_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Error: JSON schema file not found.")
        except json.JSONDecodeError:
            raise ValueError("Error: Invalid JSON format in schema file.")

    def query_gemini(self, prompt, model_name="gemini-2.0-flash-exp"):
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name)

        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()

            # Debugging: Print the raw response from Gemini
            # print(f"Raw Gemini Response: {response_text}")

            if not response_text:
                return {"error": "Empty response from Gemini API"}

            # Remove Markdown triple backticks for JSON and Makefile responses
            if response_text.startswith("```json") or response_text.startswith("```makefile"):
                response_text = response_text.split("\n", 1)[1]  # Remove the first line
            if response_text.endswith("```"):
                response_text = response_text.rsplit("\n", 1)[0]  # Remove the last line

            return response_text.strip()
        except Exception as e:
            return {"error": str(e)}

    def find_build_files(self, directory):
        cmake_path = None
        makefile_path = None
        configure_ac_path = None
        bundle_yml_path = None

        for entry in os.listdir(directory):
            full_path = os.path.join(directory, entry)
            if os.path.isfile(full_path):
                if entry.casefold() == "cmakelists.txt":
                    cmake_path = full_path
                elif entry.casefold() == "makefile":
                    makefile_path = full_path
                elif entry.casefold() == "configure.ac":
                    configure_ac_path = full_path
                elif entry.casefold() == "bundle.yml":
                    bundle_yml_path = full_path

        # Check "main" subdirectory if Makefile is not found
        if makefile_path is None:
            main_dir = os.path.join(directory, "main")
            if os.path.isdir(main_dir):
                for entry in os.listdir(main_dir):
                    full_path = os.path.join(main_dir, entry)
                    if os.path.isfile(full_path) and entry.casefold() == "makefile":
                        makefile_path = full_path
                        break

        return cmake_path, makefile_path, configure_ac_path, bundle_yml_path

    def read_file(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def find_specialization_points(self, project_dir):
        cmake_path, makefile_path, configure_ac_path, bundle_yml_path = self.find_build_files(
            project_dir
        )
        file_content = ""
        query_type = "default"

        if "llama.cpp" in project_dir.lower():
            if cmake_path:
                file_content += self.read_file(cmake_path) + "\n\n"
            if makefile_path:
                file_content += self.read_file(makefile_path) + "\n\n"
        elif "dwarf-p-cloudsc" in project_dir.lower() and bundle_yml_path:
            file_content = self.read_file(bundle_yml_path)
            query_type = "bundle"
        else:
            if cmake_path:
                file_content = self.read_file(cmake_path)
            elif makefile_path:
                file_content = self.read_file(makefile_path)
            elif configure_ac_path:
                file_content = self.read_file(configure_ac_path)
            else:
                raise FileNotFoundError(
                    "Error: No Makefile, CMakeLists.txt, or configure.ac found."
                )

        prompt = self.PROMPTS[query_type].format(
            schema=json.dumps(self.schema), file_content=file_content
        )
        response_json = self.query_gemini(prompt)

        # Handle API errors
        if isinstance(response_json, dict) and "error" in response_json:
            raise ValueError(f"Gemini API Error: {response_json['error']}")

        # Ensure the response is not empty
        if not response_json:
            raise ValueError("Error: Empty response from Gemini API.")

        try:
            return json.loads(response_json)
        except json.JSONDecodeError:
            raise ValueError(
                f"Error: Gemini response is not valid JSON. Response received:\n{response_json}"
            )

    def edit_makefile(self, selected_specializations, project_dir):
        # Convert project_dir (folder name) into a full path under /users/<username>/
        home_dir = os.path.expanduser("~")  # Get the user's home directory
        project_dir = os.path.join(home_dir, project_dir)  # Construct the full path

        if not os.path.isdir(project_dir):
            raise FileNotFoundError(f"Project directory not found: {project_dir}")

        _, makefile_path, _, _ = self.find_build_files(project_dir)

        if not makefile_path:
            raise FileNotFoundError(f"No Makefile found in {project_dir}")

        print(f"path to makefile:{makefile_path}")

        makefile_content = self.read_file(makefile_path)
        if makefile_content is None:
            raise FileNotFoundError("Could not read the Makefile.")

        prompt = self.PROMPTS["edit_makefile"].format(
            makefile_content=makefile_content,
            selected_specializations=json.dumps(selected_specializations, indent=2),
        )

        modified_makefile = self.query_gemini(prompt)

        if isinstance(modified_makefile, dict) and "error" in modified_makefile:
            raise ValueError(f"Gemini API Error: {modified_makefile['error']}")

        if not isinstance(modified_makefile, str):
            raise ValueError("Gemini returned an invalid Makefile format.")

        print(modified_makefile)

        os.remove(makefile_path)

        with open(makefile_path, "w", encoding="utf-8") as f:
            f.write(modified_makefile)

    def select_options(self, options, project_name=None) -> dict:

        # automatically selects the best specialization options using Gemini.
        # If project_name is 'gromacs', include all docs from '../gromacs-2025.0/docs' in the prompt.

        if project_name and project_name.lower() == "gromacs":
            # FIXME: make configurable
            docs_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../gromacs-2025.0/docs")
            )
            docs_content = ""
            if os.path.isdir(docs_dir):
                for file in sorted(os.listdir(docs_dir)):
                    doc_path = os.path.join(docs_dir, file)
                    if os.path.isfile(doc_path) and file.lower().endswith((".md", ".txt", ".rst")):
                        with open(doc_path, "r", encoding="utf-8") as f:
                            docs_content += f"\n\n# {file}\n" + f.read()
            else:
                raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

            prompt = self.PROMPTS["gromacs_automated_selection"].format(
                docs=docs_content, options=json.dumps(options, indent=2)
            )
        else:
            prompt = self.PROMPTS["automated_selection"].format(
                options=json.dumps(options, indent=2)
            )

        response_text = self.query_gemini(prompt)

        if isinstance(response_text, dict) and "error" in response_text:
            raise ValueError(f"Gemini API Error: {response_text['error']}")

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Error parsing Gemini response as JSON:\n{response_text}")


"""
if __name__ == "__main__":

    project_dir = "milc_qcd-7.8.1"
    helper = GeminiInterface()

    # Finding specialization points
    #specialization_points = helper.find_specialization_points(project_dir)
    #print(json.dumps(specialization_points, indent=2))

    # Example dictionary for editing Makefile
    compilation_options = {'vectorization_flags': ['sse2'], 'gpu_backends': {'CUDA': {'build_flag': '-DHAVE_QUDA', 'version': '12.1'}}, 'parallel_libraries': {'OpenMP': {'build_flag': '-qopenmp', 'version': 'Unknown', 'used_as_default': False}, 'openmpi': {'build_flag': None, 'version': '4.1.1', 'used_as_default': True, 'library_name': 'openmpi'}}, 'fft_libraries': {}, 'linear_algebra_libraries': {'MKL': {'build_flag': None, 'version': '8.4.1', 'used_as_default': False}}, 'optimization_build_flags': []}


    # Editing the Makefile
    helper.edit_makefile(compilation_options, project_dir)
    #print(result)
"""

