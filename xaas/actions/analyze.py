from __future__ import annotations

import json
import logging
import os
import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Annotated

from mashumaro.mixins.yaml import DataClassYAMLMixin
from mashumaro.types import Discriminator

from xaas.actions.action import Action
from xaas.actions.build import Config as BuildConfig


class Compiler(str, Enum):
    CLANG = "clang"
    NVCC = "nvcc"
    ICPX = "icpx"


class DivergenceReason(Enum):
    COMPILER = "compiler"
    FLAGS = "flags"
    INCLUDES = "includes"
    DEFINITIONS = "definitions"
    OPTIMIZATIONS = "optimizations"
    CPU_TUNING = "cpu-tuning"
    OTHERS = "other"

    def __str__(self):
        return self.name.lower()


@dataclass
class CompileCommand(DataClassYAMLMixin):
    source: str
    build_dir: str
    compiler: str
    compiler_type: Compiler
    flags: set = field(default_factory=set)
    includes: set = field(default_factory=set)
    optimizations: set = field(default_factory=set)
    cpu_tuning: set = field(default_factory=set)
    definitions: set = field(default_factory=set)
    others: set = field(default_factory=set)


@dataclass
class ClangCompileCommand(CompileCommand):
    compiler_type: Compiler = Compiler.CLANG


@dataclass
class NVCCCompileCommand(CompileCommand):
    compiler_type: Compiler = Compiler.NVCC
    ccbin: str | None = None
    gencode_ptx: set = field(default_factory=set)
    gencode_sass: set = field(default_factory=set)
    # CUDA specific file - list of options
    response_files: set = field(default_factory=set)


@dataclass
class ProjectDivergence(DataClassYAMLMixin):
    reasons: dict[DivergenceReason, dict[str, set[str] | str]] = field(default_factory=dict)


@dataclass
class SourceFileStatus(DataClassYAMLMixin):
    """Status of a source file across all projects."""

    default_build: str
    default_command: Annotated[
        CompileCommand,
        Discriminator(field="compiler_type", include_subtypes=True),
    ]
    present_in_projects: set[str] = field(default_factory=set)
    divergent_projects: dict[str, ProjectDivergence] = field(default_factory=dict)

    # cpu_tuning: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class ProjectResult(DataClassYAMLMixin):
    """Results from analyzing a single project build."""

    # Target file -> compile command
    files: dict[
        str,
        Annotated[
            CompileCommand,
            Discriminator(field="compiler_type", include_subtypes=True),
        ],
    ] = field(default_factory=dict)


@dataclass
class BuildComparison(DataClassYAMLMixin):
    """Comparison of builds across all projects."""

    source_files: dict[str, SourceFileStatus] = field(
        default_factory=dict
    )  # Source file location -> status
    project_results: dict[str, ProjectResult] = field(
        default_factory=dict
    )  # Project name -> build result


@dataclass
class Config(DataClassYAMLMixin):
    build: BuildConfig = field(default_factory=BuildConfig)
    build_comparison: BuildComparison = field(default_factory=BuildComparison)

    def save(self, config_path: str) -> None:
        with open(config_path, "w") as f:
            f.write(self.to_yaml())

    @staticmethod
    def load(config_path: str) -> Config:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return Config.from_yaml(f)


class BuildAnalyzer(Action):
    def __init__(self):
        super().__init__(name="buildanalyzer", description="Compare all builds")

    def print_summary(self, config: Config) -> None:
        logging.info(f"Total files: {len(config.build_comparison.source_files)}")

        for build_name, build in config.build_comparison.project_results.items():
            logging.info(f"Project {build_name}:")
            logging.info(f"\tFiles {len(build.files)} files")

        differences = defaultdict(set)
        different_files = set()
        for src, status in config.build_comparison.source_files.items():
            for _, project_status in status.divergent_projects.items():
                for key in project_status.reasons:
                    differences[key].add(src)
                    different_files.add(src)

        logging.info(f"Different files: {len(different_files)}")
        for key, value in differences.items():
            logging.info(f"Difference: {key}, count: {len(value)}")

    def execute(self, build_config: BuildConfig) -> bool:
        logging.info(f"[{self.name}] Analyzing project {build_config.project_name}")

        self._result = BuildComparison()

        for build in build_config.build_results:
            logging.info(f"Analyzing build {build}")

            # path_project = os.path.join(build_config.working_directory, "build", build.directory)
            path_project = build.directory

            self._result.project_results[build.directory] = self._analyze(path_project)

        self._compare_projects()

        analyze_config = Config(build_config)
        analyze_config.build_comparison = self._result

        config_path = os.path.join(analyze_config.build.working_directory, "build_analyze.yml")
        analyze_config.save(config_path)

        self.print_summary(analyze_config)

        return True

    def _analyze(self, path_project: str) -> ProjectResult:
        project_file = os.path.join(path_project, "compile_commands.json")
        try:
            with open(project_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Error reading {project_file}: {e}") from e

        files = {entry["output"]: entry for entry in data}

        project_result = ProjectResult()

        for target, specification in files.items():
            cmd = self._parse_command(
                specification["command"],
                specification["file"],
                specification["output"],
                specification["directory"],
            )
            assert cmd

            project_result.files[target] = cmd

        return project_result

    def _compare_projects(self) -> None:
        all_files: set[str] = set()
        for _, build_result in self._result.project_results.items():
            all_files.update(build_result.files.keys())

        """
        Logic of comparison:
        - For each file, we find all projects that use this file.
        - We take the first project - one with no features - as default build.
        - Then, for each file in the project, we compare it with the default build.

        The result is the default build and list of all divergences.
        """
        for file_path in all_files:
            projects_with_file = []
            for project_name, build_result in self._result.project_results.items():
                if file_path in build_result.files:
                    projects_with_file.append(project_name)

            assert len(projects_with_file) > 0

            default_project = projects_with_file[0]
            default_command = self._result.project_results[default_project].files[file_path]

            status = SourceFileStatus(
                default_build=default_project,
                default_command=default_command,
                present_in_projects=set(projects_with_file),
            )

            for project_name in projects_with_file[1:]:
                project_command = self._result.project_results[project_name].files[file_path]
                divergence = self._find_command_differences(default_command, project_command)
                if divergence:
                    status.divergent_projects[project_name] = ProjectDivergence(reasons=divergence)

            self._result.source_files[file_path] = status

    def _find_command_differences(
        self, cmd1: CompileCommand, cmd2: CompileCommand
    ) -> dict[DivergenceReason, dict[str, set[str] | str]] | None:
        differences: dict[DivergenceReason, dict[str, set[str] | str]] = {}

        if cmd1.compiler != cmd2.compiler:
            differences[DivergenceReason.COMPILER] = {
                "added": cmd2.compiler,
                "removed": cmd1.compiler,
            }

        flags_diff1 = cmd2.flags - cmd1.flags
        flags_diff2 = cmd1.flags - cmd2.flags
        if flags_diff1 or flags_diff2:
            differences[DivergenceReason.FLAGS] = {"added": flags_diff1, "removed": flags_diff2}

        includes_diff1 = cmd2.includes - cmd1.includes
        includes_diff2 = cmd1.includes - cmd2.includes
        if includes_diff1 or includes_diff2:
            differences[DivergenceReason.INCLUDES] = {
                "added": includes_diff1,
                "removed": includes_diff2,
            }
        # FIXME: added/removed is not the best match here
        # if we point to different build directories,
        # these could include different configs!
        added = []
        for incl in [*cmd1.includes, *cmd2.includes]:
            if "/build" in incl:
                added.append(incl)
        if len(added) > 0:
            if DivergenceReason.INCLUDES in differences:
                differences[DivergenceReason.INCLUDES]["added"].update(added)
            else:
                differences[DivergenceReason.INCLUDES] = {
                    "added": added,
                    "removed": [],
                }

        defines_diff1 = cmd2.definitions - cmd1.definitions
        defines_diff2 = cmd1.definitions - cmd2.definitions
        if defines_diff1 or defines_diff2:
            differences[DivergenceReason.DEFINITIONS] = {
                "added": defines_diff1,
                "removed": defines_diff2,
            }

        opts_diff1 = cmd2.optimizations - cmd1.optimizations
        opts_diff2 = cmd1.optimizations - cmd2.optimizations
        if opts_diff1 or opts_diff2:
            differences[DivergenceReason.OPTIMIZATIONS] = {
                "added": opts_diff1,
                "removed": opts_diff2,
            }

        opts_diff1 = cmd2.cpu_tuning - cmd1.cpu_tuning
        opts_diff2 = cmd1.cpu_tuning - cmd2.cpu_tuning
        if opts_diff1 or opts_diff2:
            differences[DivergenceReason.CPU_TUNING] = {
                "added": opts_diff1,
                "removed": opts_diff2,
            }

        others_diff1 = cmd2.others - cmd1.others
        others_diff2 = cmd1.others - cmd2.others
        if others_diff1 or others_diff2:
            differences[DivergenceReason.OTHERS] = {"added": others_diff1, "removed": others_diff2}

        return differences if differences else None

    def validate(self, build_config: BuildConfig) -> bool:
        if not os.path.exists(build_config.working_directory):
            logging.error(
                f"[{self.name}] Working directory does not exist: {build_config.source_directory}"
            )
            return False

        if len(build_config.build_results) == 0:
            logging.error(f"[{self.name}] No builds present!")
            return False

        return True

    @staticmethod
    def _parse_command(
        command: str, source: str, target: str, build_dir: str
    ) -> CompileCommand | None:
        elems = command.split()
        if not elems:
            return None

        if os.path.basename(elems[0]) in ["clang++", "clang", "cc", "c++"]:
            result = ClangCompileCommand(source, build_dir, elems[0])
        elif os.path.basename(elems[0]) == "nvcc":
            result = NVCCCompileCommand(source, build_dir, elems[0])
        elif os.path.basename(elems[0]) == "icpx":
            # compiler_type = Compiler.ICPX
            raise NotImplementedError()
        else:
            raise RuntimeError(f"Unknown compiler type {elems[0]}")

        i = 1  # Skip compiler name
        while i < len(elems):
            elem = elems[i]

            handled = True
            # Handle preprocessor definitions
            if elem.startswith("-D"):
                result.definitions.add(elem)
            # Handle include paths
            elif elem.startswith("-I") or elem.startswith("-isystem"):
                """
                Two cases:
                (a) -I path, where we need to take two elements
                (b) -Ipath, where we take one element only
                """

                # If the include path is in the same argument
                if (elem.startswith("-I") and len(elem) > 2) or (
                    len(elem) > 8 and elem.startswith("-isystem")
                ):
                    result.includes.add(elem)
                elif elem in ["-I", "-isystem"] and i + 1 < len(elems):
                    result.includes.add(f"{elem} {elems[i + 1]}")
                    i += 1
                else:
                    logging.warning(f"Invalid include path: {elem}")
                    raise RuntimeError("Invalid include path")
            # Handle optimization flags
            elif elem.startswith("-O"):
                result.optimizations.add(elem)
            # Skip output file
            elif elem == "-o":
                i += 2
                continue
            # We catch everything like `-m<...>`
            # this might also catch some other Clang options
            elif re.match(r"-m(?:tune=|arch=|)(\w+)", elem):
                result.cpu_tuning.add(elem)
            # Handle other flags starting with -f
            # ignore warnings
            elif elem.startswith("-f"):
                result.flags.add(elem)
            # ignore warnings
            elif elem.startswith("-W"):
                i += 1
                continue
            else:
                handled = False

            if not handled and isinstance(result, NVCCCompileCommand):
                i, handled = BuildAnalyzer._handle_nvcc_specific(i, elems, result, build_dir)

            # Other arguments
            # catch some outliers
            # ignore source and target files
            if not handled and elem not in ["-c", source, target]:
                result.others.add(elem)

            i += 1
        return result

    @staticmethod
    def _handle_nvcc_specific(
        pos: int, options: list[str], result: NVCCCompileCommand, build_dir: str
    ) -> tuple[int, bool]:
        """
        Handle NVCC specific options like --options-file, ccbin, --generate-code, and -x cu.
        """
        elem = options[pos]

        """
            Some options like gencode are quoted.
            We need to strip quotes to handle them correctly.
        """
        elem = elem.strip('"').strip("'").rstrip("'").rstrip('"')

        """
            Cmake will generate response files for includes, libraries, and objects.
        """
        if elem.startswith("--options-file="):
            options_file_path = elem.split("=", 1)[1]
            result.response_files.add(options_file_path)
            return pos, True
        elif elem == "--options-file":
            if pos + 1 < len(options):
                pos += 1
                result.response_files.add(os.path.join(build_dir, options[pos]))
                return pos, True
            else:
                logging.error("--options-file flag found but no path provided")
                return pos, True

        elif elem.startswith("--compiler-bindir=") or elem.startswith("-ccbin="):
            """
                Handle ccbin (CUDA compiler host compiler).
                It should always point to our Clang when configuration is correct.
            """
            ccbin_path = elem.split("=", 1)[1]
            result.ccbin = ccbin_path
            return pos, True
        elif elem in ["--compiler-bindir", "-ccbin"]:
            if pos + 1 < len(options):
                pos += 1
                result.ccbin = options[pos]
                return pos, True
            else:
                logging.error(f"{elem} flag found but no path provided")
                return pos, True

        elif elem.startswith("--generate-code=") or elem.startswith("-gencode="):
            gencode_spec = elem.split("=", 1)[1]
            BuildAnalyzer._parse_gencode_spec(gencode_spec, result)
            return pos, True
        elif elem in ["--generate-code", "-gencode"]:
            if pos + 1 < len(options):
                pos += 1
                gencode_spec = options[pos]
                BuildAnalyzer._parse_gencode_spec(gencode_spec, result)
                return pos, True
            else:
                logging.error("--generate-code flag found but no specification provided")
                return pos, True

        elif elem == "-x":
            """
                Handle -x cu; input file type is CUDA.
            """
            if pos + 1 < len(options) and options[pos + 1] == "cu":
                pos += 1
                result.others.add("-x cu")
                return pos, True

        return pos, False

    @staticmethod
    def _parse_gencode_spec(gencode_spec: str, result: NVCCCompileCommand) -> None:
        """
        Parse a gencode specification and add to appropriate gencode sets.
        Examples:
        - "arch=compute_70,code=sm_70" -> SASS 70
        - "arch=compute_75,code=compute_75" -> PTX 75
        - "arch=compute_80,code=[sm_80,compute_80]" -> SASS 80 & PTX 80
        """
        import re

        arch_match = re.search(r"arch=compute_(\d+)", gencode_spec)
        if not arch_match:
            logging.error(f"Could not parse architecture from gencode spec: {gencode_spec}")
            return

        """
            Now etract code targets - this distinguishes SASS and PTX targets.
        """
        code_match = re.search(r"code=(.+)", gencode_spec)
        if not code_match:
            logging.error(f"Could not parse code targets from gencode spec: {gencode_spec}")
            return

        code_part = code_match.group(1)

        if code_part.startswith("[") and code_part.endswith("]"):
            """
                Handle bracket notation: [sm_70,compute_70]
            """
            targets = code_part[1:-1].split(",")
        else:
            """
                Handle single bracket: sm_70
            """
            targets = [code_part]

        for target in targets:
            target = target.strip()
            if target.startswith("sm_"):
                # SASS (binary) code
                version = target[3:]
                result.gencode_sass.add(version)
            elif target.startswith("compute_"):
                # PTX code
                version = target[8:]
                result.gencode_ptx.add(version)
            else:
                raise RuntimeError(f"Unknown code target format: {target}")
