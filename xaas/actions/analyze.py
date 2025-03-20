import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum

from mashumaro.mixins.yaml import DataClassYAMLMixin

from xaas.actions.action import Action
from xaas.actions.build import Config as BuildConfig


class DivergenceReason(Enum):
    COMPILER = "compiler"
    FLAGS = "flags"
    INCLUDES = "includes"
    DEFINITIONS = "definitions"
    OPTIMIZATIONS = "optimizations"
    OTHERS = "other"

    def __str__(self):
        return self.name.lower()


@dataclass
class CompileCommand(DataClassYAMLMixin):
    compiler: str
    flags: set = field(default_factory=set)
    includes: set = field(default_factory=set)
    optimizations: set = field(default_factory=set)
    definitions: set = field(default_factory=set)
    others: set = field(default_factory=set)


@dataclass
class ProjectDivergence(DataClassYAMLMixin):
    project_name: str
    reasons: dict[DivergenceReason, dict[str, set[str] | str]] = field(default_factory=dict)


@dataclass
class SourceFileStatus(DataClassYAMLMixin):
    """Status of a source file across all projects."""

    default_command: CompileCommand
    present_in_projects: set[str] = field(default_factory=set)
    divergent_projects: dict[str, ProjectDivergence] = field(default_factory=dict)


@dataclass
class ProjectResult(DataClassYAMLMixin):
    """Results from analyzing a single project build."""

    files: dict[str, CompileCommand] = field(default_factory=dict)  # Source file -> compile command


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

            path_project = os.path.join(build_config.working_directory, "build", build.directory)

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

        files = {entry["file"]: entry for entry in data}

        project_result = ProjectResult()

        for file, specification in files.items():
            cmd = self._parse_command(
                specification["command"], specification["file"], specification["output"]
            )
            assert cmd

            project_result.files[file] = cmd

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
                default_command=default_command, present_in_projects=set(projects_with_file)
            )

            for project_name in projects_with_file[1:]:
                project_command = self._result.project_results[project_name].files[file_path]
                divergence = self._find_command_differences(default_command, project_command)
                if divergence:
                    status.divergent_projects[project_name] = ProjectDivergence(
                        project_name=project_name, reasons=divergence
                    )

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
    def _parse_command(command: str, source: str, target: str) -> CompileCommand | None:
        elems = command.split()
        if not elems:
            return None

        result = CompileCommand(elems[0])

        i = 1
        while i < len(elems):
            elem = elems[i]

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
            # Handle other flags starting with -f
            # ignore warnings
            elif elem.startswith("-f"):
                result.flags.add(elem)
            # ignore warnings
            elif elem.startswith("-W"):
                i += 1
                continue
            # Other arguments
            # catch some outliers
            # ignore source and target files
            elif elem not in ["-c", source, target]:
                result.others.add(elem)

            i += 1
        return result
