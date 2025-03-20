import json
import logging
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

from mashumaro.mixins.yaml import DataClassYAMLMixin

from xaas.actions.action import Action
from xaas.actions.build import Config as BuildConfig
from xaas.actions.docker import VolumeMount
from xaas.config import BuildResult
from xaas.config import BuildSystem
from xaas.config import FeatureType
from xaas.config import RunConfig


@dataclass
class CompileCommand(DataClassYAMLMixin):
    compiler: str
    flags: set = field(default_factory=set)
    includes: set = field(default_factory=set)
    optimizations: set = field(default_factory=set)
    definitions: set = field(default_factory=set)
    others: set = field(default_factory=set)


@dataclass
class BuildResult(DataClassYAMLMixin):
    added_files: dict[str, CompileCommand] = field(default_factory=dict)
    removed_files: dict[str, CompileCommand] = field(default_factory=dict)
    different_files: dict[str, CompileCommand] = field(default_factory=dict)


@dataclass
class BuildComparison(DataClassYAMLMixin):
    shared_files: dict[str, CompileCommand] = field(default_factory=dict)
    project_results: dict[str, BuildResult] = field(default_factory=dict)


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

    def execute(self, build_config: BuildConfig) -> bool:
        print(f"[{self.name}] Analyzing project {build_config.project_name}")

        self._result = BuildComparison()

        for build in build_config.build_results:
            logging.info(f"Analyzing build {build}")

            path_project = os.path.join(build_config.working_directory, "build", build.directory)

            self._result.project_results[build.directory] = self._analyze(path_project)

        analyze_config = Config(build_config)
        analyze_config.build_comparison = self._result

        config_path = os.path.join(analyze_config.build.working_directory, "build_analyze.yml")
        analyze_config.save(config_path)

        return True

    def _analyze(self, path_project: str) -> BuildResult:
        project_file = os.path.join(path_project, "compile_commands.json")
        try:
            with open(project_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Error reading {project_file}: {e}") from e

        files = {entry["file"]: entry for entry in data}

        project_result = BuildResult()

        for file, specification in files.items():
            cmd = self._parse_command(specification["command"])
            assert cmd

            project_result.added_files[file] = cmd

        return project_result

    def validate(self, build_config: BuildConfig) -> bool:
        if not os.path.exists(build_config.working_directory):
            print(
                f"[{self.name}] Working directory does not exist: {build_config.source_directory}"
            )
            return False

        if len(build_config.build_results) == 0:
            print(f"[{self.name}] No builds present!")
            return False

        return True

    @staticmethod
    def _parse_command(command: str) -> CompileCommand | None:
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
            # Handle other flags starting with -
            elif elem.startswith("-"):
                result.flags.add(elem)
            # Skip output file
            elif elem == "-o":
                i += 2
                continue
            # Other arguments
            else:
                result.others.add(elem)

            i += 1

        return result
