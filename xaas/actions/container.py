import json
import logging
import os
import re
import tempfile
from pathlib import Path

from xaas.actions.action import Action
from xaas.actions.analyze import DivergenceReason, CompileCommand
from xaas.actions.build import Config as BuildConfig
from xaas.actions.ir import PreprocessingResult
from xaas.actions.preprocess import FileStatus
from xaas.config import XaaSConfig


class DockerImageBuilder(Action):
    def __init__(self, docker_repository: str):
        super().__init__(
            name="dockerimagebuilder",
            description="Create a Docker image containing all build directories for IR analysis.",
        )
        self.BASE_IMAGE = "spcleth/xaas:llvm-19"
        self.BASE_IMAGE_DEV = "spcleth/xaas:llvm-19-dev"

        self.OPT_PATH_DEV = "/opt/llvm/bin/opt"

        self.docker_repository = docker_repository

    def execute(self, config: PreprocessingResult) -> bool:
        project_name = config.build.project_name
        image_name = f"{self.docker_repository}:{project_name}-ir"

        logging.info(f"[{self.name}] Building Docker image {image_name} for project {project_name}")

        build_dir = os.path.join(config.build.working_directory, os.path.pardir)

        uses_dev = False
        for build in config.build.build_results:
            file_path = os.path.join(build.directory, "build.sh")
            with open(file_path, "w") as f:
                lines, dev_flag = self._generate_bashscript(build.directory, config)
                f.write(lines)
                uses_dev |= dev_flag

            logging.info(f"[{self.name}] Created build script in {file_path}")

        dockerfile_path = os.path.join(build_dir, "Dockerfile")
        dockerfile_content = self._generate_dockerfile(build_dir, config, uses_dev)

        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        logging.info(f"[{self.name}] Created Dockerfile in {dockerfile_path}")

        logging.info(f"[{self.name}] Building Docker image: {image_name}, in {build_dir}")

        self.docker_runner.build(dockerfile="Dockerfile", path=build_dir, tag=image_name)

        logging.info(f"[{self.name}] Successfully built Docker image {image_name}")

        return True

    def _generate_bashscript(
        self, project_dir: str, config: PreprocessingResult
    ) -> tuple[str, bool]:
        lines = ["#!/bin/bash", ""]

        uses_dev = False

        project_file = os.path.join(project_dir, "compile_commands.json")
        try:
            with open(project_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Error reading {project_file}: {e}") from e

        compile_dbs = {entry["output"]: entry for entry in data}

        for target, result in config.targets.items():
            if project_dir not in result.projects:
                continue

            cmake_cmd = compile_dbs[target]["command"]
            cmake_directory = compile_dbs[target]["directory"]

            # The paths can be relative:
            # directory: /build/a/b
            # target: a/b/x/c.cpp
            # actual file in the command
            actual_target = os.path.relpath(
                os.path.join("/build", compile_dbs[target]["output"]), cmake_directory
            )
            # print(compile_dbs[src]["output"], actual_target)

            ir_file = result.projects[project_dir].ir_file.file
            ir_cmd = cmake_cmd.replace(compile_dbs[target]["file"], ir_file)

            # TODO: is this general enough?
            compiler = ir_cmd.split(" ")[0]
            ir_cmd = ir_cmd.replace(compiler, f"{compiler} -xir")

            # make sure the path does not mess anything else
            # this happens in gromacs - path to target is included in our path to ir file
            # we can't do whole word boundary with \b because we have slashes, which
            # are trated as not words
            ir_cmd = re.sub(
                rf"-o\s+\b{actual_target}\b", f"-o {compile_dbs[target]['output']}", ir_cmd
            )

            if len(result.projects[project_dir].cpu_tuning) > 0:
                opt_cmd = self._cpu_tune(ir_file, result.projects[project_dir], config.build)
                uses_dev = True

                lines.append(f"{opt_cmd} && {ir_cmd}")
            else:
                lines.append(ir_cmd)
            lines.append("")

        return ("\n".join(lines), uses_dev)

    def _cpu_tune(
        self,
        ir_file: str,
        project: FileStatus,
        build_config: BuildConfig,
    ) -> str:
        # FIXME: sanity check - this needs proper testing
        # We might miss some optimization flags
        # but it should be possible in theory
        if DivergenceReason.OPTIMIZATIONS in project.cmd_differences.reasons:
            raise NotImplementedError()

        # We do two steps
        # (1) We run the custom opt pass to replace targets
        # (2) We run optimizations (together with the previous one)
        # FIXME: hardcoding
        cmd = f"{self.OPT_PATH_DEV} -load-pass-plugin /tools/build/libReplaceTargetFeatures.so "
        cmd += '-passes="replace-target-features" '

        if project.cpu_tuning:
            found = False
            for flags, features in build_config.target_flags:
                if flags == project.cpu_tuning:
                    if features.target_features:
                        cmd += f'-new-target-features="{features.target_features}" '
                    if features.target_cpu:
                        cmd += f'-new-target-cpu="{features.target_cpu}" '
                    if features.tune_cpu:
                        cmd += f'-new-tune-cpu="{features.tune_cpu}" '
                    found = True
                    break

            if not found:
                raise RuntimeError("Not found!")

        cmd += f"{ir_file} -o {ir_file}"

        return cmd

    def _generate_dockerfile(
        self, build_dir: str, config: PreprocessingResult, uses_dev_image: bool
    ) -> str:
        lines = []

        # FIXME: full dev image!
        if uses_dev_image:
            lines.append(f"FROM {self.BASE_IMAGE_DEV} AS llvm-dev")
            lines.append("FROM spcleth/xaas:features-analyzer-dev as features-analyzer")

        lines.extend(
            [
                f"FROM {self.BASE_IMAGE}",
                "",
            ]
        )

        # FIXME: full dev image!
        if uses_dev_image:
            lines.append("COPY --from=llvm-dev /opt/llvm /opt/llvm")
            lines.append("COPY --from=features-analyzer /tools /tools")

        lines.extend(
            [
                "# Add build directories for IR analysis",
                "WORKDIR /builds/",
                "RUN apt-get update && apt-get install -y --no-install-recommends parallel && rm -rf /var/lib/apt/lists/*",
            ]
        )

        for i, build in enumerate(config.build.build_results):
            build_path = os.path.relpath(build.directory, build_dir)
            build_name = Path(build.directory).name

            lines.append(f"# Build {i + 1}: {build_name}")
            lines.append(f"COPY {build_path} /builds/{build_name}")

        """
            Necessary for the build to finish.
        """
        source_path = os.path.relpath(config.build.source_directory, build_dir)
        lines.append("")
        lines.append("# Add source code")
        lines.append(f"COPY {source_path} /source")

        source_path = os.path.relpath(config.build.source_directory)
        lines.append("")
        lines.append("# Add IR files")
        lines.append(
            f"COPY {os.path.relpath(os.path.join(config.build.working_directory, 'irs'), build_dir)} /irs"
        )

        lines.append("")
        lines.append("# Set environment variables")
        lines.append(f"ENV PROJECT_NAME={config.build.project_name}")

        lines.append("")
        lines.append("# Default command")
        lines.append('CMD ["bash"]')

        return "\n".join(lines)

    def validate(self, config: PreprocessingResult) -> bool:
        work_dir = config.build.working_directory
        if not os.path.exists(work_dir):
            logging.error(f"[{self.name}] Working directory does not exist: {work_dir}")
            return False

        if len(config.build.build_results) == 0:
            logging.error(f"[{self.name}] No builds present!")
            return False

        return True
