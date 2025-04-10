import json
import logging
import os
import re
import tempfile
from pathlib import Path

from xaas.actions.action import Action
from xaas.actions.ir import PreprocessingResult
from xaas.config import XaaSConfig


class DockerImageBuilder(Action):
    def __init__(self):
        super().__init__(
            name="dockerimagebuilder",
            description="Create a Docker image containing all build directories for IR analysis.",
        )
        self.BASE_IMAGE = "spcleth/xaas:llvm-19"

    def execute(self, config: PreprocessingResult) -> bool:
        project_name = config.build.project_name
        image_name = f"spcleth/xaas:{project_name}-ir"

        logging.info(f"[{self.name}] Building Docker image {image_name} for project {project_name}")

        build_dir = os.path.join(config.build.working_directory, os.path.pardir)
        dockerfile_path = os.path.join(build_dir, "Dockerfile")

        dockerfile_content = self._generate_dockerfile(config)

        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        logging.info(f"[{self.name}] Created Dockerfile in {dockerfile_path}")

        for build in config.build.build_results:
            file_path = os.path.join(build.directory, "build.sh")
            with open(file_path, "w") as f:
                f.write(self._generate_bashscript(build.directory, config))
            logging.info(f"[{self.name}] Created build script in {file_path}")

        logging.info(f"[{self.name}] Building Docker image: {image_name}, in {build_dir}")

        self.docker_runner.build(dockerfile="Dockerfile", path=build_dir, tag=image_name)

        logging.info(f"[{self.name}] Successfully built Docker image {image_name}")

        return True

    def _generate_bashscript(self, project_dir: str, config: PreprocessingResult) -> str:
        lines = ["#!/bin/bash", ""]

        project_file = os.path.join(project_dir, "compile_commands.json")
        try:
            with open(project_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Error reading {project_file}: {e}") from e

        compile_dbs = {entry["file"]: entry for entry in data}

        for src, result in config.source_files.items():
            cmake_cmd = compile_dbs[src]["command"]
            cmake_directory = compile_dbs[src]["directory"]

            # The paths can be relative:
            # directory: /build/a/b
            # target: a/b/x/c.cpp
            # actual file in the command
            actual_target = os.path.relpath(
                os.path.join("/build", compile_dbs[src]["output"]), cmake_directory
            )
            # print(compile_dbs[src]["output"], actual_target)

            ir_file = result.projects[project_dir].ir_file.file
            ir_cmd = cmake_cmd.replace(compile_dbs[src]["file"], ir_file)
            # make sure the path does not mess anything else
            # this happens in gromacs - path to target is included in our path to ir file
            # we can't do whole word boundary with \b because we have slashes, which
            # are trated as not words
            ir_cmd = re.sub(
                rf"-o\s+\b{actual_target}\b", f"-o {compile_dbs[src]['output']}", ir_cmd
            )

            lines.append(ir_cmd)
            lines.append("")

        return "\n".join(lines)

    def _generate_dockerfile(self, config: PreprocessingResult) -> str:
        lines = []

        for dep in config.build.layers_deps:
            dep_cfg = XaaSConfig().layers.layers_deps[dep]
            lines.append(f"FROM {XaaSConfig().docker_repository}:{dep_cfg.name} as {dep_cfg.name}")

        lines.extend(
            [
                f"FROM {self.BASE_IMAGE}",
                "",
            ]
        )

        for dep in config.build.layers_deps:
            dep_cfg = XaaSConfig().layers.layers_deps[dep]
            lines.append(
                f"COPY --from={dep_cfg.name} {dep_cfg.build_location} {dep_cfg.build_location}"
            )

        lines.extend(
            [
                "# Add build directories for IR analysis",
                "WORKDIR /builds/",
            ]
        )

        for i, build in enumerate(config.build.build_results):
            build_path = build.directory
            build_name = Path(build.directory).name

            lines.append(f"# Build {i + 1}: {build_name}")
            lines.append(f"COPY {build_path} /builds/{build_name}")

        # TODO: make this conditional!
        source_path = os.path.relpath(config.build.source_directory)
        lines.append("")
        lines.append("# Add source code")
        lines.append(f"COPY {config.build.source_directory} /source")

        source_path = os.path.relpath(config.build.source_directory)
        lines.append("")
        lines.append("# Add IR files")
        lines.append(f"COPY {os.path.join(config.build.working_directory, 'irs')} /irs")

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
