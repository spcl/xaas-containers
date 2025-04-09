import logging
import os
import tempfile
from pathlib import Path

from xaas.actions.action import Action
from xaas.actions.analyze import Config as AnalyzerConfig


class DockerImageBuilder(Action):
    def __init__(self):
        super().__init__(
            name="dockerimagebuilder",
            description="Create a Docker image containing all build directories for IR analysis.",
        )
        self.BASE_IMAGE = "spcleth/xaas:llvm-19"

    def execute(self, config: AnalyzerConfig) -> bool:
        project_name = config.build.project_name
        image_name = f"spcleth/xaas:{project_name}-ir"

        logging.info(f"[{self.name}] Building Docker image {image_name} for project {project_name}")

        build_dir = os.path.join(config.build.working_directory, os.path.pardir)
        dockerfile_path = os.path.join(build_dir, "Dockerfile")

        dockerfile_content = self._generate_dockerfile(config)

        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        logging.info(f"[{self.name}] Created Dockerfile in {dockerfile_path}")

        logging.info(f"[{self.name}] Building Docker image: {image_name}, in {build_dir}")

        self.docker_runner.build(dockerfile="Dockerfile", path=build_dir, tag=image_name)

        logging.info(f"[{self.name}] Successfully built Docker image {image_name}")

        return True

    def _generate_dockerfile(self, config: AnalyzerConfig) -> str:
        lines = [
            f"FROM {self.BASE_IMAGE}",
            "",
            "# Add build directories for IR analysis",
            "WORKDIR /builds/",
        ]

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

        lines.append("")
        lines.append("# Set environment variables")
        lines.append(f"ENV PROJECT_NAME={config.build.project_name}")

        lines.append("")
        lines.append("# Default command")
        lines.append('CMD ["bash"]')

        return "\n".join(lines)

    def validate(self, config: AnalyzerConfig) -> bool:
        work_dir = config.build.working_directory
        if not os.path.exists(work_dir):
            logging.error(f"[{self.name}] Working directory does not exist: {work_dir}")
            return False

        if len(config.build.build_results) == 0:
            logging.error(f"[{self.name}] No builds present!")
            return False

        return True
