import json
import logging
import os
import tempfile
from pathlib import Path

from xaas.actions.action import Action
from xaas.actions.build import BuildGenerator
from xaas.config import DeployConfig


class Deployment(Action):
    def __init__(self):
        super().__init__(
            name="dockerimagebuilder",
            description="Create a Docker image containing all build directories for IR analysis.",
        )
        self.BASE_REPOSITORY = "spcleth/xaas"

    def execute(self, config: DeployConfig) -> bool:
        # project_name = config.build.project_name

        # logging.info(f"[{self.name}] Building Docker image {image_name} for project {project_name}")

        active = [x for x, val in config.features_boolean.items() if val]
        name = BuildGenerator.generate_name(active)

        dockerfile_path = os.path.join(config.working_directory, name)
        os.makedirs(dockerfile_path, exist_ok=True)

        dockerfile_content = self._generate_dockerfile(name, config)

        with open(os.path.join(dockerfile_path, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)
        logging.info(f"[{self.name}] Created Dockerfile in {dockerfile_path}")

        image_name = config.ir_image.removesuffix("-ir")
        image_name = f"{self.BASE_REPOSITORY}:{image_name}-deploy-{name}"

        self.docker_runner.build(dockerfile="Dockerfile", path=dockerfile_path, tag=image_name)

        logging.info(f"[{self.name}] Successfully built Docker image {image_name}")

        return True

    def _generate_dockerfile(self, name: str, config: DeployConfig) -> str:
        # FIXME: assemble layers here
        # FIXME: support non-boolean layers
        lines = [
            f"FROM {self.BASE_REPOSITORY}:{config.ir_image}",
            "",
            "# Add build directories for IR analysis",
            f"RUN ln -s /builds/build_{name} /build",
            "WORKDIR /build/",
            "RUN /bin/bash build.sh && make",
        ]

        return "\n".join(lines)

    def validate(self, config: DeployConfig) -> bool:
        return True
