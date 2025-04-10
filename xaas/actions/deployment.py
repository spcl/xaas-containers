import json
import logging
import os
import tempfile
from pathlib import Path

from xaas.actions.action import Action
from xaas.actions.build import BuildGenerator
from xaas.config import DeployConfig, XaaSConfig


class Deployment(Action):
    def __init__(self, parallel_workers: int):
        super().__init__(
            name="dockerimagebuilder",
            description="Create a Docker image containing all build directories for IR analysis.",
        )
        self.parallel_workers = parallel_workers

    def execute(self, config: DeployConfig) -> bool:
        active = [x for x, val in config.features_boolean.items() if val]
        flags = [val for x, val in config.features_select.items()]
        name = BuildGenerator.generate_name(active, flags)

        dockerfile_path = os.path.join(config.working_directory, name)
        os.makedirs(dockerfile_path, exist_ok=True)

        dockerfile_content = self._generate_dockerfile(name, config)

        with open(os.path.join(dockerfile_path, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)
        logging.info(f"[{self.name}] Created Dockerfile in {dockerfile_path}")

        image_name = config.ir_image.removesuffix("-ir")
        image_name = f"{XaaSConfig().docker_repository}:{image_name}-deploy-{name}"

        self.docker_runner.build(dockerfile="Dockerfile", path=dockerfile_path, tag=image_name)

        logging.info(f"[{self.name}] Successfully built Docker image {image_name}")

        return True

    def _generate_dockerfile(self, name: str, config: DeployConfig) -> str:
        # FIXME: support non-boolean layers

        lines = []
        copies = []
        runtime_copies = []

        for feature, value in config.features_boolean.items():
            if not value:
                continue
            if feature in XaaSConfig().layers.layers:
                layer = XaaSConfig().layers.layers[feature]

                lines.append(
                    f"FROM {XaaSConfig().docker_repository}:{layer.name} as {layer.name}-layer"
                )
                copies.append(
                    f"COPY --from={layer.name}-layer {layer.build_location} {layer.build_location}"
                )
                runtime_copies.append(
                    f"COPY --from={layer.name}-layer {layer.runtime_location} {layer.runtime_location}"
                )

        lines.append(f"FROM {XaaSConfig().docker_repository}:{config.ir_image} as builder")
        lines.extend(copies)
        lines.extend(
            [
                "# Add build directories for IR analysis",
                f"RUN ln -s /builds/build_{name} /build",
                "WORKDIR /build/",
                # "RUN /bin/bash build.sh && make",
                f"RUN parallel -j {self.parallel_workers} < build.sh && make",
            ]
        )

        lines.append(f"FROM {XaaSConfig().docker_repository}:{XaaSConfig().runner_image}")
        lines.append("COPY --from=builder /build /build")
        lines.extend(runtime_copies)

        return "\n".join(lines)

    def validate(self, config: DeployConfig) -> bool:
        return True
