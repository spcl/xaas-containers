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

        if len(flags) == 0:
            flags.append(None)
        name = BuildGenerator.generate_name(active, flags)

        dockerfile_path = os.path.join(config.working_directory, name)
        os.makedirs(dockerfile_path, exist_ok=True)

        dockerfile_content = self._generate_dockerfile(name, config)

        with open(os.path.join(dockerfile_path, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)
        logging.info(f"[{self.name}] Created Dockerfile in {dockerfile_path}")

        image_name = config.ir_image.removesuffix("-ir")
        image_name = f"{config.docker_repository}:{image_name}-deploy-{name}"

        self.docker_runner.build(dockerfile="Dockerfile", path=dockerfile_path, tag=image_name)

        logging.info(f"[{self.name}] Successfully built Docker image {image_name}")

        return True

    def _generate_dockerfile(self, build_dir_name: str, config: DeployConfig) -> str:
        # FIXME: support non-boolean layers

        lines = []
        copies = []
        runtime_copies = []

        layers_to_add = []
        build_option = {}

        for feature, value in config.features_boolean.items():
            if not value:
                continue
            if feature in XaaSConfig().layers.layers:
                layers_to_add.append(XaaSConfig().layers.layers[feature])

        for feature in config.features_enabled:
            if feature in XaaSConfig().layers.layers:
                layers_to_add.append(XaaSConfig().layers.layers[feature])

        for layer in layers_to_add:
            layer_name = layer.name.replace("${version}", layer.version)
            layer_build_location = layer.build_location.replace("${version}", layer.version)
            layer_runtime_location = layer.runtime_location.replace("${version}", layer.version)

            lines.append(
                f"FROM {XaaSConfig().docker_repository}:{layer_name} as {layer_name}-layer"
            )
            copies.append(
                f"COPY --link --from={layer_name}-layer {layer_build_location} {layer_build_location}"
            )
            runtime_copies.append(
                f"COPY --link --from={layer_name}-layer {layer_runtime_location} {layer_runtime_location}"
            )

        for x, val in config.features_select.items():
            build_option[x] = val

        for dep_name, dependency in config.layers_deps.items():
            # FIXME: merge with the similar implementation in build. separate module
            dep_cfg = XaaSConfig().layers.layers_deps[dep_name]
            layer_build_location = dep_cfg.build_location.replace("${version}", dep_cfg.version)
            layer_runtime_location = dep_cfg.runtime_location.replace("${version}", dep_cfg.version)

            name = dep_cfg.name.replace("${version}", dep_cfg.version)
            for arg, value in dependency.arg_mapping.items():
                if not dep_cfg.arg_mapping:
                    continue

                if arg in dep_cfg.arg_mapping:
                    flag_name = dep_cfg.arg_mapping[arg].flag_name
                    flag_value = build_option[arg]

                    name = name.replace(f"${{{flag_name}}}", flag_value)

            lines.append(f"FROM {XaaSConfig().docker_repository}:{name} as {name}-layer")
            copies.append(
                f"COPY --link --from={name}-layer {layer_build_location} {layer_build_location}"
            )
            runtime_copies.append(
                f"COPY --link --from={name}-layer {layer_runtime_location} {layer_runtime_location}"
            )

        lines.append(f"FROM {XaaSConfig().docker_repository}:{config.ir_image} as builder")
        lines.extend(copies)
        lines.extend(
            [
                "# Add build directories for IR analysis",
                f"RUN ln -s /builds/build_{build_dir_name} /build",
                "WORKDIR /build/",
                # "RUN /bin/bash build.sh && make",
                f"RUN parallel -j {self.parallel_workers} < build.sh && make",
            ]
        )

        # FIXME: conditional
        lines.append(f"FROM {XaaSConfig().docker_repository}:{XaaSConfig().runner_image}-dev")
        lines.append("COPY --link --from=builder /build /build")
        lines.extend(runtime_copies)

        return "\n".join(lines)

    def validate(self, config: DeployConfig) -> bool:
        return True
