from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import field

from xaas.actions.action import Action
from xaas.actions.docker import VolumeMount
from xaas.config import BuildResult
from xaas.config import BuildSystem
from xaas.config import FeatureType
from xaas.config import XaaSConfig
from xaas.config import RunConfig
from xaas.config import FeatureSelectionType, LayerDepConfig

from mashumaro.mixins.yaml import DataClassYAMLMixin


@dataclass
class CPUTuningFeatures(DataClassYAMLMixin):
    target_cpu: str | None = None
    target_features: str | None = None
    tune_cpu: str | None = None


@dataclass
class Config(RunConfig):
    build_results: list[BuildResult] = field(default_factory=list)
    docker_image: str = "builder"
    docker_image_dev: str = "builder-dev"
    target_flags: list[tuple[set, CPUTuningFeatures]] = field(default_factory=list)

    @staticmethod
    def load(config_path: str) -> Config:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return Config.from_yaml(f)


class BuildGenerator(Action):
    def __init__(self):
        super().__init__(name="build", description="Builds the project with specified features")

    def _generate_docker_image(
        self, layers_deps: dict[str, LayerDepConfig], build_option: dict[str, str]
    ) -> str:
        lines = []

        name_suffix = []

        dep_names = []
        for dep_name, dependency in layers_deps.items():
            dep_cfg = XaaSConfig().layers.layers_deps[dep_name]

            name = dep_cfg.name.replace("${version}", dep_cfg.version)
            for arg, value in dependency.arg_mapping.items():
                if not dep_cfg.arg_mapping:
                    continue

                if arg in dep_cfg.arg_mapping:
                    flag_name = dep_cfg.arg_mapping[arg].flag_name
                    flag_value = build_option[arg]

                    name = name.replace(f"${{{flag_name}}}", flag_value)
            dep_names.append((dep_name, name))
            name_suffix.append(name)

            lines.append(f"FROM {XaaSConfig().docker_repository}:{name} as {name}")

        lines.append(f"FROM {XaaSConfig().docker_repository}:builder-dev")

        for dep, dep_name in dep_names:
            dep_cfg = XaaSConfig().layers.layers_deps[dep]
            lines.append(
                f"COPY --from={dep_name} {dep_cfg.build_location} {dep_cfg.build_location}"
            )

        return ["\n".join(lines), name_suffix]

    def execute(self, run_config: RunConfig) -> bool:
        logging.info(f"[{self.name}] Building project {run_config.project_name}")

        config_obj = Config.from_instance(run_config)

        status: bool
        if run_config.build_system == BuildSystem.CMAKE:
            status = self._build_cmake(config_obj)
        else:
            raise NotImplementedError(
                f"[{self.name}] Unsupported build system: {run_config.build_system}"
            )

        config_path = os.path.join(run_config.working_directory, "buildgen.yml")
        config_obj.save(config_path)

        return status

    def validate(self, run_config: RunConfig) -> bool:
        if not os.path.exists(run_config.source_directory):
            print(f"[{self.name}] Source location does not exist: {run_config.source_directory}")
            return False

        if run_config.build_system not in [BuildSystem.CMAKE]:
            print(f"[{self.name}] Unsupported build system: {run_config.build_system}")
            return False

        return True

    @staticmethod
    def _generate_subsets(
        features: list[FeatureType],
    ) -> list[list[tuple[FeatureType, FeatureType]]]:
        features_count = len(features)
        num_subsets = 2**features_count
        all_subsets = []

        """
        We generate all 2^n combinations by using bit positions of all integers from 0 to 2^n - 1
        """
        for i in range(num_subsets):
            subset = []
            subset_not_active = []
            for j in range(features_count):
                if i & (1 << j) > 0:
                    subset.append(features[j])
                else:
                    subset_not_active.append(features[j])
            all_subsets.append((subset, subset_not_active))

        return all_subsets

    @staticmethod
    def generate_name(active: list[FeatureType], flags: tuple[str | None]):
        active_name = "_".join([x.value for x in active])
        flags_name = "_".join(flags) if flags[0] is not None else None

        if flags_name is not None:
            return f"{active_name}_{flags_name}"
        else:
            return active_name

    def _build_cmake(self, run_config: Config) -> bool:
        build_dir = os.path.join(run_config.working_directory, "build")

        # generate all combinations
        subsets = self._generate_subsets(list(run_config.features_boolean.keys()))

        containers = []

        options_names = []
        options_select = []
        options_select_flags = []

        for key, selection in run_config.features_select.items():
            for name, option in selection.items():
                options_names.append((key,))
                options_select.append((name,))
                options_select_flags.append((option,))

        if len(options_select) == 0:
            options_names.append((None,))
            options_select.append((None,))
            options_select_flags.append((None,))

        # FIXME: test it for multiple combinations

        for option_name, option, flag in zip(
            options_names, options_select, options_select_flags, strict=True
        ):
            for active, nonactive in subsets:
                build_dir = self.generate_name(active, option)

                docker_image = f"{run_config.docker_image}-{run_config.project_name}"

                if len(run_config.layers_deps) > 0:
                    logging.info(
                        f"[{self.name}] Create custom builder image for {run_config.project_name}, {build_dir}"
                    )

                    # FIXME: support multiple values
                    settings = {}
                    for k, v in zip(option_name, option, strict=True):
                        settings[k] = v

                    dockerfile_content, flag_names = self._generate_docker_image(
                        run_config.layers_deps, settings
                    )
                    name_suffix = "_".join(flag_names)

                    build_dir = os.path.join(run_config.working_directory)
                    os.makedirs(build_dir, exist_ok=True)
                    dockerfile_path = os.path.join(build_dir, "images", name_suffix, "Dockerfile")

                    with open(dockerfile_path, "w") as f:
                        f.write(dockerfile_content)
                    logging.info(f"[{self.name}] Created Dockerfile in {dockerfile_path}")

                    logging.info(
                        f"[{self.name}] Building Docker image: {run_config.docker_image}, in {build_dir}"
                    )

                    if len(name_suffix) > 0:
                        docker_image = f"{docker_image}-{name_suffix}"

                    self.docker_runner.build(
                        dockerfile=os.path.join("images", name_suffix, "Dockerfile"),
                        path=build_dir,
                        tag=f"{XaaSConfig().docker_repository}:{docker_image}",
                    )

                    logging.info(f"[{self.name}] Successfully built Docker image {docker_image}")

                new_dir = os.path.join(run_config.working_directory, "build", f"build_{build_dir}")
                os.makedirs(new_dir, exist_ok=True)

                cmake_args = []
                for arg in active:
                    cmake_args.append(f"-D{run_config.features_boolean[arg][0]}")
                for arg in nonactive:
                    cmake_args.append(f"-D{run_config.features_boolean[arg][1]}")
                for arg in run_config.additional_args:
                    cmake_args.append(f"-D{arg}")

                for arg in flag:
                    if arg is not None:
                        cmake_args.append(f"-D{arg}")

                logging.info(f"Executing build in {new_dir}, combination: {active}")

                configure_cmd = [
                    "bash",
                    "-c",
                    "'cmake",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    *cmake_args,
                    "-S",
                    "/source",
                    "-B",
                    "/build",
                    "&&",
                    "cd /build",
                ]
                for additional_step in run_config.additional_steps:
                    configure_cmd.append(f"&& {additional_step}")
                configure_cmd.append("'")
                logging.info(f"[{self.name}] Running: {' '.join(configure_cmd)}")

                volumes = []
                volumes.append(
                    VolumeMount(
                        source=os.path.realpath(run_config.source_directory), target="/source"
                    )
                )
                volumes.append(VolumeMount(source=os.path.realpath(new_dir), target="/build"))

                res = BuildResult(
                    docker_image=docker_image, directory=new_dir, features_boolean=active
                )

                containers.append(
                    (
                        self.docker_runner.run(
                            image=docker_image,
                            command=" ".join(configure_cmd),
                            mounts=volumes,
                            remove=False,
                            working_dir="/build",
                        ),
                        res,
                    )
                )

        all_successful = True
        logging.info(f"Waiting for {len(containers)} containers to finish")
        for container, result in containers:
            ret = container.wait()

            if ret["StatusCode"] != 0:
                logging.error(f"Build failed: {container.logs().decode()}")
                all_successful = False

            run_config.build_results.append(result)

            container.remove()

        if not all_successful:
            raise RuntimeError("Build failed")

        return True
