from __future__ import annotations

import itertools
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from typing import Generator

from xaas.actions.action import Action
from xaas.docker import VolumeMount
from xaas.config import BuildResult, TargetTriple, BuildSystemArguments, PartialRunConfig, ArgumentsVariableEntry
from xaas.config import CPUArchitecture
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


# TODO: jrabil: i don't think this stuff should extend from RunConfig, it should be a separate config file...
@dataclass
class Config(RunConfig):
    build_results: list[BuildResult] = field(default_factory=list)
    target_flags: list[tuple[set, CPUTuningFeatures]] = field(default_factory=list)


class BuildGenerator(Action):
    def __init__(self):
        super().__init__(name="build", description="Builds the project with specified features")

    def _generate_docker_image(
        self,
        docker_image: str,
        layers_deps: dict[str, LayerDepConfig],
        cpu_architecture: CPUArchitecture,
        build_option: dict[str, str],
    ) -> tuple[str, str]:
        lines: list[str] = []
        name_suffix: list[str] = []

        lines.append(f"FROM {docker_image}")

        for dep_name, dependency in layers_deps.items():
            dep_cfg = XaaSConfig().layers.layers_deps[cpu_architecture][dep_name]

            layer_tag = dep_cfg.image_tag.replace("${version}", dependency.version)
            for arg, value in dependency.arg_mapping.items():
                if not dep_cfg.arg_mapping:
                    continue

                if arg in dep_cfg.arg_mapping:
                    flag_name = dep_cfg.arg_mapping[arg].flag_name
                    flag_value = build_option[arg]

                    layer_tag = layer_tag.replace(f"${{{flag_name}}}", flag_value)
            # TODO: jrabil: we split the tag here and only use the version in the name suffix, but maybe this will result in aliasing?
            name_suffix.append(layer_tag.split(":", 1)[1])

            lines.append(
                f"COPY --from={layer_tag} {dep_cfg.build_location} {dep_cfg.build_location}"
            )

        return "\n".join(lines), "_".join(name_suffix)

    def execute(self, run_config: RunConfig) -> bool:
        logging.info(f"[{self.name}] Building project {run_config.project_name}")
        config_obj = Config.from_instance(run_config)

        # simple check for cuda runtime dependency
        if config_obj.may_contain_feature_boolean(FeatureType.CUDA):
            output = subprocess.run(
                f"grep CUDART_VERSION -nrw {config_obj.source_directory}",
                capture_output=True,
                shell=True,
                text=True,
            )
            if output.returncode == 0:
                # we need to use older CUDA - not necessary so far
                raise NotImplementedError("We need to use older CUDA for this project")

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
    ) -> list[dict[FeatureType, bool]]:
        features_count = len(features)
        num_subsets = 2**features_count
        all_subsets = []

        """
        We generate all 2^n combinations by using bit positions of all integers from 0 to 2^n - 1
        """
        for i in range(num_subsets):
            states : dict[FeatureType, bool] = dict()
            for j in range(features_count):
                states[features[j]] = i & (1 << j) > 0
            all_subsets.append(states)

        return all_subsets

    @staticmethod
    def _all_feature_permutations(config: PartialRunConfig) -> Generator[tuple[dict[FeatureType, bool], dict[str, str]], None, None]:
        permutations_boolean = [
            [tuple([feature, False]), tuple([feature, True])] for feature in config.features_boolean.keys()
        ]

        permutations_select = [
            [ tuple([k, v]) for v in values ] for k, values in config.features_select.items()
        ]

        for states_boolean in itertools.product(*permutations_boolean):
            for states_select in itertools.product(*permutations_select):
                yield dict(states_boolean), dict(states_select)

    @staticmethod
    def generate_name(states_boolean: dict[FeatureType, bool], states_select: dict[str, str]) -> str:
        return "_".join([
            *[x.value for x, state in states_boolean.items() if state],
            *[f"{k}-{v}" for k, v in states_select.items()]
        ])

    def _build_cmake(self, run_config: Config) -> bool:
        containers = []

        # FIXME: test it for multiple combinations
        working_dir = os.path.join(run_config.working_directory)
        os.makedirs(working_dir, exist_ok=True)

        #this pointless if is here because we're going to add another outer loop here eventually and i want to avoid messing up the indentation
        if True:
            # TODO: jrabil: automatically build for multiple configurations
            effective_run_config = run_config.for_target(run_config.cpu_architecture)
            effective_builder_image, effective_runtime_image = effective_run_config.effective_docker_images()

            for states_boolean, states_select in self._all_feature_permutations(effective_run_config):
                build_dir = self.generate_name(states_boolean, states_select)

                builder_image = effective_builder_image

                if len(run_config.layers_deps) > 0:
                    logging.info(
                        f"[{self.name}] Create custom builder image for {run_config.project_name}, {build_dir}"
                    )

                    dockerfile_content, name_suffix = self._generate_docker_image(
                        builder_image,
                        run_config.layers_deps,
                        run_config.cpu_architecture,
                        states_select,
                    )

                    dockerfile_path = os.path.join(working_dir, "images", name_suffix, "Dockerfile")
                    os.makedirs(os.path.join(working_dir, "images", name_suffix), exist_ok=True)

                    with open(dockerfile_path, "w") as f:
                        f.write(dockerfile_content)
                    logging.info(f"[{self.name}] Created Dockerfile in {dockerfile_path}")

                    logging.info(
                        f"[{self.name}] Building Docker image based on {builder_image} for {run_config.project_name} with '{name_suffix}', in {working_dir}"
                    )

                    builder_image = self.docker_runner.build(
                        dockerfile=os.path.join(name_suffix, "Dockerfile"),
                        path=os.path.join(working_dir, "images"),
                        tag=None,
                    ).id

                    logging.info(f"[{self.name}] Successfully built Docker image {builder_image}")

                new_dir = os.path.join(run_config.working_directory, "build", f"build_{build_dir}")
                os.makedirs(new_dir, exist_ok=True)

                arguments = reduce(BuildSystemArguments.merge, [
                    # universal build arguments
                    effective_run_config.build_args,

                    # include build arguments for the current feature selection
                    *[ effective_run_config.features_boolean[feat].args_for_state(state) for feat, state in states_boolean.items() ],
                    *[ effective_run_config.features_select[feat][state] for feat, state in states_select.items() ],
                ])

                # environment variables from the build arguments should be defined when running CMake
                # TODO: jrabil: we probably want to have the environment variables be defined during compilation as well, should we store them in BuildResult?
                cmake_environment = ArgumentsVariableEntry.reduce_to_dict(arguments.environment, self.docker_runner.get_image_env(builder_image))

                target_triple = TargetTriple.from_cpu_architecture(run_config.cpu_architecture)

                toolchain_file_name = "toolchain.cmake"
                toolchain_file = os.path.join(new_dir, toolchain_file_name)
                toolchain_lines = [
                    "set(CMAKE_C_COMPILER clang)",
                    "set(CMAKE_CXX_COMPILER clang++)",
                    f"set(CMAKE_C_FLAGS_INIT \"--target={target_triple.value}\")",
                    f"set(CMAKE_CXX_FLAGS_INIT \"--target={target_triple.value}\")",

                    # properties from the build arguments should be defined as CMake variables
                    *ArgumentsVariableEntry.reduce_to_cmake(arguments.property),
                ]
                with open(toolchain_file, "w") as toolchain_output:
                    toolchain_output.write('\n'.join(toolchain_lines))

                logging.info(
                    f"Executing build in {new_dir}, image {builder_image}, combination: {states_boolean | states_select}"
                )

                cmake_command: list[str] = [
                    "cmake",
                    f"-DCMAKE_TOOLCHAIN_FILE=/build/{toolchain_file_name}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",

                    # additional CMake arguments
                    *arguments.arguments,

                    "-S",
                    "/source",
                    "-B",
                    "/build",
                ]

                configure_commands: list[list[str]] = [
                    cmake_command,
                    *run_config.additional_steps,
                ]

                configure_cmd: str = " && ".join([ shlex.join(command) for command in configure_commands ])
                logging.info(f"[{self.name}] Running: {configure_cmd}")

                volumes = []
                volumes.append(
                    VolumeMount(
                        source=os.path.realpath(run_config.source_directory), target="/source"
                    )
                )
                volumes.append(VolumeMount(source=os.path.realpath(new_dir), target="/build"))

                res = BuildResult(
                    directory=new_dir,
                    builder_image=builder_image,
                    runtime_image=effective_runtime_image,
                    features_boolean=states_boolean,
                    features_select=states_select
                )

                containers.append(
                    (
                        self.docker_runner.run(
                            image=builder_image,
                            command=[ "bash", "-c", configure_cmd ],
                            environment=cmake_environment,
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
