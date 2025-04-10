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
from xaas.config import RunConfig
from xaas.config import FeatureSelectionType


@dataclass
class Config(RunConfig):
    build_results: list[BuildResult] = field(default_factory=list)

    @staticmethod
    def load(config_path: str) -> Config:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return Config.from_yaml(f)


class BuildGenerator(Action):
    def __init__(self):
        super().__init__(name="build", description="Builds the project with specified features")

        self.DOCKER_IMAGE = "builder"

    def execute(self, run_config: RunConfig) -> bool:
        print(f"[{self.name}] Building project {run_config.project_name}")

        status: bool
        config_obj = Config.from_instance(run_config)
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
        print(flags)
        flags_name = "_".join(flags) if flags[0] is not None else None

        if flags_name is not None:
            return f"{active_name}_{flags_name}"
        else:
            return active_name

    def _build_cmake(self, run_config: RunConfig) -> bool:
        build_dir = os.path.join(run_config.working_directory, "build")

        # generate all combinations
        subsets = self._generate_subsets(list(run_config.features_boolean.keys()))

        containers = []

        options_select = []
        options_select_flags = []

        # FIXME: extend to more options
        if FeatureSelectionType.VECTORIZATION in run_config.features_select:
            for name, option in run_config.features_select[
                FeatureSelectionType.VECTORIZATION
            ].items():
                options_select.append((name,))
                options_select_flags.append((option,))

        if len(options_select) == 0:
            options_select.append((None,))
            options_select_flags.append((None,))

        for option, flag in zip(options_select, options_select_flags, strict=True):
            for active, nonactive in subsets:
                build_dir = self.generate_name(active, option)

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

                res = BuildResult(directory=new_dir, features_boolean=active)

                containers.append(
                    (
                        self.docker_runner.run(
                            image=self.DOCKER_IMAGE,
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
