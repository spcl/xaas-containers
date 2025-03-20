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


@dataclass
class Config(RunConfig):
    build_results: list[BuildResult] = field(default_factory=list)


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
    def _generate_subsets(features: list[FeatureType]) -> list[list[FeatureType]]:
        features_count = len(features)
        num_subsets = 2**features_count
        all_subsets = []

        """
        We generate all 2^n combinations by using bit positions of all integers from 0 to 2^n - 1
        """
        for i in range(num_subsets):
            subset = []
            for j in range(features_count):
                if i & (1 << j) > 0:
                    subset.append(features[j])
            all_subsets.append(subset)

        return all_subsets

    def _build_cmake(self, run_config: RunConfig) -> bool:
        build_dir = os.path.join(run_config.working_directory, "build")

        # generate all combinations
        subsets = self._generate_subsets(list(run_config.features.keys()))

        image = f"{self.xaas_config.docker_repository}:{self.DOCKER_IMAGE}"

        containers = []

        for combination in subsets:
            build_dir = "_".join([x.value for x in combination])
            new_dir = os.path.join(run_config.working_directory, "build", f"build_{build_dir}")
            os.makedirs(new_dir, exist_ok=True)

            cmake_args = []
            for arg in combination:
                cmake_args.append(f"-D{run_config.features[arg]}")

            logging.info(f"Executing build in {new_dir}, combination: {combination}")

            configure_cmd = [
                "cmake",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                *cmake_args,
                "-S",
                "/source",
                "-B",
                "/build",
            ]
            print(f"[{self.name}] Running: {' '.join(configure_cmd)}")

            volumes = []
            volumes.append(
                VolumeMount(source=os.path.realpath(run_config.source_directory), target="/source")
            )
            volumes.append(VolumeMount(source=os.path.realpath(new_dir), target="/build"))

            res = BuildResult(directory=new_dir, features=combination)

            containers.append(
                (
                    self.docker_runner.run(
                        image=image, command=" ".join(configure_cmd), mounts=volumes, remove=False
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
