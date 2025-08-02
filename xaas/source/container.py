import json
import logging
import os
from collections import defaultdict

from xaas.config import (
    SourceContainerConfig,
    SourceContainerMode,
    SourceDeploymentConfig,
    ConfigSelection,
    Language,
)
from xaas.docker import Runner as DockerRunner
from xaas.source.applications import (
    Application,
    ApplicationSpecialization,
    ApplicationSpecializationBuilder,
)
from xaas.source.system_discovery import discover_system
from xaas.source.checker import Checker
from xaas.source.dockerfile_creator import DockerfileCreator
from xaas.source.gemini_interface import GeminiInterface
import xaas.source.utils as utils


class SourceContainerGenerator:
    def __init__(self, config: SourceContainerConfig):
        self._config = config
        self._docker_runner = DockerRunner(self._config.docker_repository)

    def generate(self):
        dockerfile_creator = DockerfileCreator(
            self._config.project_name,
            self._config.working_directory,
            self._config.cpu_architecture,
        )
        dfile_path = dockerfile_creator.create_source_dockerfile(self._config.source_directory)

        image_name = f"{self._config.docker_repository}:{self._config.project_name}-source-{self._config.cpu_architecture}"
        logging.info(f"Building Docker image: {image_name}")
        build_dir = os.path.join(self._config.working_directory, os.path.pardir)
        self._docker_runner.build(
            dockerfile=dfile_path,
            path=build_dir,
            tag=image_name,
        )


class SourceContainerDeployment:
    def __init__(self, config: SourceDeploymentConfig):
        self._config = config
        self._gemini_interface: GeminiInterface | None = None
        self._docker_runner = DockerRunner(self._config.docker_repository)

        if self._config.mode == SourceContainerMode.AUTOMATED:
            self._gemini_interface = GeminiInterface()

    @staticmethod
    def intersect_specializations(
        app_name: str, app_language: Language, system_features: dict
    ) -> tuple[dict, dict, Checker]:
        specialization_points = utils.load_specialization_points(app_name)
        logging.debug(f"Loaded specialization points: {specialization_points}")

        checker = Checker(specialization_points, system_features)
        options = checker.perform_check(app_language)

        return specialization_points, options, checker

    def generate(self, parallel_workers: int, build: bool):
        application = Application(self._config.project_name)

        if self._config.system.system_discovery is not None:
            with open(self._config.system.system_discovery) as f:
                system_features = json.load(f)
            logging.debug(f"Loaded system features: {system_features}")
        else:
            system_features = discover_system()
            logging.debug(f"Discovered system features: {system_features}")

        # add mpich if does not exist - this is the default in our container
        system_features["Parallel Libraries"]["mpich"] = {"libmpich.so": "container_default"}

        specialization_points, options, checker = self.intersect_specializations(
            self._config.project_name, self._config.language, system_features
        )
        logging.debug(f"Available specialization options: {options}")

        if self._config.mode.mode == SourceContainerMode.AUTOMATED:
            assert self._gemini_interface is not None, "Gemini interface is not initialized."
            selected_options = self._gemini_interface.select_options(
                options, self._config.project_name
            )
        elif self._config.mode.mode == SourceContainerMode.INTERACTIVE:
            selected_options = utils.get_user_choices(
                checker, options, self._config.project_name, system_features
            )
            logging.debug(f"Selected specializations: {selected_options}")
        elif self._config.mode.mode == SourceContainerMode.PREDEFINED:
            # selected_specializations = self._config.mode.predefined_config
            # integration with the old interface
            selected_options = defaultdict(dict)
            for key in [
                "vectorization_flags",
                "gpu_backends",
                "parallel_libraries",
                "fft_libraries",
                "linear_algebra_libraries",
                "compiler",
            ]:
                selected_value = getattr(self._config.mode.predefined_config, key)
                if selected_value is None:
                    selected_options[key] = {}
                else:
                    if isinstance(selected_value, list):
                        for selection in selected_value:
                            selected_options[key][selection] = options[key][selection]
                    else:
                        selected_options[key][selected_value] = options[key][selected_value]

        else:
            raise RuntimeError(f"Unsupported mode: {self._config.mode}")

        logging.debug(f"Selected specialization options: {selected_options}")
        app_specialzer = ApplicationSpecialization(system_features, self._gemini_interface)
        app_func = ApplicationSpecializationBuilder.application_configurer(application)
        build_command = app_func(app_specialzer, selected_options, specialization_points)

        dockerfile_creator = DockerfileCreator(
            self._config.project_name,
            self._config.working_directory,
            self._config.system.cpu_architecture,
        )
        dockerfile_name = os.path.join(
            self._config.working_directory, f"Dockerfile.deployment-{self._config.system.name}"
        )
        dockerfile_creator.create_deployment_dockerfile(
            selected_options,
            system_features,
            build_command,
            self._config.source_container,
            dockerfile_name,
            self._config.system.base_image,
        )

        if build:
            image_name = f"{self._config.docker_repository}:{self._config.project_name}"
            image_name += f"-source-deploy-{self._config.system.name}"
            logging.info(f"Building deployed Docker image: {image_name} from {dockerfile_name}")
            build_dir = os.path.join(self._config.working_directory, os.path.pardir)
            self._docker_runner.build(
                dockerfile=dockerfile_name,
                path=build_dir,
                tag=image_name,
                build_args={"nproc": str(parallel_workers)},
            )
