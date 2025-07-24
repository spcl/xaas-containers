import argparse
import json
import logging

from xaas.config import SourceContainerConfig, SourceContainerMode
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
        self._gemini_interface: GeminiInterface | None = None

        if self._config.mode == SourceContainerMode.AUTOMATED:
            self._gemini_interface = GeminiInterface()

    def generate(self):

        specialization_points = utils.load_specialization_points(self._config.project_name)
        logging.debug(f"Loaded specialization points: {specialization_points}")

        application = Application(self._config.project_name)

        if self._config.system_discovery is not None:
            with open(self._config.system_discovery, "r") as f:
                system_features = json.load(f)
            logging.debug(f"Loaded system features: {system_features}")
        else:
            system_features = discover_system()
            logging.debug(f"Discovered system features: {system_features}")

        checker = Checker(specialization_points, system_features)
        options = checker.perform_check()
        logging.debug(f"Available specialization options: {options}")

        if self._config.mode == SourceContainerMode.AUTOMATED:
            assert self._gemini_interface is not None, "Gemini interface is not initialized."
            selected_specializations = self._gemini_interface.select_options(
                options, self._config.project_name
            )
        elif self._config.mode in [SourceContainerMode.PREDEFINED, SourceContainerMode.INTERACTIVE]:
            selected_specializations = utils.get_user_choices(
                checker,
                options,
                self._config.project_name,
                system_features,
                mode=self._config.mode,
                test_options_str=self._config.predefined_config_string,
            )
            logging.debug(f"Selected specializations: {selected_specializations}")
        else:
            raise RuntimeError(f"Unsupported mode: {self._config.mode}")

        app_specialzer = ApplicationSpecialization(
            self._config.source_directory, system_features, self._gemini_interface
        )
        app_func = ApplicationSpecializationBuilder.application_configurer(application)
        build_command = app_func(app_specialzer, selected_specializations, specialization_points)

        dockerfile_creator = DockerfileCreator(
            self._config.source_directory,
            selected_specializations,
            system_features,
            build_command,
            base_image=self._config.deployment_base_image,
        )
        dockerfile_creator.create_dockerfile()

    def deploy(self):
        pass
