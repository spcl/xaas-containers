from abc import ABC
from abc import abstractmethod

from xaas.actions.docker import Runner as DockerRunner
from xaas.config import RunConfig
from xaas.config import XaaSConfig


class Action(ABC):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str, description: str):
        self._name = name
        self.description = description
        self.xaas_config = XaaSConfig()
        self.docker_runner = DockerRunner(self.xaas_config.docker_repository)

    @abstractmethod
    def execute(self, run_config: RunConfig) -> bool:
        pass

    @abstractmethod
    def validate(self, run_config: RunConfig) -> bool:
        pass
