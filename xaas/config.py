from __future__ import annotations

import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path

import yaml
from mashumaro.mixins.yaml import DataClassYAMLMixin


class IRType(Enum):
    LLVM_IR = "llvm-ir"


class BuildSystem(Enum):
    CMAKE = "cmake"
    AUTOTOOLS = "autotools"


class XaaSConfig:
    _instance: XaaSConfig | None = None

    DEFAULT_CONFIGURATION = os.path.join(Path(__file__).parent, "config", "system.yaml")

    @property
    def initialized(self) -> bool:
        return self._initialized

    def __new__(cls) -> XaaSConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self):
        self._initialized: bool
        self.docker_repository: str
        self.ir_type: IRType
        self.parallelism_level: int

    def initialize(self, config_path: str) -> None:
        if self._initialized:
            raise RuntimeError("Configuration already initialized")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        self.docker_repository = config_data["docker_repository"]
        self.parallelism_level = config_data["parallelism_level"]

        match config_data["ir_type"]:
            case IRType.LLVM_IR.value:
                self.ir_type = IRType.LLVM_IR
            case _:
                raise ValueError(f"Unsupported IR type: {config_data['ir_type']}")

        self._initialized = True


class FeatureType(Enum):
    OPENMP = "OPENMP"
    MPI = "MPI"
    CUDA = "CUDA"


class FeatureSelectionType(Enum):
    VECTORIZATION = "VECTORIZATION"


@dataclass
class BuildResult(DataClassYAMLMixin):
    directory: str
    features_boolean: list[FeatureType]
    features_select: dict[FeatureSelectionType, str] = field(default_factory=dict)


@dataclass
class RunConfig(DataClassYAMLMixin):
    working_directory: str
    project_name: str
    build_system: BuildSystem
    source_directory: str
    features_boolean: dict[FeatureType, tuple[str, str]]
    features_select: dict[FeatureSelectionType, dict[str, str]]
    additional_args: list[str]
    additional_steps: list[str]

    @staticmethod
    def load(config_path: str) -> RunConfig:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return RunConfig.from_yaml(f)

    def save(self, config_path: str) -> None:
        with open(config_path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_instance(cls, instance):
        return cls(**asdict(instance))


@dataclass
class DeployConfig(DataClassYAMLMixin):
    ir_image: str
    working_directory: str
    features_boolean: dict[FeatureType, bool]
    features_select: dict[FeatureSelectionType, str]

    @staticmethod
    def load(config_path: str) -> DeployConfig:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return DeployConfig.from_yaml(f)

    def save(self, config_path: str) -> None:
        with open(config_path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_instance(cls, instance):
        return cls(**asdict(instance))
