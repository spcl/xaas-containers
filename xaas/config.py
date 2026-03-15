from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path

import yaml
from mashumaro.mixins.yaml import DataClassYAMLMixin


class CPUArchitecture(str, Enum):
    X86_64 = "x86_64"
    ARM_64 = "arm64"


# TODO: Replace CPUArchitecture with this!
#   (it's separate for now to avoid breaking existing configs)
class TargetTriple(str, Enum):
    X86_64_LINUX_GNU = "x86_64-linux-gnu"
    AARCH64_LINUX_GNU = "aarch64-linux-gnu"

    @staticmethod
    def from_cpu_architecture(cpu_architecture: CPUArchitecture) -> TargetTriple:
        match cpu_architecture:
            case CPUArchitecture.X86_64:
                return TargetTriple.X86_64_LINUX_GNU
            case CPUArchitecture.ARM_64:
                return TargetTriple.AARCH64_LINUX_GNU
            case _:
                raise ValueError(f"Unsupported CPU architecture: {cpu_architecture}")


class IRType(Enum):
    LLVM_IR = "llvm-ir"


class BuildSystem(Enum):
    CMAKE = "cmake"
    AUTOTOOLS = "autotools"


class SourceContainerMode(Enum):
    INTERACTIVE = "interactive"
    PREDEFINED = "predefined"
    AUTOMATED = "automated"


class SourceContainerAutomated(Enum):
    GEMINI = "gemini"


class Language(str, Enum):
    CXX = "cxx"
    FORTRAN = "fortran"


@dataclass
class DockerLayerVersion(DataClassYAMLMixin):
    flag_name: str
    build_args: dict[str, str]


@dataclass
class DockerLayer(DataClassYAMLMixin):
    dockerfile: str
    name: str
    versions: list[str]
    version_arg: str
    build_location: str
    runtime_location: str
    arg_mapping: dict[str, DockerLayerVersion] | None = None
    envs: dict[str, str] = field(default_factory=dict)


@dataclass
class DockerLayers(DataClassYAMLMixin):
    layers: dict[CPUArchitecture, dict[FeatureType, DockerLayer]]
    layers_deps: dict[CPUArchitecture, dict[str, DockerLayer]]

    @staticmethod
    def load(config_path: str) -> DockerLayers:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return DockerLayers.from_yaml(f)


class XaaSConfig:
    _instance: XaaSConfig | None = None

    DEFAULT_CONFIGURATION = os.path.join(Path(__file__).parent, "config", "system.yaml")
    LAYERS_CONFIGURATION = os.path.join(Path(__file__).parent, "config", "layers.yml")

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
        self.runner_image: str
        self.parallelism_level: int
        self.layers: DockerLayers

    def initialize(self, config_path: str) -> None:
        if self._initialized:
            raise RuntimeError("Configuration already initialized")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        self.docker_repository = config_data["docker_repository"]
        self.parallelism_level = config_data["parallelism_level"]
        self.runner_image = config_data["runner_image"]

        match config_data["ir_type"]:
            case IRType.LLVM_IR.value:
                self.ir_type = IRType.LLVM_IR
            case _:
                raise ValueError(f"Unsupported IR type: {config_data['ir_type']}")

        self.layers = DockerLayers.load(XaaSConfig.LAYERS_CONFIGURATION)

        self._initialized = True


class FeatureType(str, Enum):
    OPENMP = "OPENMP"
    MPI = "MPI"
    CUDA = "CUDA"
    ONEAPI = "ONEAPI"
    ROCM = "ROCM"
    SYCL = "SYCL"
    ROCFFT = "ROCFFT"
    FFTW3 = "FFTW3"
    ICPX = "ICPX"


class FeatureSelectionType(Enum):
    VECTORIZATION = "VECTORIZATION"


@dataclass
class BuildSystemArguments(DataClassYAMLMixin):
    environment_set: dict[str, list[str]] = field(default_factory=dict)
    environment_add: dict[str, list[str]] = field(default_factory=dict)
    property_set: dict[str, list[str]] = field(default_factory=dict)
    property_add: dict[str, list[str]] = field(default_factory=dict)
    arguments_add: list[str] = field(default_factory=list)

    @staticmethod
    def __merge_set(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
        if not a.keys().isdisjoint(b.keys()):
            raise RuntimeError(f"duplicate keys: {a.keys() & b.keys()}")

        return a | b

    @staticmethod
    def __merge_add(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
        res = a | b
        for k in a.keys() & b.keys():
            res[k] = a[k] + b[k]
        return res

    @staticmethod
    def merge(a: BuildSystemArguments, b: BuildSystemArguments) -> BuildSystemArguments:
        res = BuildSystemArguments()

        # merge *_set mappings, throwing an exception if the same key is present in both

        res.environment_set = a.environment_set | b.environment_set
        if not a.environment_set.keys().isdisjoint(b.environment_set.keys()):
            raise RuntimeError(f"duplicate keys in environment_set: {a.environment_set.keys() & b.environment_set.keys()}")

        res.property_set = a.property_set | b.property_set
        if not a.property_set.keys().isdisjoint(b.property_set.keys()):
            raise RuntimeError(f"duplicate keys in property_set: {a.property_set.keys() & b.property_set.keys()}")

        # merge *_add mappings, concatenating the values if the same key is present in both

        res.environment_add = a.environment_add | b.environment_add
        for k in a.environment_add.keys() & b.environment_add.keys():
            res.environment_add[k] = a.environment_add[k] + b.environment_add[k]

        res.property_add = a.property_add | b.property_add
        for k in a.property_add.keys() & b.property_add.keys():
            res.property_add[k] = a.property_add[k] + b.property_add[k]

        # simply concatenate any additional arguments
        res.arguments_add = a.arguments_add + b.arguments_add

        return res

    @staticmethod
    def __effective_mapping(
            mappings_default: dict[str, str], mappings_set: dict[str, list[str]], mappings_add: dict[str, list[str]],
            separator: str) -> dict[str, str]:
        if not separator:
            raise RuntimeError("separator string must not be empty!")

        result = mappings_default.copy()

        if not result.keys().isdisjoint(mappings_set.keys()):
            raise RuntimeError(f"duplicate keys: {result.keys() & mappings_set.keys()}")
        
        # insert all set mappings
        for k, vs in mappings_set.items():
            result[k] = separator.join(vs)

        # append all add mappings
        for k, vs in mappings_add.items():
            if k in result:
                result[k] = separator.join([ result[k] ] + vs)
            else:
                result[k] = separator.join(vs)

        return result

    def effective_environment(
            self, default_environment: dict[str, str] = {}, separator: str = " ") -> dict[str, str]:
        return BuildSystemArguments.__effective_mapping(default_environment, self.environment_set, self.environment_add, separator)

    def effective_properties(
            self, default_property: dict[str, str] = {}, separator: str = " ") -> dict[str, str]:
        return BuildSystemArguments.__effective_mapping(default_property, self.property_set, self.property_add, separator)


@dataclass
class FeatureConfigBoolean(DataClassYAMLMixin):
    enabled: BuildSystemArguments
    disabled: BuildSystemArguments


@dataclass
class BuildResult(DataClassYAMLMixin):
    directory: str
    docker_image: str
    features_boolean: list[FeatureType]
    features_select: dict[FeatureSelectionType, str] = field(default_factory=dict)


@dataclass
class LayerDepConfig(DataClassYAMLMixin):
    version: str
    arg_mapping: dict[str, str]


@dataclass
class RunConfig(DataClassYAMLMixin):
    working_directory: str
    project_name: str
    build_system: BuildSystem
    source_directory: str
    cpu_architecture: CPUArchitecture
    features_boolean: dict[FeatureType, FeatureConfigBoolean]
    features_select: dict[str, dict[str, BuildSystemArguments]]
    build_args: BuildSystemArguments
    # TODO: jrabil: what is this?
    additional_steps: list[str]
    layers_deps: dict[str, LayerDepConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str) -> RunConfig:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return cls.from_yaml(f)

    def save(self, config_path: str) -> None:
        with open(config_path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_instance(cls, instance):
        return cls.from_dict(instance.to_dict())


@dataclass
class DeployConfig(DataClassYAMLMixin):
    ir_image: str
    working_directory: str
    cpu_architecture: CPUArchitecture
    features_boolean: dict[FeatureType, bool]
    features_versions: dict[FeatureType, str]
    features_select: dict[str, str]
    docker_repository: str
    # FIXME: hide this config in image config
    # should be selected by features automatically
    layers_deps: dict[str, LayerDepConfig] = field(default_factory=dict)

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
        return cls.from_dict(instance.to_dict())


@dataclass
class SourceContainerConfig(DataClassYAMLMixin):
    working_directory: str
    source_directory: str
    project_name: str
    cpu_architecture: CPUArchitecture = CPUArchitecture.X86_64
    docker_repository: str = "spcleth/xaas-artifact"

    @staticmethod
    def load(config_path: str) -> SourceContainerConfig:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return SourceContainerConfig.from_yaml(f)

    def save(self, config_path: str) -> None:
        with open(config_path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_instance(cls, instance):
        return cls.from_dict(instance.to_dict())
        return obj


@dataclass
class SourceDeploymentConfigBaseImage(DataClassYAMLMixin):
    name: str
    provided_features: list[FeatureType]
    additional_commands: list[str]


@dataclass
class SourceDeploymentConfigSystem(DataClassYAMLMixin):
    name: str
    cpu_architecture: CPUArchitecture = CPUArchitecture.X86_64
    system_discovery: str | None = None
    base_image: SourceDeploymentConfigBaseImage | None = None


@dataclass
class ConfigSelection(DataClassYAMLMixin):
    vectorization_flags: str | None
    gpu_backends: str | None
    parallel_libraries: list[str]
    fft_libraries: list[str]
    linear_algebra_libraries: str | None
    compiler: str


@dataclass
class SourceDeploymentConfigMode(DataClassYAMLMixin):
    mode: SourceContainerMode
    # FIXME: remove that - replace with mode
    # predefined_config_string: str | None
    predefined_config: ConfigSelection | None = None
    automated_mode: SourceContainerAutomated | None = None


@dataclass
class SourceDeploymentConfig(DataClassYAMLMixin):
    source_container: str
    working_directory: str
    project_name: str
    language: Language
    system: SourceDeploymentConfigSystem
    mode: SourceDeploymentConfigMode
    docker_repository: str = "spcleth/xaas-artifact"

    @staticmethod
    def load(config_path: str) -> SourceDeploymentConfig:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return SourceDeploymentConfig.from_yaml(f)

    def save(self, config_path: str) -> None:
        with open(config_path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_instance(cls, instance):
        return cls.from_dict(instance.to_dict())
        return obj
