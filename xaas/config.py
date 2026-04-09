from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path

import yaml
from mashumaro.config import BaseConfig
from mashumaro.mixins.yaml import DataClassYAMLMixin

from xaas.util.dict_utils import union_distinct, union_merge


class BaseXaasConfigModel(DataClassYAMLMixin):
    class Config(BaseConfig):
        forbid_extra_keys = True


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
class DockerLayerVersion(BaseXaasConfigModel):
    flag_name: str
    build_args: dict[str, str]


@dataclass
class DockerLayer(BaseXaasConfigModel):
    dockerfile: str
    image_tag: str
    versions: list[str]
    version_arg: str
    build_location: str
    runtime_location: str
    arg_mapping: dict[str, DockerLayerVersion] | None = None
    envs: dict[str, str] = field(default_factory=dict)


@dataclass
class DockerLayers(BaseXaasConfigModel):
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
        self.ir_type: IRType
        self.default_builder_image: str
        self.default_runtime_image: str
        self.parallelism_level: int
        self.layers: DockerLayers

    def initialize(self, config_path: str) -> None:
        if self._initialized:
            raise RuntimeError("Configuration already initialized")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        self.parallelism_level = config_data["parallelism_level"]
        self.default_builder_image = config_data["default_builder_image"]
        self.default_runtime_image = config_data["default_runtime_image"]

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


class ArgumentsVariableEntryType(Enum):
    APPEND = "append"
    SET = "set"


@dataclass
class ArgumentsVariableEntry(BaseXaasConfigModel):
    type: ArgumentsVariableEntryType
    value: str
    separator: str = ""

    class Config(BaseXaasConfigModel.Config):
        # Don't include separator when it's left as the default
        omit_default = True

    @classmethod
    def __pre_deserialize__(cls, arg):
        if isinstance(arg, str):
            # For convenience, a plain string value can be implicitly converted into a 'SET' entry
            return { "type": "set", "value": arg }
        else:
            return arg

    @staticmethod
    def merge(a: ArgumentsVariableEntry, b: ArgumentsVariableEntry) -> ArgumentsVariableEntry:
        # We can combine two set directives if they have the same value
        if a.type is ArgumentsVariableEntryType.SET and b.type is ArgumentsVariableEntryType.SET:
            if a.value != b.value:
                raise RuntimeError(f"Cannot set variable with different values: '{a.value}' and '{b.value}'")

            return ArgumentsVariableEntry(
                type=ArgumentsVariableEntryType.SET,
                value=a.value,
            )

        # We can combine two append directives as long as they both use the same separator
        if a.type is ArgumentsVariableEntryType.APPEND and b.type is ArgumentsVariableEntryType.APPEND:
            if a.separator != b.separator:
                raise RuntimeError(f"Mismatched separators: '{a.separator}' and '{b.separator}'")

            return ArgumentsVariableEntry(
                type=ArgumentsVariableEntryType.APPEND,
                value=f"{a.value}{b.separator}{b.value}",
                separator=a.separator,
            )

        # Refuse to handle any other combination
        raise RuntimeError(f"Mismatched variable update types: {a} and {b}")

    def to_shell_export(self, name: str) -> list[str]:
        """
        Gets a shell 'export' command which sets or updates this variable's value.

        Returns a single quoted shell command.
        """

        if self.type is ArgumentsVariableEntryType.APPEND:
            return [ "export", f'{name}="${{{name}}}${{{name}:+{self.separator}}}"{shlex.quote(self.value)}' ]
        elif self.type is ArgumentsVariableEntryType.SET:
            return [ "export", f'{name}={shlex.quote(self.value)}' ]
        else:
            raise RuntimeError(f"Unknown type: {self.type}")

    @staticmethod
    def reduce_to_dict(entries: dict[str, ArgumentsVariableEntry], defaults: dict[str, str]) -> dict[str, str]:
        """
        Reduces a group of argument variables down to a single dict containing the effective key-value mappings.

        :param entries: the argument variable entries
        :param defaults: a dict containing the inherited initial variable values
        """

        result: dict[str, str] = {}
        for name, entry in entries.items():
            if entry.type is ArgumentsVariableEntryType.APPEND:
                if name in defaults:
                    result[name] = f"{defaults[name]}{entry.separator}{entry.value}"
                else:
                    result[name] = entry.value
            elif entry.type is ArgumentsVariableEntryType.SET:
                result[name] = entry.value
            else:
                raise RuntimeError(f"Unknown type: {entry.type}")
        return result

    @staticmethod
    def reduce_to_cmake(entries: dict[str, ArgumentsVariableEntry]) -> list[str]:
        """
        Reduces a group of argument variables down to a list of CMake code lines.

        :param entries: the argument variable entries
        """

        result: list[str] = []
        for name, entry in entries.items():
            if entry.type is ArgumentsVariableEntryType.SET:
                result.extend([
                    f"if (NOT DEFINED {name})",
                    f"    set({name} \"{entry.value}\")",
                    f"else()",
                    f"    message(FATAL_ERROR, \"CMake variable {name} was already set to '${{{name}}}'\")",
                    f"endif()",
                ])
            elif entry.type is ArgumentsVariableEntryType.APPEND:
                result.extend([
                    f"if (NOT DEFINED {name})",
                    f"    set({name} \"{entry.value}\")",
                    f"else()",
                    f"    set({name} \"${{{name}}}{entry.separator}{entry.value}\")",
                    f"endif()",
                ])
            else:
                raise RuntimeError(f"Unknown type: {entry.type}")
        return result


@dataclass
class BuildSystemArguments(BaseXaasConfigModel):
    environment: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)
    property: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)

    arguments: list[str] = field(default_factory=list)

    @staticmethod
    def merge(a: BuildSystemArguments, b: BuildSystemArguments) -> BuildSystemArguments:
        return BuildSystemArguments(
            environment = union_merge(a.environment, b.environment, ArgumentsVariableEntry.merge),
            property = union_merge(a.property, b.property, ArgumentsVariableEntry.merge),

            # simply concatenate any additional arguments
            arguments = a.arguments + b.arguments,
        )


@dataclass
class FeatureConfigBoolean(BaseXaasConfigModel):
    enabled: BuildSystemArguments
    disabled: BuildSystemArguments

    def args_for_state(self, state: bool) -> BuildSystemArguments:
        return self.enabled if state else self.disabled


@dataclass
class BuildResult(BaseXaasConfigModel):
    directory: str
    docker_image: str
    features_boolean: dict[FeatureType, bool]
    features_select: dict[str, str]


@dataclass
class LayerDepConfig(BaseXaasConfigModel):
    version: str
    arg_mapping: dict[str, str]


@dataclass
class PartialRunConfig(BaseXaasConfigModel):
    """
    Defines a set of features and other build system arguments for a RunConfig.
    """

    features_boolean: dict[FeatureType, FeatureConfigBoolean]
    features_select: dict[str, dict[str, BuildSystemArguments]]
    build_args: BuildSystemArguments

    @staticmethod
    def merge(a: PartialRunConfig, b: PartialRunConfig) -> PartialRunConfig:
        return PartialRunConfig(
            features_boolean = union_distinct(a.features_boolean, b.features_boolean),
            features_select = union_distinct(a.features_select, b.features_select),
            build_args = BuildSystemArguments.merge(a.build_args, b.build_args),
        )


@dataclass
class RunConfig(DataClassYAMLMixin):
    working_directory: str
    project_name: str
    build_system: BuildSystem
    source_directory: str
    cpu_architecture: CPUArchitecture
    all_targets: PartialRunConfig
    cpu_specific: dict[CPUArchitecture, PartialRunConfig]
    additional_steps: list[list[str]]
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

    def for_target(self, cpu_architecture: CPUArchitecture) -> PartialRunConfig:
        """
        Gets the PartialRunConfig for a specific CPU architecture.
        """
        if cpu_architecture in self.cpu_specific:
            return PartialRunConfig.merge(self.all_targets, self.cpu_specific[cpu_architecture])
        else:
            return self.all_targets

    def may_contain_feature_boolean(self, feature: FeatureType) -> bool:
        """
        Checks if any permutations of this configuration may contain the given boolean feature.
        """

        if self.all_targets.features_boolean.get(feature, False):
            return True

        for cfg in self.cpu_specific.values():
            if cfg.features_boolean.get(feature, False):
                return True

        return False


@dataclass
class DeployConfig(BaseXaasConfigModel):
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
class SourceContainerConfig(BaseXaasConfigModel):
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
class SourceDeploymentConfigBaseImage(BaseXaasConfigModel):
    name: str
    provided_features: list[FeatureType]
    additional_commands: list[str]


@dataclass
class SourceDeploymentConfigSystem(BaseXaasConfigModel):
    name: str
    cpu_architecture: CPUArchitecture = CPUArchitecture.X86_64
    system_discovery: str | None = None
    base_image: SourceDeploymentConfigBaseImage | None = None


@dataclass
class ConfigSelection(BaseXaasConfigModel):
    vectorization_flags: str | None
    gpu_backends: str | None
    parallel_libraries: list[str]
    fft_libraries: list[str]
    linear_algebra_libraries: str | None
    compiler: str


@dataclass
class SourceDeploymentConfigMode(BaseXaasConfigModel):
    mode: SourceContainerMode
    # FIXME: remove that - replace with mode
    # predefined_config_string: str | None
    predefined_config: ConfigSelection | None = None
    automated_mode: SourceContainerAutomated | None = None


@dataclass
class SourceDeploymentConfig(BaseXaasConfigModel):
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
