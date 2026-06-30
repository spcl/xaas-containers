from __future__ import annotations

import dataclasses
import os
import shlex
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import expandvars
import yaml
from mashumaro.config import BaseConfig
from mashumaro.mixins.yaml import DataClassYAMLMixin
from mashumaro.types import Discriminator

from xaas.docker import DockerRunner
from xaas.util.dict_utils import union_distinct, union_merge
from xaas.util.dockerfile import DockerfileStage, DockerfileStep, EnvStep, CopyStep, RunStep, DockerfileBuilder


def _variable_expand(raw: str, mapping: dict[str, str]) -> str:
    # wrap mapping in MappingProxyType to disallow '${NAME:=value}' syntax from modifying global state
    return expandvars.expand(raw, nounset=True, environ=MappingProxyType(mapping))


def _variable_escape(raw: str) -> str:
    return raw.replace("\\", "\\\\").replace("$", "\\$")


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

    def to_shell(self, name: str) -> str:
        """
        Gets a shell directive which sets or updates this variable's value.

        This may be prepended directly to a command to set the value for a specific command invocation, or prefixed
        with ``export`` to set the value globally.

        :return: a single quoted shell directive
        """

        if self.type is ArgumentsVariableEntryType.APPEND:
            return f'{name}="${{{name}}}${{{name}:+{self.separator}}}"{shlex.quote(self.value)}'
        elif self.type is ArgumentsVariableEntryType.SET:
            return f'{name}={shlex.quote(self.value)}'
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
    def reduce_to_cmake_args(entries: dict[str, ArgumentsVariableEntry]) -> list[str]:
        """
        Reduces a group of argument variables down to a list of unquoted shell arguments which may be passed to a CMake command.

        Note that this doesn't support appending onto any initial default values.

        :param entries: the argument variable entries
        """

        return [ f"-D{name}={entry.value}" for name, entry in entries.items() ]

    @staticmethod
    def reduce_to_dockerfile_env(entries: dict[str, ArgumentsVariableEntry]) -> dict[str, str]:
        """
        Reduces a group of argument variables down to a single dict containing escaped key-value mappings, suitable to include in a Dockerfile ENV directive.

        :param entries: the argument variable entries
        """

        result: dict[str, str] = {}
        for name, entry in entries.items():
            if entry.type is ArgumentsVariableEntryType.APPEND:
                result[name] = f'${{{name}}}${{{name}:+{entry.separator}}}{shlex.quote(entry.value)}'
            elif entry.type is ArgumentsVariableEntryType.SET:
                result[name] = shlex.quote(entry.value)
            else:
                raise RuntimeError(f"Unknown type: {entry.type}")
        return result


@dataclass
class DockerLayerVersion(BaseXaasConfigModel):
    flag_name: str


@dataclass
class DockerLayerCopyStep(BaseXaasConfigModel):
    """
    A specification of a single path for a dependency which will be copied into the resulting docker image.

    This works like the Dockerfile COPY directive.
    """

    image_tag: str
    src_path: str
    dst_path: str


@dataclass
class DockerLayerPrepared(BaseXaasConfigModel):
    """
    A full specification of the Docker layers (and other properties) for a specific version of a dependency.
    """

    name: str

    builder_paths: list[DockerLayerCopyStep]
    runtime_paths: list[DockerLayerCopyStep]

    builder_env: dict[str, ArgumentsVariableEntry]
    runtime_env: dict[str, ArgumentsVariableEntry]


# TODO: jrabil: get rid of the many unnecessary fields here
#   (they're still here for now because they're used by the source container stuff, we should probably rewrite that to use the new dockerfile generation infrastructure as IR containers)
# TODO: jrabil: maybe turn the 'version' into an ordinary build argument?
@dataclass
class DockerLayerTemplate(BaseXaasConfigModel):
    """
    A full specification of the Docker layers (and other properties) for a dependency which may support multiple versions or other arguments.
    """

    versions: list[str]
    version_arg: str

    image_tag: str | None = None # TODO: jrabil: remove this
    build_location: str | None = None # TODO: jrabil: remove this
    runtime_location: str | None = None # TODO: jrabil: remove this
    arg_mapping: dict[str, DockerLayerVersion] | None = None
    envs: dict[str, str] = field(default_factory=dict) # TODO: jrabil: remove this

    all_paths: list[DockerLayerCopyStep] = field(default_factory=list)
    builder_paths: list[DockerLayerCopyStep] = field(default_factory=list)
    runtime_paths: list[DockerLayerCopyStep] = field(default_factory=list)

    all_env: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)
    builder_env: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)
    runtime_env: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)

    def _prepare_copy_steps(self, variable_mapping: dict[str, str], steps: list[DockerLayerCopyStep]) -> list[DockerLayerCopyStep]:
        return [dataclasses.replace(
            step,
            image_tag=_variable_expand(step.image_tag, variable_mapping),
            src_path=_variable_expand(step.src_path, variable_mapping),
            dst_path=_variable_expand(step.dst_path, variable_mapping),
        ) for step in steps]

    def _prepare_variable_entries(self, variable_mapping: dict[str, str], entries: dict[str, ArgumentsVariableEntry]) -> dict[str, ArgumentsVariableEntry]:
        return {name: dataclasses.replace(
            entry,
            value=_variable_expand(entry.value, variable_mapping),
        ) for name, entry in entries.items()}

    def prepare(
            self, name: str, version: str, arg_mapping: dict[str, str] | None,
            variable_ctx: dict[str, str],
            states_boolean: dict[FeatureType, bool], states_select: dict[str, str]
    ) -> DockerLayerPrepared:
        variable_mapping = variable_ctx | (arg_mapping or {}) | {self.version_arg: version}

        # insert variable mappings for all the items in self.arg_mappings
        for arg, value in (self.arg_mapping or {}).items():
            mapped_arg = arg_mapping[arg] if arg_mapping is not None else arg
            variable_mapping[value.flag_name] = states_select[mapped_arg]

        all_paths_prepared = self._prepare_copy_steps(variable_mapping, self.all_paths)
        builder_paths_prepared = self._prepare_copy_steps(variable_mapping, self.builder_paths)
        runtime_paths_prepared = self._prepare_copy_steps(variable_mapping, self.runtime_paths)

        all_env_prepared = self._prepare_variable_entries(variable_mapping, self.all_env)
        builder_env_prepared = self._prepare_variable_entries(variable_mapping, self.builder_env)
        runtime_env_prepared = self._prepare_variable_entries(variable_mapping, self.runtime_env)

        return DockerLayerPrepared(
            name=name,

            builder_paths=all_paths_prepared + builder_paths_prepared,
            runtime_paths=all_paths_prepared + runtime_paths_prepared,

            builder_env=union_merge(all_env_prepared, builder_env_prepared, ArgumentsVariableEntry.merge),
            runtime_env=union_merge(all_env_prepared, runtime_env_prepared, ArgumentsVariableEntry.merge),
        )


@dataclass
class DockerLayers(BaseXaasConfigModel):
    layers: dict[CPUArchitecture, dict[str, DockerLayerTemplate]]

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
        self.config_vars: dict[str, str]
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

        self.config_vars = config_data["config_vars"]
        self.parallelism_level = config_data["parallelism_level"]
        self.default_builder_image = _variable_expand(config_data["default_builder_image"], self.config_vars)
        self.default_runtime_image = _variable_expand(config_data["default_runtime_image"], self.config_vars)

        match config_data["ir_type"]:
            case IRType.LLVM_IR.value:
                self.ir_type = IRType.LLVM_IR
            case _:
                raise ValueError(f"Unsupported IR type: {config_data['ir_type']}")

        self.layers = DockerLayers.load(XaaSConfig.LAYERS_CONFIGURATION)

        self._initialized = True


@dataclass
class LayerDepBase(BaseXaasConfigModel):
    class Config(BaseXaasConfigModel.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)

    # For some reason the type is ignored when serializing this class, so we'll add it manually
    def __post_serialize__(self, d: dict) -> dict:
        d['type'] = self.type
        return d

    def prepare(
            self, cpu_architecture: CPUArchitecture,
            states_boolean: dict[FeatureType, bool], states_select: dict[str, str]
    ) -> DockerLayerPrepared:
        raise NotImplementedError


@dataclass
class LayerDepReference(LayerDepBase):
    type: ClassVar[str] = "default"

    name: str
    version: str
    arg_mapping: dict[str, str] | None = None

    def prepare(
            self, cpu_architecture: CPUArchitecture,
            states_boolean: dict[FeatureType, bool], states_select: dict[str, str]
    ) -> DockerLayerPrepared:
        return XaaSConfig().layers.layers[cpu_architecture][self.name].prepare(
            self.name, self.version, self.arg_mapping,
            XaaSConfig().config_vars,
            states_boolean, states_select)


@dataclass
class LayerDepCustom(LayerDepBase):
    type: ClassVar[str] = "custom"

    name: str
    version: str | None = None

    all_paths: list[DockerLayerCopyStep] = field(default_factory=list)
    builder_paths: list[DockerLayerCopyStep] = field(default_factory=list)
    runtime_paths: list[DockerLayerCopyStep] = field(default_factory=list)

    all_env: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)
    builder_env: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)
    runtime_env: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)

    def prepare(
            self, cpu_architecture: CPUArchitecture,
            states_boolean: dict[FeatureType, bool], states_select: dict[str, str]
    ) -> DockerLayerPrepared:
        return DockerLayerPrepared(
            name=self.name,

            builder_paths=self.all_paths + self.builder_paths,
            runtime_paths=self.all_paths + self.runtime_paths,

            builder_env=union_merge(self.all_env, self.builder_env, ArgumentsVariableEntry.merge),
            runtime_env=union_merge(self.all_env, self.runtime_env, ArgumentsVariableEntry.merge),
        )


@dataclass
class BuildSystemArguments(BaseXaasConfigModel):
    environment: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)
    property: dict[str, ArgumentsVariableEntry] = field(default_factory=dict)

    arguments: list[str] = field(default_factory=list)

    dependencies: list[LayerDepBase] = field(default_factory=list)

    @staticmethod
    def merge(a: BuildSystemArguments, b: BuildSystemArguments) -> BuildSystemArguments:
        return BuildSystemArguments(
            environment = union_merge(a.environment, b.environment, ArgumentsVariableEntry.merge),
            property = union_merge(a.property, b.property, ArgumentsVariableEntry.merge),

            # simply concatenate any additional arguments
            arguments = a.arguments + b.arguments,

            # simply concatenate any additional dependencies
            dependencies = a.dependencies + b.dependencies,
        )


@dataclass
class FeatureConfigBoolean(BaseXaasConfigModel):
    enabled: BuildSystemArguments
    disabled: BuildSystemArguments

    def args_for_state(self, state: bool) -> BuildSystemArguments:
        return self.enabled if state else self.disabled


@dataclass
class PartialRunConfig(BaseXaasConfigModel):
    """
    Defines a set of features and other build system arguments for a RunConfig.
    """

    features_boolean: dict[FeatureType, FeatureConfigBoolean]
    features_select: dict[str, dict[str, BuildSystemArguments]]
    build_args: BuildSystemArguments

    builder_image: str | None = None
    runtime_image: str | None = None

    @staticmethod
    def merge(a: PartialRunConfig, b: PartialRunConfig) -> PartialRunConfig:
        return PartialRunConfig(
            features_boolean = union_distinct(a.features_boolean, b.features_boolean),
            features_select = union_distinct(a.features_select, b.features_select),
            build_args = BuildSystemArguments.merge(a.build_args, b.build_args),

            builder_image = a.builder_image or b.builder_image,
            runtime_image = a.runtime_image or b.runtime_image,
        )

    def effective_docker_images(self) -> tuple[str, str]:
        effective_builder_image = self.builder_image or XaaSConfig().default_builder_image
        effective_runtime_image = self.runtime_image or XaaSConfig().default_runtime_image
        return effective_builder_image, effective_runtime_image


@dataclass
class RunConfig(DataClassYAMLMixin):
    working_directory: str
    project_name: str
    build_system: BuildSystem
    source_directory: str
    cpu_architectures: list[CPUArchitecture]
    all_targets: PartialRunConfig
    cpu_specific: dict[CPUArchitecture, PartialRunConfig]
    additional_steps: list[list[str]]

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
class DerivedDockerImageDescriptor(BaseXaasConfigModel):
    """
    Describes a Docker image which may have some modifications.
    """

    base_image: str
    paths: list[DockerLayerCopyStep] | None = None
    env: dict[str, ArgumentsVariableEntry] | None = None

    @staticmethod
    def create_builder_and_runtime(
            builder_base_image: str,
            runtime_base_image: str,
            layers: list[DockerLayerPrepared],
    ) -> tuple[DerivedDockerImageDescriptor, DerivedDockerImageDescriptor]:
        builder_paths = None
        runtime_paths = None

        builder_env = None
        runtime_env = None

        for layer in layers:
            if len(layer.builder_paths) > 0:
                builder_paths = (builder_paths or []) + layer.builder_paths
            if len(layer.runtime_paths) > 0:
                runtime_paths = (runtime_paths or []) + layer.runtime_paths

            if layer.builder_env:
                builder_env = union_merge(builder_env or {}, layer.builder_env, ArgumentsVariableEntry.merge)
            if layer.runtime_env:
                runtime_env = union_merge(runtime_env or {}, layer.runtime_env, ArgumentsVariableEntry.merge)

        builder_descriptor = DerivedDockerImageDescriptor(base_image=builder_base_image, paths=builder_paths, env=builder_env)
        runtime_descriptor = DerivedDockerImageDescriptor(base_image=runtime_base_image, paths=runtime_paths, env=runtime_env)
        return builder_descriptor, runtime_descriptor

    def prepared_dockerfile_stage(self) -> DockerfileStage:
        """
        Gets a :py:class:`~xaas.util.dockerfile.DockerfileStage` which will result in an image fully configured according to this descriptor.

        Specifically, the resulting image will contain all necessary environment variables and have all paths copied.
        """

        steps: list[DockerfileStep] = []

        if self.env:
            steps.append(EnvStep(ArgumentsVariableEntry.reduce_to_dockerfile_env(self.env)))

        for path in (self.paths or []):
            steps.append(CopyStep(
                from_context=path.image_tag,
                source=path.src_path,
                target=path.dst_path,
                link=True
            ))

        return DockerfileStage(from_context=self.base_image, steps=steps)

    def run_in_prepared_context(self, dockerfile_builder: DockerfileBuilder, orig_run_step: RunStep) -> DockerfileStage:
        """
        Tries to adapt the given :py:class:`~xaas.util.dockerfile.RunStep` so that it will run in a context equivalent
        to an image fully configured according to this descriptor. This assumes that the given
        :py:class:`~xaas.util.dockerfile.RunStep` is being executed in a stage whose base image is :py:attr:`~.base_image`.

        A return value of ``None`` indicates that it is not possible to represent the derived image using only bind
        mounts and shell environment overrides, in which case the :py:class:`~xaas.util.dockerfile.RunStep` should be
        executed in a stage whose base image is :py:func:`~.prepared_dockerfile_stage`.

        :param dockerfile_builder: the :py:class:`~xaas.util.dockerfile.DockerfileBuilder` to use if additional stages need to be built
        :param orig_run_step: the original :py:class:`~xaas.util.dockerfile.RunStep` to adapt
        :return: the adapted :py:class:`~xaas.util.dockerfile.RunStep`, or ``Ǹone`` if not possible
        """

        # if None, we'll have to fall back to preparing the whole derived image in a separate stage
        inline_run_step: RunStep | None = orig_run_step

        if inline_run_step is not None and self.paths:
            # add bind mount entries for each of the requested paths
            # TODO: jrabil: in some cases this may be impossible, in which case we should set inline_run_step to None

            bind_mounts = [ RunStep.BindMount(
                from_context=path.image_tag,
                source=path.src_path,
                target=path.dst_path,
            ) for path in self.paths ]

            inline_run_step = dataclasses.replace(inline_run_step, mounts=bind_mounts + (inline_run_step.mounts or []))

        if inline_run_step is not None and self.env:
            # prepend the command with some shell code which exports the updated environment variables
            # (this always converts the command to shell form)

            export_statements = "".join([ f"export {entry.to_shell(name)}; " for name, entry in self.env.items()])

            inline_run_step = dataclasses.replace(inline_run_step, command=export_statements + inline_run_step.get_command_in_shell_form())

        if inline_run_step is not None:
            # the derived image is similar enough to the base image that it can be described inline in the RUN step
            return DockerfileStage(
                from_context=self.base_image,
                steps=[ inline_run_step ],
            )

        # TODO: jrabil: the prepared image might have already been built if this descriptor is a child of a BuildResult, ideally we should re-use that whenever possible

        # fallback: prepare the whole derived docker image in a separate stage
        prepared_stage = self.prepared_dockerfile_stage()
        prepared_stage_name = dockerfile_builder.add_stage(prepared_stage)

        return DockerfileStage(
            from_context=prepared_stage_name,
            steps=[ orig_run_step ],
        )

    def build_prepared_image(self, docker_runner: DockerRunner) -> str:
        """
        Gets the tag of a Docker image fully configured according to this descriptor. This may cause the image to be
        built if necessary.

        :param docker_runner: the Docker API client instance
        :param empty_dir_path: an absolute filesystem path to an empty directory
        :return: a Docker image tag
        """

        if not self.env and not self.paths:
            # the derived image is exactly the same as the base
            return self.base_image

        dockerfile_lines = DockerfileBuilder().build(self.prepared_dockerfile_stage()).to_lines()

        return docker_runner.build(
            path=None,
            dockerfile_content=dockerfile_lines,
            tag=None,
        ).id


@dataclass
class BuildResult(BaseXaasConfigModel):
    directory: str

    features_boolean: dict[FeatureType, bool]
    features_select: dict[str, str]

    builder_image: DerivedDockerImageDescriptor
    runtime_image: DerivedDockerImageDescriptor

    # these are tags for the docker images which have been pre-staged to build the image, if any
    prepared_builder_image: str | None = None
    prepared_runtime_image: str | None = None


@dataclass
class DeployConfig(BaseXaasConfigModel):
    ir_image: str
    working_directory: str
    cpu_architecture: CPUArchitecture
    features_boolean: dict[FeatureType, bool]
    features_select: dict[str, str]
    docker_repository: str

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
