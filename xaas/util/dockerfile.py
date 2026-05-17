import itertools
import shlex
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


def _to_dockerfile_quoted_list(l: list[str]) -> str:
    quoted_parts = [ f'"{s.replace('\\', '\\\\').replace('"', '\\"')}"' for s in l ]
    return f"[{", ".join(quoted_parts)}]"


def _command_shell_or_exec_form_to_dockerfile(command: str | list[str]) -> str:
    if isinstance(command, str):
        return command
    else:
        return _to_dockerfile_quoted_list(command)


class DockerfileStep:
    def to_dockerfile_line(self) -> str:
        raise NotImplementedError


@dataclass
class CmdStep(DockerfileStep):
    NAME: ClassVar[str] = "CMD"

    # shell form if str, exec form if list[str]
    command: str | list[str]

    def to_dockerfile_line(self) -> str:
        return f"{self.NAME} {_command_shell_or_exec_form_to_dockerfile(self.command)}"


@dataclass
class CopyStep(DockerfileStep):
    NAME: ClassVar[str] = "COPY"

    src: str | list[str]
    dst: str
    from_context: str | None = None
    link: bool = False

    def to_dockerfile_line(self) -> str:
        paths: list[str]
        if isinstance(self.src, str):
            paths = [ self.src, self.dst ]
        else:
            paths = [ *self.src, self.dst ]

        return f"{self.NAME} {"--link" if self.link else ""} {f"--from={self.from_context}" if self.from_context else ""} {_to_dockerfile_quoted_list(paths)}"


@dataclass
class EntrypointStep(DockerfileStep):
    NAME: ClassVar[str] = "ENTRYPOINT"

    # shell form if str, exec form if list[str]
    command: str | list[str]

    def to_dockerfile_line(self) -> str:
        return f"{self.NAME} {_command_shell_or_exec_form_to_dockerfile(self.command)}"


@dataclass
class EnvStep(DockerfileStep):
    NAME: ClassVar[str] = "ENV"

    # values must already be quoted
    vars: dict[str, str]

    def to_dockerfile_line(self) -> str:
        return f"{self.NAME} {" ".join([ f"{k}={v}" for k, v in self.vars.items() ])}"


@dataclass
class RunStep(DockerfileStep):
    NAME: ClassVar[str] = "RUN"

    @dataclass
    class Mount:
        # @staticmethod
        # def apt_cache_mounts() -> list[Mount]:
        #     return [
        #         RunStep.CacheMount(target="/var/cache/apt", sharing=RunStep.CacheMount.Sharing.LOCKED),
        #         RunStep.CacheMount(target="/var/lib/apt", sharing=RunStep.CacheMount.Sharing.LOCKED),
        #         RunStep.TmpfsMount(target="/var/log/apt"),
        #     ]

        def to_dockerfile_run_option(self) -> str:
            raise NotImplementedError


    @dataclass
    class BindMount(Mount):
        TYPE: ClassVar[str] = "bind"

        target: str
        source: str | None = None
        from_context: str | None = None
        rw: bool = False

        def to_dockerfile_run_option(self) -> str:
            opts: list[str] = []
            if self.from_context is not None:
                opts.append(f"from={shlex.quote(self.from_context)}")
            if self.source is not None:
                opts.append(f"source={shlex.quote(self.source)}")
            opts.append(f"target={shlex.quote(self.target)}")
            if self.rw:
                opts.append("rw")
            return f"--mount,type={self.TYPE},{",".join(opts)}"


    @dataclass
    class CacheMount(Mount):
        TYPE: ClassVar[str] = "cache"

        class Sharing(Enum):
            SHARED = "shared"
            PRIVATE = "private"
            LOCKED = "locked"


        target: str
        source: str | None = None
        id: str | None = None
        sharing: Sharing = Sharing.SHARED
        from_context: str | None = None
        ro: bool = False

        def to_dockerfile_run_option(self) -> str:
            opts: list[str] = []
            if self.id is not None:
                opts.append(f"id={shlex.quote(self.id)}")
            if self.from_context is not None:
                opts.append(f"from={shlex.quote(self.from_context)}")
            if self.source is not None:
                opts.append(f"source={shlex.quote(self.source)}")
            opts.append(f"target={shlex.quote(self.target)}")
            opts.append(f"sharing={self.sharing.value}")
            if self.ro:
                opts.append("ro")
            return f"--mount,type={self.TYPE},{",".join(opts)}"


    @dataclass
    class TmpfsMount(Mount):
        TYPE: ClassVar[str] = "tmpfs"

        target: str

        def to_dockerfile_run_option(self) -> str:
            return f"--mount,type={self.TYPE},target={shlex.quote(self.target)}"


    # shell form if str, exec form if list[str]
    command: str | list[str]
    mounts: list[Mount] | None = None

    def to_dockerfile_line(self) -> str:
        mounts = [ mount.to_dockerfile_run_option() for mount in (self.mounts or []) ]
        return f"{self.NAME} {" ".join(mounts)} {_command_shell_or_exec_form_to_dockerfile(self.command)}"

    def get_command_in_shell_form(self) -> str:
        if isinstance(self.command, str):
            return self.command
        else:
            return shlex.join(self.command)


@dataclass
class DockerfileStage:
    from_context: str
    steps: list[DockerfileStep]

    def to_dockerfile_lines(self, name: str) -> list[str]:
        return [
            f"FROM {self.from_context} AS {name}",
            *[ step.to_dockerfile_line() for step in self.steps ],
            "",
        ]


@dataclass
class Dockerfile:
    stages: list[tuple[str | None, DockerfileStage]]
    syntax: str | None = None

    def to_dockerfile_lines(self) -> list[str]:
        """
        Converts this :py:class:`~Dockerfile` to a list of lines which can be written out to an actual dockerfile on disk.

        :return: the dockerfile lines
        """

        return [
            f"# syntax=docker/dockerfile:{self.syntax or "1"}",
            "",
            *itertools.chain.from_iterable([ stage.to_dockerfile_lines(name) for name, stage in self.stages ]),
        ]


class DockerfileBuilder:
    stages: OrderedDict[str, DockerfileStage]

    def __init__(self):
        self.stages = OrderedDict()

    def add_stage(self, stage: DockerfileStage, name: str | None = None) -> str:
        """
        Adds the given stage to this dockerfile with the given name, ensuring that the stage name is unique.

        :param stage: the :py:class:`~DockerfileStage` to add
        :param name: the stage's name (optional). If ``Ǹone`` (default), a unique name will be generated.
        :return: the stage's name
        :raises RuntimeError: if this dockerfile already contains a stage with the given name
        """

        if name is None:
            i = len(self.stages)
            while f"stage_{i}" not in self.stages:
                i = i + 1
            name = f"stage_{i}"

        if name in self.stages:
            raise RuntimeError(f"dockerfile already contains a stage named '{name}' ({self.stages.keys()})")

        self.stages[name] = stage
        return name

    def build(self, terminal_stage: DockerfileStage) -> Dockerfile:
        """
        Constructs a :py:class:´~Dockerfile´ from this builder's current state and the provided terminal stage.

        Note that the returned :py:class:´~Dockerfile´ will refer to the same :py:class:´~DockerfileStage´ instances as
        this builder instance, so any subsequent modifications to them will be visible in the result.

        :param terminal_stage: the final stage in the resulting dockerfile, which doesn't have a name
        :return: the new :py:class:´~Dockerfile´ instance
        """

        return Dockerfile(
            stages=[
                *list(self.stages.items()),
                ( None, terminal_stage ),
            ])
