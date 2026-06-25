import logging
import os
import subprocess
from abc import abstractmethod, ABC
from dataclasses import dataclass
from io import BytesIO

import docker
from docker import DockerClient
from docker.models.containers import Container
from docker.models.images import Image


@dataclass
class VolumeMount:
    source: str
    target: str
    mode: str = "rw"


@dataclass
class DockerBuildFeatures:
    has_buildkit: bool = False
    dockerfile_supports_copy_link: bool = False
    dockerfile_supports_run_mount: bool = False


class BuildInterface(ABC):
    @abstractmethod
    def get_build_features(self) -> DockerBuildFeatures:
        pass

    @abstractmethod
    def build(
        self,
        path: str | None,
        dockerfile: str | None = None,
        dockerfile_content: list[str] | str | None = None,
        tag: str | None = None,
        build_args: dict[str, str] | None = None,
        platform: str | None = None,
        labels: dict[str, str] | None = None,
        show_progress: bool = False,
    ) -> Image:
        """
        Builds a Docker image.

        :param path: the path to the build context directory. If ``None``, no context will be sent (this requires setting ``dockerfile_content``)
        :param dockerfile: the name of the Dockerfile to use (optional). Ignored if ``dockerfile_content`` is set.
        :param dockerfile_content: the lines of the dockerfile's code (optional)
        :param tag: the tag to assign the build result (optional). If ``None`` (default), the resulting image will not be tagged.
        :return: a reference to the completed docker image
        """
        pass


class BuildxDispatcher(BuildInterface):
    def __init__(self, client: DockerClient, executable_path: str):
        self.client = client
        self.executable_path = executable_path

        # assert that the given docker executable is runnable
        subprocess.run(
            [self.executable_path, "buildx", "version"],
            capture_output=True, text=True, check=True)

    def get_build_features(self) -> DockerBuildFeatures:
        return DockerBuildFeatures(
            has_buildkit=True,
            # TODO: we probably shouldn't assume that buildkit always supports these features?
            dockerfile_supports_copy_link=True,
            dockerfile_supports_run_mount=True,
        )

    def build(
        self,
        path: str | None,
        dockerfile: str | None = None,
        dockerfile_content: list[str] | str | None = None,
        tag: str | None = None,
        build_args: dict[str, str] | None = None,
        platform: str | None = None,
        labels: dict[str, str] | None = None,
        show_progress: bool = False,
    ) -> Image:
        assert not (dockerfile and dockerfile_content), "dockerfile name and raw dockerfile aren't compatible"

        if not tag:
            raise RuntimeError("Buildx dispatcher doesn't support untagged images!")

        if isinstance(dockerfile_content, list):
            dockerfile_content = "\n".join(dockerfile_content)
        assert dockerfile_content is None or isinstance(dockerfile_content, str), str(dockerfile_content)

        cmdline_args = [self.executable_path, "buildx", "build"]

        for k, v in build_args or {}:
            cmdline_args.append(f"--build-arg={k}={v}")

        for k, v in labels or {}:
            cmdline_args.append(f"--label={k}={v}")

        if platform:
            cmdline_args.append(f"--platform={platform}")

        if tag:
            cmdline_args.append(f"--tag={tag}")

        if dockerfile:
            if not path:
                raise RuntimeError("dockerfile path argument requires a context path!")

            cmdline_args.append(f"--file={os.path.join(path, dockerfile)}")
        elif dockerfile_content:
            if path:
                cmdline_args.append("-f-")

        cmdline_args.append(path or "-")

        # actually build the image
        subprocess.run(
            cmdline_args,
            stdout=subprocess.PIPE,
            stderr=None if show_progress else subprocess.PIPE,
            input=dockerfile_content if dockerfile_content is not None else None,
            encoding="utf-8",
            check=True,
            text=True,
        )

        return self.client.images.get(tag)


class Runner(BuildInterface):
    # TODO: jrabil: remove docker_repository argument
    def __init__(self, docker_repository: str = ""):
        self.client = docker.from_env()
        self.uid = os.getuid()
        self.gid = os.getgid()
        self.docker_repository = docker_repository

    def try_get_buildkit_builder(self) -> BuildInterface | None:
        try:
            # try to run 'docker buildx version', which will fail if either the docker binary or buildx isn't available
            subprocess.run(
                ["docker", "buildx", "version"],
                capture_output=True, text=True, check=True)

            return BuildxDispatcher(self.client, "docker")
        except Exception as e:
            logging.warning(f"'docker buildx' is unavailable, this may affect build performance!\n{str(e)}")
            return None

    def get_build_features(self) -> DockerBuildFeatures:
        #return self.features
        return DockerBuildFeatures(
            has_buildkit=False,
            # TODO: jrabil: we should probably implement a better system for auto-detecting this
            dockerfile_supports_copy_link=True,
            dockerfile_supports_run_mount=False,
        )

    def build(
        self,
        path: str | None,
        dockerfile: str | None = None,
        dockerfile_content: list[str] | str | None = None,
        tag: str | None = None,
        build_args: dict[str, str] | None = None,
        platform: str | None = None,
        labels: dict[str, str] | None = None,
        show_progress: bool = False,
    ) -> Image:
        if isinstance(dockerfile_content, list):
            dockerfile_content = "\n".join(dockerfile_content)
        assert dockerfile_content is None or isinstance(dockerfile_content, str), str(dockerfile_content)

        try:
            image, _ = self.client.images.build(
                path=path,
                dockerfile=dockerfile,
                fileobj=BytesIO(dockerfile_content.encode()) if dockerfile_content is not None else None,
                tag=tag,
                buildargs=build_args or {},
                platform=(f"linux/{platform}" if platform is not None else None),
                labels=labels,
                rm=True,
            )
            return image
        except docker.errors.APIError as e:
            logging.error(f"Docker API error: {str(e)}")
            raise
        except docker.errors.BuildError as e:
            logging.error(f"Docker build error: {str(e)}")
            for it in e.build_log:
                logging.error(f"\t{it}")
            raise

    def run(
        self,
        image: str,
        command: str | list[str],
        working_dir: str,
        mounts: list[VolumeMount] | None = None,
        detach: bool = True,
        remove: bool = True,
        tty: bool = False,
        environment: dict[str, str] | None = None,
    ) -> Container:
        try:
            volumes = {}
            if mounts:
                for mount in mounts:
                    volumes[mount.source] = {"bind": mount.target, "mode": mount.mode}
            logging.debug(f"Starting container from image '{image}'")

            # TODO: jrabil: we should make the base docker image be defined explicitly
            if self.docker_repository:
                image = f"{self.docker_repository}:{image}"

            envs = {
                "USER_ID": str(self.uid),
                "GROUP_ID": str(self.gid),
            }

            if environment is not None:
                envs |= environment

            container = self.client.containers.run(
                image=image,
                command=command,
                environment=envs,
                # user=f"{self.uid}:{self.gid}",
                volumes=volumes,
                detach=detach,
                remove=remove,
                tty=tty,
                working_dir=working_dir,
            )

            logging.info(f"Container started with ID: {container.id}")
            return container

        except docker.errors.ImageNotFound:
            logging.error(f"Image not found: {image}")
            raise
        except docker.errors.APIError as e:
            logging.error(f"Docker API error: {str(e)}")
            raise

    def exec_run(
        self, container: Container, command: list[str], working_dir: str
    ) -> tuple[int, str]:
        try:
            return container.exec_run(command, workdir=working_dir)
        except docker.errors.APIError as e:
            logging.error(f"Docker API error: {str(e)}")
            raise

    def get_image(self, image: str) -> Image:
        return self.client.images.get(image)

    def get_image_env(self, image: str | Image) -> dict[str, str]:
        if isinstance(image, str):
            image = self.get_image(image)

        result: dict[str, str] = {}
        for item in image.attrs["Config"]["Env"]:
            name, value = item.split("=", 1)
            result[name] = value
        return result
