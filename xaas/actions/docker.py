import logging
import os
from dataclasses import dataclass

import docker
from docker.models.containers import Container


@dataclass
class VolumeMount:
    source: str
    target: str
    mode: str = "rw"


class Runner:
    def __init__(self, docker_repository: str):
        self.client = docker.from_env()
        self.uid = os.getuid()
        self.gid = os.getgid()
        self.docker_repository = docker_repository

    def build(self, dockerfile: str, path: str, tag: str):
        try:
            self.client.images.build(dockerfile=dockerfile, path=path, tag=tag)
        except docker.errors.APIError as e:
            logging.error(f"Docker API error: {str(e)}")
            raise

    def run(
        self,
        image: str,
        command: str,
        working_dir: str,
        mounts: list[VolumeMount] | None = None,
        detach: bool = True,
        remove: bool = True,
        tty: bool = False,
    ) -> Container:
        try:
            volumes = {}
            if mounts:
                for mount in mounts:
                    volumes[mount.source] = {"bind": mount.target, "mode": mount.mode}
            logging.debug(f"Starting container from image '{image}'")

            image = f"{self.docker_repository}:{image}"

            envs = {
                "USER_ID": str(self.uid),
                "GROUP_ID": str(self.gid),
            }

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
