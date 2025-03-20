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
    def __init__(self):
        self.client = docker.from_env()
        self.uid = os.getuid()
        self.gid = os.getgid()

    def run(
        self,
        image: str,
        command: str,
        mounts: list[VolumeMount] | None = None,
        remove: bool = True,
    ) -> Container:
        try:
            volumes = {}
            if mounts:
                for mount in mounts:
                    volumes[mount.source] = {"bind": mount.target, "mode": mount.mode}
            logging.debug(f"Starting container from image '{image}'")

            container = self.client.containers.run(
                image=image,
                command=command,
                user=f"{self.uid}:{self.gid}",
                volumes=volumes if volumes else None,
                detach=True,
                remove=remove,
            )

            logging.info(f"Container started with ID: {container.id}")
            return container

        except docker.errors.ImageNotFound:
            logging.error(f"Image not found: {image}")
            raise
        except docker.errors.APIError as e:
            logging.error(f"Docker API error: {str(e)}")
            raise
