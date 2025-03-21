import logging
import os
from collections import defaultdict
from hashlib import md5
from mmap import ACCESS_READ
from mmap import mmap
from pathlib import Path

import tqdm
from docker.models.containers import Container

from xaas.actions.action import Action
from xaas.actions.analyze import CompileCommand
from xaas.actions.analyze import Config as AnalyzerConfig
from xaas.actions.analyze import DivergenceReason
from xaas.actions.analyze import SourceFileStatus
from xaas.actions.docker import VolumeMount


class ClangPreprocesser(Action):
    def __init__(self):
        super().__init__(
            name="clangpreproceser", description="Apply Clang preprocessing to detect differences."
        )

        self.DOCKER_IMAGE = "builder"
        self.CLANG_PATH = "/usr/bin/clang++-16"

    def print_summary(self, config: AnalyzerConfig) -> None:
        logging.info(f"Total files: {len(config.build_comparison.source_files)}")

        for build_name, build in config.build_comparison.project_results.items():
            logging.info(f"Project {build_name}:")
            logging.info(f"\tFiles {len(build.files)} files")

        differences = defaultdict(set)
        different_files = set()
        difference_identical_hash = defaultdict(set)
        identical_hash_different_flags = set()
        identical_after_processing = set()
        for src, status in config.build_comparison.source_files.items():
            hash_val = status.hash

            for _, project_status in status.divergent_projects.items():
                if hash_val == project_status.hash and (
                    len(
                        set(project_status.reasons.keys())
                        - {DivergenceReason.DEFINITIONS, DivergenceReason.INCLUDES}
                    )
                    == 0
                ):
                    identical_after_processing.add(src)
                elif hash_val == project_status.hash:
                    for key in project_status.reasons:
                        difference_identical_hash[key].add(src)
                        identical_hash_different_flags.add(src)
                else:
                    for key in project_status.reasons:
                        differences[key].add(src)
                        different_files.add(src)

        logging.info(f"Different files: {len(different_files)}")
        for key, value in differences.items():
            logging.info(f"\tDifference: {key}, count: {len(value)}")
        logging.info(
            f"Identical to baseline after preprocessing: {len(identical_after_processing)}"
        )
        logging.info(
            f"Identical to baseline after preprocessing, different flags: {len(identical_hash_different_flags)}"
        )
        for key, value in difference_identical_hash.items():
            logging.info(f"\tDifference: {key}, count: {len(value)}")

    def execute(self, config: AnalyzerConfig) -> bool:
        logging.info(f"[{self.name}] Preprocessing project {config.build.project_name}")

        containers = {}

        for build in config.build.build_results:
            logging.info(f"Analyzing build {build}")

            volumes = []
            volumes.append(
                VolumeMount(
                    source=os.path.realpath(config.build.source_directory), target="/source"
                )
            )
            volumes.append(VolumeMount(source=os.path.realpath(build.directory), target="/build"))

            containers[build.directory] = self.docker_runner.run(
                command="/bin/bash",
                image=self.DOCKER_IMAGE,
                mounts=volumes,
                remove=False,
                detach=True,
                tty=True,
                working_dir="/build",
            )

        baseline_project = config.build.build_results[0].directory

        for src, status in tqdm.tqdm(config.build_comparison.source_files.items()):
            if len(status.divergent_projects) == 0:
                logging.debug(f"Skipping {src}, no differences found")

            logging.debug(f"Preprocess baseline {src}")
            cmd = config.build_comparison.project_results[baseline_project].files[src]

            original_processed_file = self._preprocess_file(containers[baseline_project], src, cmd)
            if not original_processed_file:
                logging.error("Skip because of an error")
                continue

            success = []
            for name, _ in status.divergent_projects.items():
                logging.debug(f"Preprocess {src} for project {name}")

                cmd = config.build_comparison.project_results[name].files[src]

                processed_file = self._preprocess_file(containers[name], src, cmd)
                if processed_file:
                    success.append((name, processed_file))

            self._compare_preprocessed_files(
                src,
                (baseline_project, original_processed_file),
                success,
                config.build_comparison.source_files[src],
            )

        for container in containers.values():
            container.stop(timeout=0)

        config_path = os.path.join(config.build.working_directory, "preprocess.yml")
        config.save(config_path)

        self.print_summary(config)

    def _preprocess_file(
        self, container: Container, source_file: str, command: CompileCommand
    ) -> str | None:
        preprocess_cmd = [self.CLANG_PATH, "-E"]

        preprocess_cmd.extend(command.includes)
        preprocess_cmd.extend(command.definitions)

        preprocess_cmd.append(source_file)

        preprocessed_file = str(Path(command.target).with_suffix(".i"))

        preprocess_cmd.extend([">", preprocessed_file])

        # Docker will not allow us to run directly "cmd > output"
        # We need to redirect this as a shell command
        cmd = ["/bin/bash", "-c", " ".join(preprocess_cmd)]

        code, output = self.docker_runner.exec_run(container, cmd, "/build")

        if code != 0:
            logging.error(f"Error preprocessing {source_file}: {output}")
            return None
        else:
            return preprocessed_file

    def _hash_file(self, path: str) -> str:
        with open(path) as f:
            file_hash = md5()
            with mmap(f.fileno(), 0, access=ACCESS_READ) as m:
                file_hash.update(m)
        return file_hash.hexdigest()

    def _compare_preprocessed_files(
        self,
        src: str,
        original_processed_file: tuple[str, str],
        processed_files: list[tuple[str, str]],
        result: SourceFileStatus,
    ):
        logging.debug(f"Comparing preprocessed files for {src}")

        original_path = os.path.join(*original_processed_file)
        result.hash = self._hash_file(original_path)
        os.remove(original_path)

        for processed_file in processed_files:
            new_path = os.path.join(*processed_file)
            result.divergent_projects[processed_file[0]].hash = self._hash_file(new_path)
            os.remove(new_path)

    def validate(self, build_config: AnalyzerConfig) -> bool:
        work_dir = build_config.build.working_directory
        if not os.path.exists(work_dir):
            logging.error(f"[{self.name}] Working directory does not exist: {work_dir}")
            return False

        if len(build_config.build.build_results) == 0:
            logging.error(f"[{self.name}] No builds present!")
            return False

        return True
