from __future__ import annotations

import logging
import os
from collections import defaultdict, namedtuple
from itertools import islice
from enum import Enum
from hashlib import md5
from pathlib import Path
from mmap import ACCESS_READ, mmap
from dataclasses import dataclass, field

import tqdm
from docker.models.containers import Container
from mashumaro.mixins.yaml import DataClassYAMLMixin

from xaas.actions.action import Action
from xaas.actions.analyze import (
    CompileCommand,
    Config as AnalyzerConfig,
    DivergenceReason,
    SourceFileStatus,
    ProjectDivergence,
)
from xaas.actions.docker import VolumeMount

from concurrent.futures import ThreadPoolExecutor, as_completed, process


class IRStatus(Enum):
    SHARED_IR = "shared"
    INDIVIDUAL_IR = "individual"
    SOURCE = "source"

    def __str__(self):
        return self.name.lower()


@dataclass
class FileStatus(DataClassYAMLMixin):
    # command: CompileCommand
    cmd_differences: ProjectDivergence
    hash: str | None = None
    has_omp: bool = False
    ir_file: str | None = None
    ir_file_status: IRStatus = IRStatus.SOURCE


@dataclass
class ProcessedResults(DataClassYAMLMixin):
    baseline_project: str
    baseline_command: CompileCommand
    # Mapping: hash -> list of [(config, path)]
    # We use to decide when two file with the same hash are compatible with each other
    ir_files: dict[str, list[tuple[ProjectDivergence, str]]] = field(default_factory=dict)
    projects: dict[str, FileStatus] = field(default_factory=dict)


@dataclass
class PreprocessingResult(DataClassYAMLMixin):
    source_files: dict[str, ProcessedResults] = field(default_factory=dict)

    def save(self, config_path: str) -> None:
        with open(config_path, "w") as f:
            f.write(self.to_yaml())

    @staticmethod
    def load(config_path: str) -> PreprocessingResult:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Runtime configuration file not found: {config_path}")

        with open(config_path) as f:
            return PreprocessingResult.from_yaml(f)


def contains_openmp_flag(flags) -> bool:
    # gcc: -fopenmp
    # clang: -fopenmp=libomp
    # intel (icx/icpc): -qopenmp
    # intel (icx/icpx): -fopenmp
    # FIXME: add more for nvidia hpc of Cray if we need to
    return any("-fopenmp" in flag or "-qopenmp" in flag for flag in flags)


class ClangPreprocesser(Action):
    def __init__(self, parallel_workers: int, openmp_check: bool):
        super().__init__(
            name="clangpreproceser", description="Apply Clang preprocessing to detect differences."
        )

        self.parallel_workers = parallel_workers
        self.openmp_check = openmp_check
        self.DOCKER_IMAGE = "builder"
        self.CLANG_PATH = "/usr/bin/clang++-19"
        self.OMP_TOOL_PATH = "/tools/openmp-finder/omp-finder"

    def print_summary(self, config: PreprocessingResult) -> None:
        logging.info(f"Total files: {len(config.source_files)}")

        differences = defaultdict(set)
        different_files = set()
        difference_identical_hash = defaultdict(set)
        identical_hash_different_flags = set()
        identical_after_processing = set()
        for src, status in config.source_files.items():
            hash_val = status.projects[status.baseline_project].hash

            for project_name, project_status in status.projects.items():
                if project_name == status.baseline_project:
                    continue

                if hash_val == project_status.hash and (
                    len(
                        set(project_status.cmd_differences.reasons.keys())
                        - {DivergenceReason.DEFINITIONS, DivergenceReason.INCLUDES}
                    )
                    == 0
                ):
                    identical_after_processing.add(src)
                elif hash_val == project_status.hash:
                    for key in project_status.cmd_differences.reasons:
                        difference_identical_hash[key].add(src)
                        identical_hash_different_flags.add(src)
                else:
                    for key in project_status.cmd_differences.reasons:
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

        Container = namedtuple("Container", ["container", "working_dir"])  # noqa: F821
        containers = {}

        new_results = PreprocessingResult()

        try:
            for build in config.build.build_results:
                logging.info(f"Analyzing build {build}")

                volumes = []
                volumes.append(
                    VolumeMount(
                        source=os.path.realpath(config.build.source_directory), target="/source"
                    )
                )

                build_dir = os.path.basename(build.directory)
                target = "/build"
                volumes.append(VolumeMount(source=os.path.realpath(build.directory), target=target))

                containers[build.directory] = Container(
                    self.docker_runner.run(
                        command="/bin/bash",
                        image=self.DOCKER_IMAGE,
                        mounts=volumes,
                        remove=False,
                        detach=True,
                        tty=True,
                        working_dir=target,
                    ),
                    target,
                )

            all_projects = list(config.build_comparison.source_files.items())

            size = 0
            for _, status in all_projects:
                size += 1 + len(status.divergent_projects)

            # avoid problems caused by creating too many preprocessed files
            BATCH_SIZE = 128
            logging.info(f"We have total {size} files to process")
            with tqdm.tqdm(total=size) as pbar:  # noqa: SIM117
                with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                    iterator = iter(all_projects)
                    while slice_projects := list(islice(iterator, BATCH_SIZE)):
                        logging.info(f"Processing batch of {len(slice_projects)} files")
                        futures = []
                        results = []

                        for src, status in slice_projects:
                            # new_results.source_files[src] =
                            #
                            process_result = ProcessedResults(
                                baseline_project=status.default_build,
                                baseline_command=status.default_command,
                            )
                            process_result.projects[status.default_build] = FileStatus(
                                ProjectDivergence()
                            )

                            if len(status.divergent_projects) == 0:
                                continue

                            # FIXME: disable building of MPI files

                            cmd = config.build_comparison.project_results[
                                status.default_build
                            ].files[src]
                            futures.append(
                                executor.submit(
                                    self._preprocess_file,
                                    containers[status.default_build].container,
                                    src,
                                    status.default_command,
                                    cmd,
                                    containers[status.default_build].working_dir,
                                )
                            )

                            for name, div_status in status.divergent_projects.items():
                                cmd = config.build_comparison.project_results[name].files[src]

                                process_result.projects[name] = FileStatus(div_status)
                                # div_status.ir_file_status = FileStatus.INDIVIDUAL_IR
                                futures.append(
                                    executor.submit(
                                        self._preprocess_file,
                                        containers[name].container,
                                        src,
                                        status.default_command,
                                        cmd,
                                        containers[name].working_dir,
                                    )
                                )

                            new_results.source_files[src] = process_result

                        for future in futures:
                            pbar.update(1)
                            res = future.result()
                            results.append(res)

                        result_iter = iter(results)
                        for src, status in slice_projects:
                            if len(status.divergent_projects) == 0:
                                logging.debug(f"Skipping {src}, no differences found")
                                continue

                            logging.debug(f"Preprocess baseline {src}")
                            cmd = config.build_comparison.project_results[
                                status.default_build
                            ].files[src]

                            original_processed_file = next(result_iter)
                            if not original_processed_file:
                                logging.error("Skip because of an error")

                                # drop results
                                for _ in range(len(status.divergent_projects)):
                                    next(result_iter)

                                continue

                            success = []
                            for name, _ in status.divergent_projects.items():
                                logging.debug(f"Preprocess {src} for project {name}")

                                cmd = config.build_comparison.project_results[name].files[src]

                                processed_file = next(result_iter)
                                if processed_file:
                                    success.append((name, *processed_file))

                            self._compare_preprocessed_files(
                                src,
                                (status.default_build, *original_processed_file),
                                success,
                                new_results.source_files[src],
                            )

                # if self.openmp_check:
                #    self._optimize_omp(src, config.build_comparison.source_files[src])
        finally:
            for container in containers.values():
                container.container.stop(timeout=0)

        config_path = os.path.join(config.build.working_directory, "preprocess.yml")
        new_results.save(config_path)

        self.print_summary(new_results)

    def _preprocess_file(
        self,
        container: Container,
        source_file: str,
        baseline_command: CompileCommand,
        command: CompileCommand,
        working_dir: str,
    ) -> tuple[str, bool] | None:
        preprocess_cmd = [self.CLANG_PATH, "-E"]

        preprocess_cmd.extend(command.includes)
        preprocess_cmd.extend(command.definitions)

        preprocess_cmd.append(source_file)

        preprocessed_file = str(Path(command.target).with_suffix(".i"))

        preprocess_cmd.extend([">", preprocessed_file])

        # Docker will not allow us to run directly "cmd > output"
        # We need to redirect this as a shell command
        cmd = ["/bin/bash", "-c", " ".join(preprocess_cmd)]

        code, output = self.docker_runner.exec_run(container, cmd, working_dir)

        if code != 0:
            logging.error(f"Error preprocessing {source_file}: {output}")
            return None

        if not self.openmp_check:
            return preprocessed_file, True

        omp_tool_cmd = [self.OMP_TOOL_PATH, preprocessed_file]
        cmd = ["/bin/bash", "-c", " ".join(omp_tool_cmd)]
        code, output = self.docker_runner.exec_run(container, cmd, working_dir)
        if code != 0:
            logging.error(f"Error OMP processing {source_file}: {output}")
            return None

        return preprocessed_file, "XAAS_OMP_FOUND" in output.decode("utf-8")

    def _hash_file(self, path: str) -> str:
        with open(path) as f:
            file_hash = md5()
            with mmap(f.fileno(), 0, access=ACCESS_READ) as m:
                file_hash.update(m)
        return file_hash.hexdigest()

    def _compare_preprocessed_files(
        self,
        src: str,
        original_processed_file: tuple[str, str, bool],
        processed_files: list[tuple[str, str, bool]],
        result: ProcessedResults,
    ):
        logging.debug(f"Comparing preprocessed files for {src}")

        original_path = os.path.join(*original_processed_file[0:2])
        result.projects[original_processed_file[0]].hash = self._hash_file(original_path)
        result.projects[original_processed_file[0]].has_omp = original_processed_file[2]
        os.remove(original_path)

        for processed_file in processed_files:
            new_path = os.path.join(*processed_file[0:2])

            result.projects[processed_file[0]].hash = self._hash_file(new_path)
            result.projects[processed_file[0]].has_omp = processed_file[2]
            os.remove(new_path)

    def _optimize_omp(
        self,
        src: str,
        result: SourceFileStatus,
    ):
        to_delete = []
        for name, divergent_project in result.divergent_projects.items():
            # Optimization for OpenMP
            # If hash is the same, and there is no difference in compiler and 'others'
            # and the only difference in flags is `fopenmp`
            # and there is no openmp
            # then we can remove this divergence
            if result.hash != divergent_project.hash:
                continue

            if divergent_project.has_omp:
                continue

            found = False
            for reason in divergent_project.reasons:
                if reason in [
                    DivergenceReason.OPTIMIZATIONS,
                    DivergenceReason.OTHERS,
                    DivergenceReason.COMPILER,
                ]:
                    found = True
                    break

            if found:
                continue

            if DivergenceReason.FLAGS not in divergent_project.reasons:
                continue

            library_flags = divergent_project.reasons[DivergenceReason.FLAGS]
            if (
                len(library_flags["added"]) == 1
                and len(library_flags["removed"]) == 0
                and "fopenmp" in next(iter(library_flags["added"]))
            ):
                logging.info(
                    f"For source file {src}, the project {name} differs only in OpenMP flag - but not OpenMP present!"
                )
                # to_delete.append(name)
                # divergent_project.ir_file_status = FileStatus.SHARED_IR

        # for del_key in to_delete:
        #    print("Delete", src, del_key)
        #    del result.divergent_projects[del_key]

    def validate(self, build_config: AnalyzerConfig) -> bool:
        work_dir = build_config.build.working_directory
        if not os.path.exists(work_dir):
            logging.error(f"[{self.name}] Working directory does not exist: {work_dir}")
            return False

        if len(build_config.build.build_results) == 0:
            logging.error(f"[{self.name}] No builds present!")
            return False

        return True
