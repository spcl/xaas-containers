import logging
import os
from collections import defaultdict, namedtuple
from hashlib import md5
from mmap import ACCESS_READ
from mmap import mmap
from pathlib import Path

import tqdm
from docker.models.containers import Container

from xaas.actions.action import Action
from xaas.actions.analyze import FileStatus
from xaas.actions.analyze import CompileCommand
from xaas.actions.analyze import Config as AnalyzerConfig
from xaas.actions.analyze import DivergenceReason
from xaas.actions.analyze import SourceFileStatus
from xaas.actions.docker import VolumeMount

from concurrent.futures import ThreadPoolExecutor, as_completed


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

        Container = namedtuple("Container", ["container", "working_dir"])  # noqa: F821
        containers = {}

        for build in config.build.build_results:
            logging.info(f"Analyzing build {build}")

            volumes = []
            volumes.append(
                VolumeMount(
                    source=os.path.realpath(config.build.source_directory), target="/source"
                )
            )

            build_dir = os.path.basename(build.directory)
            target = f"/builds/{build_dir}"
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

        baseline_project = config.build.build_results[0].directory

        futures = []
        results = []

        all_projects = list(config.build_comparison.source_files.items())

        size = 0
        for _, status in all_projects:
            size += 1 + len(status.divergent_projects)

        with tqdm.tqdm(total=size) as pbar:  # noqa: SIM117
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                for src, status in tqdm.tqdm(all_projects):
                    if len(status.divergent_projects) == 0:
                        continue

                    # FIXME: disable building of MPI files

                    cmd = config.build_comparison.project_results[baseline_project].files[src]
                    status.ir_file_status = FileStatus.INDIVIDUAL_IR
                    futures.append(
                        executor.submit(
                            self._preprocess_file,
                            containers[baseline_project].container,
                            src,
                            cmd,
                            containers[baseline_project].working_dir,
                        )
                    )

                    for name, div_status in status.divergent_projects.items():
                        cmd = config.build_comparison.project_results[name].files[src]

                        div_status.ir_file_status = FileStatus.INDIVIDUAL_IR
                        futures.append(
                            executor.submit(
                                self._preprocess_file,
                                containers[name].container,
                                src,
                                cmd,
                                containers[name].working_dir,
                            )
                        )

                for future in futures:
                    pbar.update(1)
                    results.append(future.result())

        result_iter = iter(results)
        for src, status in all_projects:
            if len(status.divergent_projects) == 0:
                logging.debug(f"Skipping {src}, no differences found")
                continue

            logging.debug(f"Preprocess baseline {src}")
            cmd = config.build_comparison.project_results[baseline_project].files[src]

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
                (baseline_project, *original_processed_file),
                success,
                config.build_comparison.source_files[src],
            )

            if self.openmp_check:
                self._optimize_omp(src, config.build_comparison.source_files[src])

        for container in containers.values():
            container.container.stop(timeout=0)

        config_path = os.path.join(config.build.working_directory, "preprocess.yml")
        config.save(config_path)

        self.print_summary(config)

    def _preprocess_file(
        self, container: Container, source_file: str, command: CompileCommand, working_dir: str
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
        result: SourceFileStatus,
    ):
        logging.debug(f"Comparing preprocessed files for {src}")

        original_path = os.path.join(*original_processed_file[0:2])
        result.hash = self._hash_file(original_path)
        result.has_omp = original_processed_file[2]
        os.remove(original_path)

        for processed_file in processed_files:
            new_path = os.path.join(*processed_file[0:2])

            result.divergent_projects[processed_file[0]].hash = self._hash_file(new_path)
            result.divergent_projects[processed_file[0]].has_omp = processed_file[2]
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
                divergent_project.ir_file_status = FileStatus.SHARED_IR

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
