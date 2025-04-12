import copy
import logging
import os
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

import tqdm
from docker.models.containers import RUN_CREATE_KWARGS, Container

from xaas.actions.action import Action
from xaas.actions.analyze import CompileCommand
from xaas.actions.preprocess import (
    ProcessedResults,
    ProjectDivergence,
    DivergenceReason,
    contains_openmp_flag,
)
from xaas.actions.preprocess import PreprocessingResult, IRFileStatus, FileStatus
from xaas.actions.docker import VolumeMount


def is_vectoriation_flag(flag: str) -> bool:
    """
    This implementation only supports Clang.
    """
    # gcc: -fopenmp
    # clang: -fopenmp=libomp
    # intel (icx/icpc): -qopenmp
    # intel (icx/icpx): -fopenmp
    # FIXME: add more for nvidia hpc of Cray if we need to
    return any("-fopenmp" in flag or "-qopenmp" in flag for flag in flags)


class IRCompiler(Action):
    def __init__(self, parallel_workers: int, build_projects: list[str]):
        super().__init__(
            name="ircompiler", description="Generate LLVM IR bitcode for source files."
        )

        self.build_projects = build_projects
        self.conditional_build = len(self.build_projects) > 0
        self.parallel_workers = parallel_workers
        self.CLANG_PATH = "/usr/bin/clang++-19"
        self.IR_PATH = "irs"

    def print_summary(self, config: PreprocessingResult) -> None:
        if not config or not config.targets:
            logging.warning("Configuration data is empty or has no targets.")
            return

        """
            - All configurations, and for each one of them how many total targets are needed
            - Number of IR files shared across all configurations
            - Number of IR files in the baseline configuration
            - Number of IR files used only by other configurations (not the baseline)
        """

        all_configurations: set[str] = set()
        targets_per_config = defaultdict(int)
        hashes_per_config = defaultdict(set)
        new_hashes_per_config = defaultdict(set)
        baseline_hashes: set[str] = set()
        non_baseline_hashes: set[str] = set()
        all_hashes: set[str] = set()

        shared_by_everyone = set()

        for target_name, target_info in config.targets.items():
            baseline_project_name = target_info.baseline_project
            if not baseline_project_name:
                logging.warning(f"Target '{target_name}' is missing a baseline_project.")
                continue

            if baseline_project_name in target_info.projects:
                baseline_status = target_info.projects[baseline_project_name]
                all_configurations.add(baseline_project_name)
                # targets_per_config[baseline_project_name] += 1
                assert baseline_status.hash
                baseline_hashes.add(baseline_status.hash)
                hashes_per_config[baseline_project_name].add(baseline_status.hash)
                new_hashes_per_config[baseline_project_name].add(baseline_status.hash)
                all_hashes.add(baseline_status.hash)
            else:
                logging.warning(
                    f"Baseline project '{baseline_project_name}' not found in projects for target '{target_name}'."
                )

            for project_name, project_status in target_info.projects.items():
                all_configurations.add(project_name)
                hashes_per_config[project_name].add(project_status.hash)
                targets_per_config[project_name] += 1

                if project_name != baseline_project_name:
                    assert project_status.hash
                    non_baseline_hashes.add(project_status.hash)

                    all_hashes.add(project_status.hash)
                    if project_status.hash not in baseline_hashes:
                        new_hashes_per_config[project_name].add(project_status.hash)

        non_baseline_only_hashes = non_baseline_hashes - baseline_hashes

        logging.info("IR Generation Summary")

        logging.info(f"\tTotal unique IRs {len(all_hashes)}")

        logging.info("\tConfigurations and Target Counts:")
        if all_configurations:
            sorted_configs = sorted(all_configurations)
            for config_name in sorted_configs:
                # Adjusting count: A target is counted for *each* config it's part of.
                # The user asked "how many total targets are *needed*" for each config.
                # Interpretation: How many target entries list this configuration?
                logging.info(f"- {config_name}: {targets_per_config[config_name]} targets")
        else:
            logging.info("No configurations found.")

        logging.info("\tUnique Hashes per Configuration:")
        if hashes_per_config:
            # Sort for consistent output
            sorted_configs = sorted(hashes_per_config.keys())
            for config_name in sorted_configs:
                logging.info(
                    f"- {config_name}: {len(new_hashes_per_config[config_name])} unique hashes"
                )
        else:
            logging.info("No hash data found.")

        logging.info(f"\tUnique Hashes in Baseline Configurations: {len(baseline_hashes)}")

        logging.info(
            f"\tUnique Hashes Used Only by Non-Baseline Configurations: {len(non_baseline_only_hashes)}"
        )

    def execute(self, config: PreprocessingResult) -> bool:
        logging.info("[{self.name}] Generating LLVM IR")

        containers = {}
        compile_dbs = {}

        ir_dir = os.path.realpath(os.path.join(config.build.working_directory, self.IR_PATH))
        os.makedirs(ir_dir, exist_ok=True)

        for build in config.build.build_results:
            if (
                self.conditional_build
                and os.path.basename(build.directory) not in self.build_projects
            ):
                continue

            logging.info(f"Setting up container for build {build}")

            volumes = []
            volumes.append(
                VolumeMount(
                    source=os.path.realpath(config.build.source_directory), target="/source"
                )
            )
            volumes.append(VolumeMount(source=os.path.realpath(build.directory), target="/build"))
            volumes.append(
                VolumeMount(
                    source=ir_dir,
                    target="/irs",
                )
            )

            containers[build.directory] = self.docker_runner.run(
                command="/bin/bash",
                image=config.build.docker_image,
                mounts=volumes,
                remove=False,
                detach=True,
                tty=True,
                working_dir="/build",
            )

            project_file = os.path.join(build.directory, "compile_commands.json")
            try:
                with open(project_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                raise RuntimeError(f"Error reading {project_file}: {e}") from e

            compile_dbs[build.directory] = {entry["output"]: entry for entry in data}

        total_tasks = 0
        for _, status in config.targets.items():
            total_tasks += 1
            total_tasks += len(status.projects)

        futures = []
        results = []

        with tqdm.tqdm(total=total_tasks) as pbar:  # noqa: SIM117
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                for target, status in config.targets.items():
                    baseline_project = status.baseline_project

                    if (
                        not self.conditional_build
                        or os.path.basename(baseline_project) in self.build_projects
                    ):
                        logging.info(f"[{self.name}] Build file {target} for {baseline_project}")
                        cmake_cmd = compile_dbs[baseline_project][target]["command"]
                        cmake_directory = compile_dbs[baseline_project][target]["directory"]
                        ir_path, is_new = self._find_id(
                            target,
                            status.projects[status.baseline_project],
                            status,
                        )
                        assert is_new

                        ir_target = os.path.join("/irs", target, ir_path)
                        futures.append(
                            executor.submit(
                                self._compile_ir,
                                containers[baseline_project],
                                target,
                                status.baseline_command,
                                cmake_cmd,
                                cmake_directory,
                                ir_path,
                                ir_target,
                                config.build.working_directory,
                            )
                        )

                    for project_name, project_status in status.projects.items():
                        if project_name == baseline_project:
                            continue
                        if (
                            self.conditional_build
                            and os.path.basename(project_name) not in self.build_projects
                        ):
                            continue

                        ir_path, is_new = self._find_id(
                            target,
                            project_status,
                            status,
                        )
                        ir_target = os.path.join("/irs", target, ir_path)
                        if not is_new:
                            logging.info(
                                f"[{self.name}] Skip building shared IR file {target} for {project_name}"
                            )
                            fut = Future()
                            fut.set_result(ir_target)
                            futures.append(fut)
                            continue

                        cmake_cmd = compile_dbs[project_name][target]["command"]
                        cmake_directory = compile_dbs[project_name][target]["directory"]

                        futures.append(
                            executor.submit(
                                self._compile_ir,
                                containers[project_name],
                                target,
                                status.baseline_command,
                                cmake_cmd,
                                cmake_directory,
                                ir_path,
                                ir_target,
                                config.build.working_directory,
                            )
                        )

                for future in futures:
                    results.append(future.result())
                    pbar.update(1)

        errors = 0
        result_iter = iter(results)
        for target, status in config.targets.items():
            if (
                not self.conditional_build
                or os.path.basename(status.baseline_project) in self.build_projects
            ):
                baseline_result = next(result_iter)
                if baseline_result:
                    status.projects[status.baseline_project].ir_file.file = baseline_result
                else:
                    logging.error(f"Failed to build IR {target} for baseline project")
                    errors += 1

            for project_name, _ in status.projects.items():
                if project_name == status.baseline_project:
                    continue
                if (
                    self.conditional_build
                    and os.path.basename(project_name) not in self.build_projects
                ):
                    continue
                divergent_result = next(result_iter)
                if divergent_result:
                    status.projects[project_name].ir_file.file = divergent_result
                else:
                    logging.error(f"Failed to build IR {target} for other project")
                    errors += 1

        if errors > 0:
            logging.error(f"Failed to build {errors} IR files")

        for container in containers.values():
            container.stop(timeout=0)

        config_path = os.path.join(config.build.working_directory, "ir_compilation.yml")
        config.save(config_path)

        # Print summary
        # self.print_summary(config)

        return True

    @staticmethod
    def _divergence_equal(old: tuple[ProjectDivergence, IRFileStatus], new: FileStatus) -> bool:
        for reason in [
            DivergenceReason.COMPILER,
            DivergenceReason.OPTIMIZATIONS,
            DivergenceReason.OTHERS,
        ]:
            if reason in old[0].reasons and reason in new.cmd_differences.reasons:
                if old[0].reasons[reason] != new.cmd_differences.reasons[reason]:
                    return False
            elif reason in old[0].reasons or reason in new.cmd_differences.reasons:
                return False

        """
            Apply the OpenMP optimization. Only possible if the only flag difference between builds.
        """
        reason = DivergenceReason.FLAGS
        old_library_flags = (
            old[0].reasons[DivergenceReason.FLAGS] if reason in old[0].reasons else {}
        )
        new_library_flags = (
            new.cmd_differences.reasons[DivergenceReason.FLAGS]
            if reason in new.cmd_differences.reasons
            else {}
        )
        if old_library_flags != new_library_flags:
            differences = set()
            for flag in [old_library_flags, new_library_flags]:
                if "added" in flag:
                    for change in flag["added"]:
                        differences.add(change)
                if "removed" in flag:
                    for change in flag["removed"]:
                        differences.add(change)

            if len(differences) == 1 and contains_openmp_flag(differences):
                return not new.ir_file.has_omp and not old[1].has_omp
            else:
                return False

        return True

    def _find_id(
        self,
        target: str,
        status: FileStatus,
        process_results: ProcessedResults,
    ) -> tuple[str, bool]:
        file_hash = status.hash
        assert file_hash is not None

        current_divergence = status.cmd_differences
        if file_hash not in process_results.ir_files:
            path = self._generate_ir_path(status, target, 0)

            file = copy.deepcopy(status.ir_file)
            file.file = path
            process_results.ir_files[file_hash] = [(current_divergence, file)]
            return (path, True)
        else:
            # This hash exists, check existing objects for it.
            existing_statuses = process_results.ir_files[file_hash]
            for existing_hash in existing_statuses:
                # existing_divergence = existing_hash[0]

                if self._divergence_equal(existing_hash, status):
                    return (existing_hash[1].file, False)

            path = self._generate_ir_path(status, target, len(existing_statuses))
            file = copy.deepcopy(status.ir_file)
            file.file = path
            process_results.ir_files[file_hash].append((current_divergence, file))
            return (path, True)

    def _generate_ir_path(
        self,
        status: FileStatus,
        target: str,
        id: int,
    ) -> str:
        # location?
        # for first file, we put in /irs/<cmake-target>/hash/0/
        # what if the file is still different because of flags?
        # for every other, we put in /irs/<cmake-target>/hash/<id>

        ir_file = str(os.path.basename(Path(target).with_suffix(".bc")))

        assert status.hash is not None
        ir_path = os.path.join(status.hash, str(id), ir_file)

        return ir_path

    def _compile_ir(
        self,
        container: Container,
        target: str,
        baseline_command: CompileCommand,
        cmake_cmd: str,
        cmake_directory: str,
        ir_path: str,
        ir_target: str,
        working_directory: str,
    ) -> str | None:
        local_ir_target = os.path.join(working_directory, self.IR_PATH, target, ir_path)
        os.makedirs(os.path.dirname(local_ir_target), exist_ok=True)

        # The paths can be relative:
        # directory: /build/a/b
        # target: a/b/x/c.cpp
        # actual file in the command
        actual_target = os.path.relpath(os.path.join("/build", target), cmake_directory)

        ir_cmd = cmake_cmd.replace(actual_target, ir_target)
        ir_cmd = f"{ir_cmd} -emit-llvm"

        logging.info(f"IR Compilation of {baseline_command.source}, {target} -> {ir_target}")
        # if we just pass the raw comamnd
        code, output = self.docker_runner.exec_run(container, ["/bin/bash", "-c", ir_cmd], "/build")

        if code != 0:
            logging.error(f"Error generating IR for {baseline_command.source}: {output}")
            return None

        logging.debug(f"Successfully generated IR for {baseline_command.source}")
        return ir_target

    def validate(self, build_config: PreprocessingResult) -> bool:
        work_dir = build_config.build.working_directory
        if not os.path.exists(work_dir):
            logging.error(f"[{self.name}] Working directory does not exist: {work_dir}")
            return False

        return True
