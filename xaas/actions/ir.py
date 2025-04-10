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
        self.parallel_workers = parallel_workers
        self.CLANG_PATH = "/usr/bin/clang++-19"
        self.IR_PATH = "irs"

    # def print_summary(self, config: AnalyzerConfig) -> None:
    #    logging.info(f"Total files processed: {len(config.build_comparison.source_files)}")

    #    for build_name, build in config.build_comparison.project_results.items():
    #        logging.info(f"Project {build_name}:")
    #        logging.info(f"\tFiles {len(build.files)} files")

    #    ir_successes = 0
    #    ir_failures = 0
    #    divergent_ir_successes = 0
    #    divergent_ir_failures = 0

    #    for src, status in config.build_comparison.source_files.items():
    #        if hasattr(status, "ir_generated") and status.ir_generated:
    #            ir_successes += 1
    #        else:
    #            ir_failures += 1

    #        for _, project_status in status.divergent_projects.items():
    #            if hasattr(project_status, "ir_generated") and project_status.ir_generated:
    #                divergent_ir_successes += 1
    #            else:
    #                divergent_ir_failures += 1

    #    logging.info(f"IR Generation Summary:")
    #    logging.info(f"\tBase files successfully compiled: {ir_successes}")
    #    logging.info(f"\tBase files failed to compile: {ir_failures}")
    #    logging.info(f"\tDivergent files successfully compiled: {divergent_ir_successes}")
    #    logging.info(f"\tDivergent files failed to compile: {divergent_ir_failures}")

    def execute(self, config: PreprocessingResult) -> bool:
        logging.info("[{self.name}] Generating LLVM IR")

        containers = {}
        compile_dbs = {}

        ir_dir = os.path.realpath(os.path.join(config.build.working_directory, self.IR_PATH))
        os.makedirs(ir_dir, exist_ok=True)

        for build in config.build.build_results:
            if os.path.basename(build.directory) not in self.build_projects:
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

            compile_dbs[build.directory] = {entry["file"]: entry for entry in data}

        total_tasks = 0
        for src, status in config.source_files.items():
            total_tasks += 1
            total_tasks += len(status.projects)

        futures = []
        results = []

        with tqdm.tqdm(total=total_tasks) as pbar:  # noqa: SIM117
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                for src, status in config.source_files.items():
                    baseline_project = status.baseline_project

                    if os.path.basename(baseline_project) in self.build_projects:
                        logging.info(f"[{self.name}] Build file {src} for {baseline_project}")
                        cmake_cmd = compile_dbs[baseline_project][src]["command"]
                        cmake_directory = compile_dbs[baseline_project][src]["directory"]
                        ir_path, is_new = self._find_id(
                            status.projects[status.baseline_project],
                            status.baseline_command,
                            status,
                        )
                        assert is_new

                        ir_target = os.path.join("/irs", status.baseline_command.target, ir_path)
                        futures.append(
                            executor.submit(
                                self._compile_ir,
                                containers[baseline_project],
                                src,
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
                        if os.path.basename(project_name) not in self.build_projects:
                            continue

                        ir_path, is_new = self._find_id(
                            project_status,
                            status.baseline_command,
                            status,
                        )
                        ir_target = os.path.join("/irs", status.baseline_command.target, ir_path)
                        if not is_new:
                            logging.info(
                                f"[{self.name}] Skip building shared IR file {src} for {project_name}"
                            )
                            fut = Future()
                            fut.set_result(ir_target)
                            futures.append(fut)
                            continue

                        cmake_cmd = compile_dbs[project_name][src]["command"]
                        cmake_directory = compile_dbs[project_name][src]["directory"]

                        futures.append(
                            executor.submit(
                                self._compile_ir,
                                containers[project_name],
                                src,
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
        for src, status in config.source_files.items():
            if os.path.basename(status.baseline_project) in self.build_projects:
                baseline_result = next(result_iter)
                if baseline_result:
                    status.projects[status.baseline_project].ir_file.file = baseline_result
                else:
                    logging.error(f"Failed to build IR {src} for baseline project")
                    errors += 1

            for project_name, _ in status.projects.items():
                if project_name == status.baseline_project:
                    continue
                if os.path.basename(project_name) not in self.build_projects:
                    continue
                divergent_result = next(result_iter)
                if divergent_result:
                    status.projects[project_name].ir_file.file = divergent_result
                else:
                    logging.error(f"Failed to build IR {src} for other project")
                    errors += 1

        if errors > 0:
            logging.error(f"Failed to build {errors} IR files")

        # for container in containers.values():
        #    container.stop(timeout=0)

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
        status: FileStatus,
        baseline_command: CompileCommand,
        process_results: ProcessedResults,
    ) -> tuple[str, bool]:
        file_hash = status.hash
        assert file_hash is not None

        current_divergence = status.cmd_differences
        if file_hash not in process_results.ir_files:
            path = self._generate_ir_path(status, baseline_command, 0)

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

            path = self._generate_ir_path(status, baseline_command, len(existing_statuses))
            file = copy.deepcopy(status.ir_file)
            file.file = path
            process_results.ir_files[file_hash].append((current_divergence, file))
            return (path, True)

    def _generate_ir_path(
        self,
        status: FileStatus,
        baseline_command: CompileCommand,
        id: int,
    ) -> str:
        # location?
        # for first file, we put in /irs/<cmake-target>/hash/0/
        # what if the file is still different because of flags?
        # for every other, we put in /irs/<cmake-target>/hash/<id>

        ir_file = str(os.path.basename(Path(baseline_command.target).with_suffix(".bc")))

        assert status.hash is not None
        ir_path = os.path.join(status.hash, str(id), ir_file)

        return ir_path

    def _compile_ir(
        self,
        container: Container,
        source_file: str,
        baseline_command: CompileCommand,
        cmake_cmd: str,
        cmake_directory: str,
        ir_path: str,
        ir_target: str,
        working_directory: str,
    ) -> str | None:
        local_ir_target = os.path.join(
            working_directory, self.IR_PATH, baseline_command.target, ir_path
        )
        os.makedirs(os.path.dirname(local_ir_target), exist_ok=True)

        # The paths can be relative:
        # directory: /build/a/b
        # target: a/b/x/c.cpp
        # actual file in the command
        actual_target = os.path.relpath(
            os.path.join("/build", baseline_command.target), cmake_directory
        )

        ir_cmd = cmake_cmd.replace(actual_target, ir_target)
        ir_cmd = f"{ir_cmd} -emit-llvm"

        logging.info(f"IR Compilation of {source_file}, {baseline_command.target} -> {ir_target}")
        # if we just pass the raw comamnd
        code, output = self.docker_runner.exec_run(container, ["/bin/bash", "-c", ir_cmd], "/build")

        if code != 0:
            logging.error(f"Error generating IR for {source_file}: {output}")
            return None

        logging.debug(f"Successfully generated IR for {source_file}")
        return ir_target

    def validate(self, build_config: PreprocessingResult) -> bool:
        work_dir = build_config.build.working_directory
        if not os.path.exists(work_dir):
            logging.error(f"[{self.name}] Working directory does not exist: {work_dir}")
            return False

        return True
