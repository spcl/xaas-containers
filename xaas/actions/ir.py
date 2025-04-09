import logging
import os
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
from docker.models.containers import RUN_CREATE_KWARGS, Container

from xaas.actions.action import Action
from xaas.actions.analyze import CompileCommand
from xaas.actions.preprocess import PreprocessingResult, IRStatus, FileStatus
from xaas.actions.docker import VolumeMount


class IRCompiler(Action):
    def __init__(self, parallel_workers: int):
        super().__init__(
            name="ircompiler", description="Generate LLVM IR bitcode for source files."
        )

        self.parallel_workers = parallel_workers
        self.DOCKER_IMAGE = "builder"
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

        # Start containers for each build
        for build in config.build.build_results:
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
                image=self.DOCKER_IMAGE,
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

        file_builds = {}
        with tqdm.tqdm(total=total_tasks) as pbar:  # noqa: SIM117
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                for src, status in config.source_files.items():
                    baseline_project = status.baseline_project
                    cmake_cmd = compile_dbs[baseline_project][src]["command"]

                    logging.info(f"[{self.name}] Build file {src} for {baseline_project}")

                    file_builds[src] = set()

                    cmd = status.baseline_command
                    futures.append(
                        executor.submit(
                            self._compile_ir,
                            containers[baseline_project],
                            src,
                            cmd,
                            cmake_cmd,
                            status.projects[status.baseline_project],
                            "",
                            config.build.working_directory,
                        )
                    )

                    for project_name, status in status.projects.items():
                        if project_name == baseline_project:
                            continue

                        cmake_cmd = compile_dbs[project_name][src]["command"]

                        if (
                            status.hash in file_builds
                            and status.ir_file_status == FileStatus.SHARED_IR
                        ):
                            logging.info(
                                f"[{self.name}] Skip building shared IR file {src} for {project_name}"
                            )
                            results.append(None)
                            continue

                        # file_builds.add(status.hash)

                        futures.append(
                            executor.submit(
                                self._compile_ir,
                                containers[project_name],
                                src,
                                cmd,
                                cmake_cmd,
                                status,
                                project_name,
                                config.build.working_directory,
                            )
                        )

                for future in as_completed(futures):
                    pbar.update(1)
                    results.append(future.result())

        result_iter = iter(results)
        for src, status in config.source_files.items():
            baseline_result = next(result_iter)
            if baseline_result:
                status.projects[status.baseline_project].ir_file = baseline_result

            for project_name, _ in status.projects.items():
                if project_name == status.baseline_project:
                    continue
                divergent_result = next(result_iter)
                if divergent_result:
                    status.projects[project_name].ir_file = divergent_result
                else:
                    status.projects[project_name].ir_file = divergent_result

        for container in containers.values():
            container.stop(timeout=0)

        config_path = os.path.join(config.build.working_directory, "ir_compilation.yml")
        config.save(config_path)

        # Print summary
        # self.print_summary(config)

        return True

    def _compile_ir(
        self,
        container: Container,
        source_file: str,
        baseline_command: CompileCommand,
        cmake_cmd: str,
        status: FileStatus,
        project_name: str,
        working_directory: str,
    ) -> str | None:
        # filename = os.path.basename(command.target)
        ir_file = str(os.path.basename(Path(baseline_command.target).with_suffix(".bc")))

        # location?
        # for shared file, we put in /irs/file/common/hash/
        # for individual, we put in /irs/file/<project-name>/hash/

        assert status.hash is not None

        if status.ir_file_status == IRStatus.SHARED_IR:
            ir_path = os.path.join("common", status.hash)
        else:
            ir_path = os.path.join(project_name, status.hash)
        ir_target = os.path.join("/irs", baseline_command.target, ir_path, ir_file)
        local_ir_target = os.path.join(
            working_directory, self.IR_PATH, baseline_command.target, ir_path
        )
        os.makedirs(local_ir_target, exist_ok=True)

        ir_cmd = cmake_cmd.replace(baseline_command.target, ir_target)
        ir_cmd = f"{ir_cmd} -emit-llvm"

        logging.info(f"IR Compilation of {source_file}, {baseline_command.target} -> {ir_target}")
        code, output = self.docker_runner.exec_run(container, ir_cmd.split(" "), "/build")

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
