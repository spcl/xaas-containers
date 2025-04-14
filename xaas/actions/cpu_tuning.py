from __future__ import annotations

import logging
import os
import re
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
from xaas.actions.build import Config as BuildConfig, CPUTuningFeatures
from xaas.actions.analyze import (
    CompileCommand,
    Config as AnalyzerConfig,
    DivergenceReason,
    SourceFileStatus,
    ProjectDivergence,
)
from xaas.actions.preprocess import PreprocessingResult, ProcessedResults
from xaas.actions.docker import VolumeMount

from concurrent.futures import ThreadPoolExecutor, as_completed, process


class CPUTuning(Action):
    """
    When using clang, we can detect flags like `-m<arch>` to detect if the code is vectorized.
    Then, we can skip this in the following way:
    - Build Clang -> IR without any LLVM optimizations. We remove optimization flag.
    - At target system, use LLVM's opt to run optimization flags and readd the target features.
    - Only then we perform lowering.

    We need to disable optimizations since otherwise the loop vectorization would not be reapplied.
    I couldn't find a way to disable vectorization at the first step, and then redo it with opt.
    The result was different, and the assembly would not use the best possible vector instructions.

    In addition, we need to detect feature flags. For example, if we provide `-mavx512f`, Clang will
    add many different features like `avx` and `avx2`. Opt will not do that.
    Furthermore, LLVM no longer allows us to override target-cpu.

    Thus, we do the following: we extract IR attributes for each platform, skip optimizations,
    and then override them at deployment. Afterward, we can do the lowering.
    """

    def __init__(self):
        super().__init__(name="clang-vectorizer", description="Detect vectorization flags .")
        self.CLANG_PATH = "/usr/bin/c++"

        self.SUPPORTED_FLAGS: set[str] = {"mavx", "mfma"}

    # def print_summary(self, config: PreprocessingResult) -> None:
    #    logging.info(f"Total files: {len(config.targets)}")

    #    differences = defaultdict(set)
    #    different_files = set()
    #    difference_identical_hash = defaultdict(set)
    #    identical_hash_different_flags = set()
    #    identical_after_processing = set()
    #    for target, status in config.targets.items():
    #        hash_val = status.projects[status.baseline_project].hash

    #        for project_name, project_status in status.projects.items():
    #            if project_name == status.baseline_project:
    #                continue

    #            if hash_val == project_status.hash and (
    #                len(
    #                    set(project_status.cmd_differences.reasons.keys())
    #                    - {DivergenceReason.DEFINITIONS, DivergenceReason.INCLUDES}
    #                )
    #                == 0
    #            ):
    #                identical_after_processing.add(target)
    #            elif hash_val == project_status.hash:
    #                for key in project_status.cmd_differences.reasons:
    #                    difference_identical_hash[key].add(target)
    #                    identical_hash_different_flags.add(target)
    #            else:
    #                for key in project_status.cmd_differences.reasons:
    #                    differences[key].add(target)
    #                    different_files.add(target)

    #    logging.info(f"Different files: {len(different_files)}")
    #    for key, value in differences.items():
    #        logging.info(f"\tDifference: {key}, count: {len(value)}")
    #    logging.info(
    #        f"Identical to baseline after preprocessing: {len(identical_after_processing)}"
    #    )
    #    logging.info(
    #        f"Identical to baseline after preprocessing, different flags: {len(identical_hash_different_flags)}"
    #    )
    #    for key, value in difference_identical_hash.items():
    #        logging.info(f"\tDifference: {key}, count: {len(value)}")

    def execute(self, config: PreprocessingResult) -> bool:
        logging.info(f"[{self.name}] Preprocessing project {config.build.project_name}")

        Container = namedtuple("Container", ["container", "working_dir"])  # noqa: F821
        containers = {}

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
                        image=config.build.docker_image_dev,
                        mounts=volumes,
                        remove=True,
                        detach=True,
                        tty=True,
                        working_dir=target,
                    ),
                    target,
                )

            for target, status in list(config.targets.items()):
                baseline_flags: set[str] = status.baseline_command.cpu_tuning

                cmd = status.baseline_command
                # baseline_projectbaseline_projectprojects[status.baseline_project].files[target]
                self.get_flags(
                    config,
                    containers[status.baseline_project].container,
                    baseline_flags,
                    target,
                    cmd,
                    containers[status.baseline_project].working_dir,
                )

                # FIXME: dirty and simpliistic implementation
                # in the long run, we need to detect groups like in the OMP impl
                # merge with the OMP implementation - similar logic
                files_divergent = 0
                for project_name, project_status in status.projects.items():
                    if DivergenceReason.CPU_TUNING in project_status.cmd_differences.reasons:
                        for (
                            nested_project_name,
                            nested_project_status,
                        ) in status.projects.items():
                            if project_name == nested_project_name:
                                continue

                            if project_status.hash == nested_project_status.hash:
                                files_divergent += 1

                # FIXME: this is a hck to not process files with differnet hash
                # needs proper invesigation
                if files_divergent == 0:
                    continue

                for project_name, project_status in status.projects.items():
                    if DivergenceReason.COMPILER in project_status.cmd_differences.reasons:
                        continue

                    if DivergenceReason.CPU_TUNING in project_status.cmd_differences.reasons:
                        new_flags = set(
                            project_status.cmd_differences.reasons[DivergenceReason.CPU_TUNING][
                                "added"
                            ]
                        )
                        new_flags = new_flags | (
                            baseline_flags
                            - set(
                                project_status.cmd_differences.reasons[DivergenceReason.CPU_TUNING][
                                    "removed"
                                ]
                            )
                        )
                        self.get_flags(
                            config,
                            containers[status.baseline_project].container,
                            baseline_flags,
                            target,
                            cmd,
                            containers[status.baseline_project].working_dir,
                        )
                        divergent_cmd = project_status.command
                        self.get_flags(
                            config,
                            containers[project_name].container,
                            new_flags,
                            target,
                            divergent_cmd,
                            containers[project_name].working_dir,
                        )
                        status.projects[status.baseline_project].cpu_tuning = baseline_flags
                        project_status.cpu_tuning = new_flags

                        logging.debug(f"[{self.name}] Simplify file {target}")
                        del project_status.cmd_differences.reasons[DivergenceReason.CPU_TUNING]

                        # sanity check
                        for flags in [baseline_flags, new_flags]:
                            for elem in flags:
                                if (
                                    not re.match(r"-m(?:tune=|arch=|)(\w+)", elem)
                                    and elem not in self.SUPPORTED_FLAGS
                                ):
                                    raise RuntimeError(f"Unknown flag: {elem} in {flags}!")

        finally:
            for container in containers.values():
                container.container.stop(timeout=0)

        config_path = os.path.join(config.build.working_directory, "cpu_tuning.yml")
        config.save(config_path)

    def get_flags(
        self,
        config: PreprocessingResult,
        container: Container,
        flags: set[str],
        target: str,
        command: CompileCommand,
        working_dir: str,
    ) -> CPUTuningFeatures:
        for k, v in config.build.target_flags:
            if k == flags:
                return v

        features = CPUTuningFeatures()

        preprocess_cmd = [self.CLANG_PATH]

        preprocess_cmd.extend(command.includes)
        preprocess_cmd.extend(command.definitions)
        preprocess_cmd.extend(command.flags)
        preprocess_cmd.append(command.source)
        preprocess_cmd.extend(command.others)

        preprocess_cmd.extend(flags)

        ir_file = str(Path(target).with_suffix(".ll"))

        preprocess_cmd.extend(["-S", "-emit-llvm", "-o", ir_file])

        cmd = ["/bin/bash", "-c", " ".join(preprocess_cmd)]

        code, output = self.docker_runner.exec_run(container, cmd, working_dir)

        if code != 0:
            raise RuntimeError(f"Error preprocessing {target}: {output}")

        # FIXME: make into nice config
        get_features_cmd = [
            "/opt/llvm/bin/opt",
            "-load-pass-plugin",
            "/tools/build/libReplaceTargetFeatures.so",
            '-passes="replace-target-features"',
            "-query-features=true",
            ir_file,
            "-o",
            "/dev/null",
        ]
        cmd = ["/bin/bash", "-c", " ".join(get_features_cmd)]
        code, output = self.docker_runner.exec_run(container, cmd, working_dir)

        if code != 0:
            raise RuntimeError(f"Error extracting features! {target}: {output}")

        individual_pattern = r'"([^"]+)"="([^"]+)"'

        matches = re.findall(individual_pattern, output.decode())
        for param, value in matches:
            if param == "target-features":
                features.target_features = value
            elif param == "target-cpu":
                features.target_cpu = value
            elif param == "tune-cpu":
                features.tune_cpu = value
            else:
                raise RuntimeError("Unknown parameter: {param}")

        config.build.target_flags.append((flags, features))

        return features

    def validate(self, build_config: AnalyzerConfig) -> bool:
        work_dir = build_config.build.working_directory
        if not os.path.exists(work_dir):
            logging.error(f"[{self.name}] Working directory does not exist: {work_dir}")
            return False

        if len(build_config.build.build_results) == 0:
            logging.error(f"[{self.name}] No builds present!")
            return False

        return True
