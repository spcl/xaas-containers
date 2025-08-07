#!/usr/bin/env python3

import logging
import json
import os
from pathlib import Path

import click

from xaas.actions.analyze import BuildAnalyzer
from xaas.actions.analyze import Config as AnalyzerConfig
from xaas.actions.container import DockerImageBuilder
from xaas.source.container import SourceContainerDeployment, SourceContainerGenerator

from xaas.actions.ir import IRCompiler
from xaas.actions.build import BuildGenerator
from xaas.actions.build import Config as BuildConfig
from xaas.actions.cpu_tuning import CPUTuning
from xaas.actions.deployment import Deployment
from xaas.actions.preprocess import ClangPreprocesser, PreprocessingResult
from xaas.config import (
    DeployConfig,
    FeatureType,
    RunConfig,
    SourceContainerConfig,
    SourceDeploymentConfig,
    Language,
    XaaSConfig,
)
from xaas.docker import Runner as DockerRunner
import xaas.source.utils as utils


def initialize():
    config = XaaSConfig()
    config.initialize(XaaSConfig.DEFAULT_CONFIGURATION)

    logging.basicConfig(level=logging.INFO, force=True)


@click.group()
@click.version_option()
def cli() -> None:
    logging.info("XaaS Builder")


@cli.group()
def source() -> None:
    logging.info("XaaS Source Builder")


@source.command("container")
@click.argument("config", type=click.Path(exists=True))
def source_container(config) -> None:
    initialize()

    config_obj = SourceContainerConfig.load(config)
    action = SourceContainerGenerator(config_obj)
    action.generate()


@source.command("specializations")
@click.argument("app-name", type=str)
@click.argument("app-language", type=str)
@click.argument("system-discovery", type=click.Path(exists=True))
@click.option("--verbose", is_flag=True, default=False, help="Enable debug output")
def source_specialization(app_name, app_language, system_discovery, verbose) -> None:
    initialize()

    if verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)

    with open(system_discovery) as f:
        system_features = json.load(f)
    _, options, checker = SourceContainerDeployment.intersect_specializations(
        app_name, Language(app_language), system_features
    )
    utils.display_options(options, app_name, checker)


@source.command("deploy")
@click.argument("config", type=click.Path(exists=True))
@click.option("--parallel-workers", type=int, default=1, help="Parallel wokers")
@click.option("--build/--no-build", is_flag=True, default=True, help="Build image immediately")
@click.option("--verbose", is_flag=True, default=False, help="Enable debug output")
def source_deploy(config, parallel_workers, build: bool, verbose: bool) -> None:
    initialize()

    if verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)

    config_obj = SourceDeploymentConfig.load(config)
    action = SourceContainerDeployment(config_obj)
    action.generate(parallel_workers=parallel_workers, build=build)


@cli.group()
def ir() -> None:
    logging.info("XaaS IR Builder")


@ir.command()
@click.argument("config", type=click.Path(exists=True))
def buildgen(config) -> None:
    initialize()

    config_obj = RunConfig.load(config)
    action = BuildGenerator()
    action.validate(config_obj)
    action.execute(config_obj)


@ir.command()
@click.argument("config", type=click.Path(exists=True))
def analyze(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = BuildConfig.load(os.path.join(run_config.working_directory, "buildgen.yml"))
    action = BuildAnalyzer()
    action.validate(config_obj)
    action.execute(config_obj)


@ir.group()
def preprocess():
    pass


@preprocess.command("run")
@click.argument("config", type=click.Path(exists=True))
@click.option("--parallel-workers", type=int, default=1, help="Parallel wokers")
@click.option(
    "--no-openmp-check", "openmp_check", is_flag=True, default=True, help="Enable OpenMP support"
)
def preprocess_run(config, parallel_workers, openmp_check) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = AnalyzerConfig.load(
        os.path.join(run_config.working_directory, "build_analyze.yml")
        # os.path.join(run_config.working_directory, "cpu_tuning.yml")
    )
    action = ClangPreprocesser(parallel_workers, openmp_check)
    action.validate(config_obj)
    action.execute(config_obj)


@preprocess.command("summary")
@click.argument("config", type=click.Path(exists=True))
def preprocess_summary(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "preprocess.yml")
    )
    action = ClangPreprocesser(1, False)
    action.print_summary(config_obj)


@ir.group("cpu-tuning")
def cpu_tuning():
    pass


@cpu_tuning.command("run")
@click.argument("config", type=click.Path(exists=True))
def cpu_tuning_run(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "preprocess.yml")
    )
    action = CPUTuning()
    action.validate(config_obj)
    action.execute(config_obj)


@cpu_tuning.command("summary")
@click.argument("config", type=click.Path(exists=True))
def cpu_tuning_summary(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "cpu_tuning.yml")
    )
    action = ClangPreprocesser(1, False)
    action.print_summary(config_obj)


@ir.group("irs")
def irs():
    pass


@irs.command("run")
@click.argument("config", type=click.Path(exists=True))
@click.option("--parallel-workers", type=int, default=1, help="Parallel wokers")
@click.option("--build-project", type=str, multiple=True)
def ir_compiler_run(config, parallel_workers, build_project) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "cpu_tuning.yml")
    )
    action = IRCompiler(parallel_workers, build_project)
    action.validate(config_obj)
    action.execute(config_obj)


@irs.command("summary")
@click.argument("config", type=click.Path(exists=True))
def ir_compiler_run_summary(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "ir_compilation.yml")
    )
    action = IRCompiler(1, [])
    action.print_summary(config_obj)


@ir.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--docker-repository", type=str, default="spcleth:xaas", help="Docker repository")
def container(config, docker_repository) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "ir_compilation.yml")
    )
    action = DockerImageBuilder(docker_repository=docker_repository)
    action.validate(config_obj)
    action.execute(config_obj)


@ir.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--parallel-workers", type=int, default=1, help="Parallel wokers")
def deploy(config, parallel_workers) -> None:
    initialize()

    config_obj = DeployConfig.load(config)
    action = Deployment(parallel_workers)
    action.validate(config_obj)
    action.execute(config_obj)


@cli.command("build-layers")
@click.argument("dep_name", type=str)
@click.option("--version", type=str, default=None)
def build_layers(dep_name: str, version: str | None) -> None:
    initialize()

    DOCKERFILES_DIR = os.path.join(Path(__file__).parent.parent, "dockerfiles")

    dependency = FeatureType(dep_name.upper())

    if dependency not in XaaSConfig().layers.layers:
        raise ValueError(f"Dependency {dep_name} not found in configuration.")

    dep_config = XaaSConfig().layers.layers[dependency]

    docker_runner = DockerRunner(XaaSConfig().docker_repository)

    if dep_config.arg_mapping:
        versions = dep_config.versions
        if version is not None:
            versions = [version]

        for version in versions:
            for _, flag_config in dep_config.arg_mapping.items():
                # if the argument is the version, then we don't need to iterate across all values
                # since we will iterate over the version.
                if flag_config.flag_name == dep_config.version_arg:
                    name = dep_config.name.replace(f"${{{flag_config.flag_name}}}", version)

                    build_args = {flag_config.flag_name: flag_config.build_args[version]}

                    dockerfile = os.path.join(DOCKERFILES_DIR, dep_config.dockerfile)
                    logging.info(
                        f"Building layer {name} with Dockerfile {dockerfile}, arguments {build_args}"
                    )
                    print(
                        docker_runner.build(
                            dockerfile=dockerfile,
                            path=os.path.curdir,
                            tag=f"{XaaSConfig().docker_repository}:{name}",
                            build_args=build_args,
                        )
                    )

                # non-version argument, iterate over all values
                else:
                    for flag_value, build_arg in flag_config.build_args.items():
                        name = dep_config.name.replace("${version}", version)
                        name = name.replace(f"${{{flag_config.flag_name}}}", flag_value)

                        build_args = {flag_config.flag_name: build_arg}

                        dockerfile = os.path.join(DOCKERFILES_DIR, dep_config.dockerfile)
                        logging.info(
                            f"Building layer {name} with Dockerfile {dockerfile}, arguments {build_args}"
                        )
                        print(
                            docker_runner.build(
                                dockerfile=dockerfile,
                                path=os.path.curdir,
                                tag=f"{XaaSConfig().docker_repository}:{name}",
                                build_args=build_args,
                            )
                        )

    else:
        versions = dep_config.versions
        if version is not None:
            versions = [version]

        for version in versions:
            name = dep_config.name.replace("${version}", version)

            build_args = {dep_config.version_arg: version}

            dockerfile = os.path.join(DOCKERFILES_DIR, dep_config.dockerfile)
            logging.info(f"Building layer {name} with Dockerfile {dockerfile}")
            docker_runner.build(
                dockerfile=dockerfile,
                path=os.path.curdir,
                tag=f"{XaaSConfig().docker_repository}:{name}",
                build_args=build_args,
            )


@cli.command("build-deps")
@click.argument("dep_name", type=str)
@click.argument("version", type=str, default=None)
def build_deps(dep_name: str, version: str | None) -> None:
    initialize()

    DOCKERFILES_DIR = os.path.join(Path(__file__).parent.parent, "dockerfiles")

    if dep_name not in XaaSConfig().layers.layers_deps:
        raise ValueError(f"Dependency {dep_name} not found in configuration.")

    dep_config = XaaSConfig().layers.layers_deps[dep_name]

    docker_runner = DockerRunner(XaaSConfig().docker_repository)

    if dep_config.arg_mapping:
        for _, flag_config in dep_config.arg_mapping.items():
            for flag_value, build_arg in flag_config.build_args.items():
                versions = dep_config.versions
                if version is not None:
                    versions = [version]

                # FIXME: proper build arg for version
                for version in versions:
                    name = dep_config.name.replace("${version}", version)
                    name = name.replace(f"${{{flag_config.flag_name}}}", flag_value)

                    build_args = {flag_config.flag_name: build_arg}

                    dockerfile = os.path.join(DOCKERFILES_DIR, dep_config.dockerfile)
                    print(
                        docker_runner.build(
                            dockerfile=dockerfile,
                            path=os.path.curdir,
                            tag=f"{XaaSConfig().docker_repository}:{name}",
                            build_args=build_args,
                        )
                    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
