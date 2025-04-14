#!/usr/bin/env python3

import logging
import os

import click

from xaas.actions.analyze import BuildAnalyzer
from xaas.actions.analyze import Config as AnalyzerConfig
from xaas.actions.container import DockerImageBuilder

from xaas.actions.ir import IRCompiler
from xaas.actions.build import BuildGenerator
from xaas.actions.build import Config as BuildConfig
from xaas.actions.cpu_tuning import CPUTuning
from xaas.actions.deployment import Deployment
from xaas.actions.preprocess import ClangPreprocesser, PreprocessingResult
from xaas.config import DeployConfig, RunConfig
from xaas.config import XaaSConfig


def initialize():
    config = XaaSConfig()
    config.initialize(XaaSConfig.DEFAULT_CONFIGURATION)

    logging.basicConfig(level=logging.INFO, force=True)


@click.group()
@click.version_option()
def cli() -> None:
    logging.info("XaaS Builder")


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def buildgen(config) -> None:
    initialize()
    logging.info("t")

    config_obj = RunConfig.load(config)
    action = BuildGenerator()
    action.validate(config_obj)
    action.execute(config_obj)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def analyze(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = BuildConfig.load(os.path.join(run_config.working_directory, "buildgen.yml"))
    action = BuildAnalyzer()
    action.validate(config_obj)
    action.execute(config_obj)


@cli.group()
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
        # os.path.join(run_config.working_directory, "build_analyze.yml")
        os.path.join(run_config.working_directory, "cpu_tuning.yml")
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


@cli.group("cpu-tuning")
def cpu_tuning():
    pass


@cpu_tuning.command("run")
@click.argument("config", type=click.Path(exists=True))
def cpu_tuning_run(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = AnalyzerConfig.load(
        os.path.join(run_config.working_directory, "build_analyze.yml")
    )
    action = CPUTuning()
    action.validate(config_obj)
    action.execute(config_obj)


@cpu_tuning.command("summary")
@click.argument("config", type=click.Path(exists=True))
def cpu_tuning_summary(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = AnalyzerConfig.load(os.path.join(run_config.working_directory, "cpu_tuning.yml"))
    action = BuildAnalyzer()
    action.print_summary(config_obj)


@cli.group()
def ir():
    pass


@ir.command("run")
@click.argument("config", type=click.Path(exists=True))
@click.option("--parallel-workers", type=int, default=1, help="Parallel wokers")
@click.option("--build-project", type=str, multiple=True)
def ir_compiler_run(config, parallel_workers, build_project) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "preprocess.yml")
    )
    action = IRCompiler(parallel_workers, build_project)
    action.validate(config_obj)
    action.execute(config_obj)


@ir.command("summary")
@click.argument("config", type=click.Path(exists=True))
def ir_compiler_run_summary(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "ir_compilation.yml")
    )
    action = IRCompiler(1, [])
    action.print_summary(config_obj)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def container(config) -> None:
    initialize()

    run_config = RunConfig.load(config)

    config_obj = PreprocessingResult.load(
        os.path.join(run_config.working_directory, "ir_compilation.yml")
    )
    action = DockerImageBuilder()
    action.validate(config_obj)
    action.execute(config_obj)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--parallel-workers", type=int, default=1, help="Parallel wokers")
def deploy(config, parallel_workers) -> None:
    initialize()

    config_obj = DeployConfig.load(config)
    action = Deployment(parallel_workers)
    action.validate(config_obj)
    action.execute(config_obj)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
