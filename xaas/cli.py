#!/usr/bin/env python3

import logging

import click

from xaas.actions.analyze import BuildAnalyzer
from xaas.actions.analyze import Config as AnalyzerConfig
from xaas.actions.container import DockerImageBuilder

from xaas.actions.ir import IRCompiler
from xaas.actions.build import BuildGenerator
from xaas.actions.build import Config as BuildConfig
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

    config_obj = BuildConfig.load(config)
    action = BuildAnalyzer()
    action.validate(config_obj)
    action.execute(config_obj)


@cli.group()
def preprocess():
    pass


@preprocess.command("run")
@click.argument("config", type=click.Path(exists=True))
@click.option("--parallel-workers", type=int, default=1, help="Parallel wokers")
@click.option("--openmp-check", is_flag=True, help="Enable OpenMP support")
def preprocess_run(config, parallel_workers, openmp_check) -> None:
    initialize()

    config_obj = AnalyzerConfig.load(config)
    action = ClangPreprocesser(parallel_workers, openmp_check)
    action.validate(config_obj)
    action.execute(config_obj)


@preprocess.command("summary")
@click.argument("config", type=click.Path(exists=True))
def preprocess_summary(config) -> None:
    initialize()

    config_obj = AnalyzerConfig.load(config)
    action = ClangPreprocesser()
    action.print_summary(config_obj)


@cli.group()
def ir():
    pass


@ir.command("run")
@click.argument("config", type=click.Path(exists=True))
@click.option("--parallel-workers", type=int, default=1, help="Parallel wokers")
def ir_compiler_run(config, parallel_workers) -> None:
    initialize()

    config_obj = PreprocessingResult.load(config)
    action = IRCompiler(parallel_workers)
    action.validate(config_obj)
    action.execute(config_obj)


@ir.command("summary")
@click.argument("config", type=click.Path(exists=True))
def ir_compiler_run_summary(config) -> None:
    initialize()

    config_obj = PreprocessingResult.load(config)
    action = IRCompiler(1)
    # action.print_summary(config_obj)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def container(config) -> None:
    initialize()

    config_obj = PreprocessingResult.load(config)
    action = DockerImageBuilder()
    action.validate(config_obj)
    action.execute(config_obj)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def deploy(config) -> None:
    initialize()

    config_obj = DeployConfig.load(config)
    action = Deployment()
    action.validate(config_obj)
    action.execute(config_obj)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
