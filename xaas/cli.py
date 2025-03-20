#!/usr/bin/env python3

import logging
import os

import click

from xaas.actions.analyze import BuildAnalyzer
from xaas.actions.build import BuildGenerator
from xaas.actions.build import Config as BuildConfig
from xaas.config import RunConfig
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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
