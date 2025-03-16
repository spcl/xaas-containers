#!/usr/bin/env python3

import logging
from typing import Optional

import click


@click.group()
@click.version_option()
def cli() -> None:
    logging.info("XaaS Builder")


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
