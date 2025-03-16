#!/usr/bin/env python3

import logging

import click


@click.group()
@click.version_option()
def cli() -> None:
    logging.info("XaaS Builder")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
