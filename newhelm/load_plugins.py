"""
This is an example of a very simple namespace plugin loader that will discover and load all plugins from newhelm/plugins
and will declare and run all of those discovered plugins.

To see this in action:

* poetry install
* poetry run python newhelm/load_plugins.py
* poetry install --extras demo
* poetry run python newhelm/load_plugins.py

The demo plugin modules will only print on the second run.
"""
import importlib
import logging
import pkgutil
from types import ModuleType
from typing import Iterator

import click

import newhelm
import newhelm.annotators
import newhelm.benchmarks
import newhelm.suts
import newhelm.tests
import newhelm.runners


def iter_namespace(ns_pkg: ModuleType) -> Iterator[pkgutil.ModuleInfo]:
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def load_plugins() -> None:
    for ns in ["tests", "suts", "benchmarks", "runners", "annotators"]:
        for _, name, _ in iter_namespace(getattr(newhelm, ns)):
            logging.info(f"Importing: {name}")
            importlib.import_module(name)


if __name__ == "__main__":
    load_plugins()
    for ns in ["tests", "suts", "benchmarks", "runners", "annotators"]:
        click.echo(click.style(f"These are the {ns} modules I know about:", bold=True))
        for plugin in list(pkgutil.iter_modules(getattr(newhelm, ns).__path__)):
            click.echo("\t" + plugin.name)
        click.echo("\n")
