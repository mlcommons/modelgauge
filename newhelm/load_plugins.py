"""
This namespace plugin loader will discover and load all plugins from newhelm's plugin directories.

To see this in action:

* poetry install
* poetry run newhelm list
* poetry install --extras demo
* poetry run newhelm list

The demo plugin modules will only print on the second run.
"""

import importlib

from tqdm import tqdm
import newhelm
import newhelm.annotators
import newhelm.runners
import newhelm.suts
import newhelm.tests
import pkgutil
from types import ModuleType
from typing import Iterator, List


def _iter_namespace(ns_pkg: ModuleType) -> Iterator[pkgutil.ModuleInfo]:
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def list_plugins() -> List[str]:
    module_names = []
    for ns in ["tests", "suts", "runners", "annotators"]:
        for _, name, _ in _iter_namespace(getattr(newhelm, ns)):
            module_names.append(name)
    return module_names


def load_plugins(disable_progress_bar: bool = False) -> None:
    for module_name in tqdm(
        list_plugins(), desc="Loading plugins", disable=disable_progress_bar
    ):
        importlib.import_module(module_name)
