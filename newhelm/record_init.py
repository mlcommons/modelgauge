from functools import wraps
import importlib
from typing import Any, List, Mapping, Optional
from pydantic import BaseModel
from newhelm.dependency_injection import create_obj, replace_args_with_injector

from newhelm.ephemeral_secrets import EphemeralSecrets


class InitializationRecord(BaseModel):
    """Holds data sufficient to reconstruct an object."""

    module: str
    qual_name: str
    args: List[Any]
    kwargs: Mapping[str, Any]

    def recreate_object(self, secrets: Optional[EphemeralSecrets] = None):
        """Redoes the init call from this record."""
        cls = getattr(importlib.import_module(self.module), self.qual_name)
        return create_obj(cls, self.args, self.kwargs, secrets=secrets)


def record_init(init):
    """Decorator for the __init__ function to store what arguments were passed."""

    @wraps(init)
    def wrapped_init(*args, **kwargs):
        self, real_args = args[0], args[1:]
        # We want the outer-most init to be recorded, so don't overwrite it.
        if not hasattr(self, "_initialization_record"):
            # When using dependency injection, we don't want to record the injected
            # dependency.
            record_args, record_kwargs = replace_args_with_injector(real_args, kwargs)
            self._initialization_record = InitializationRecord(
                module=self.__class__.__module__,
                qual_name=self.__class__.__qualname__,
                args=record_args,
                kwargs=record_kwargs,
            )
        init(*args, **kwargs)

    return wrapped_init


def get_initialization_record(obj) -> InitializationRecord:
    """Get the initialization record from an object."""
    try:
        return obj._initialization_record
    except AttributeError:
        raise AssertionError(
            f"Class {obj.__class__.__qualname__} in module "
            f"{obj.__class__.__module__} needs to add "
            f"`@record_init` to its `__init__` function to "
            f"enable system reproducibility."
        )
