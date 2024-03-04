from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

from newhelm.ephemeral_secrets import EphemeralSecrets, InjectSecrets


_T = TypeVar("_T")


def _replace(obj):
    if isinstance(obj, EphemeralSecrets):
        return InjectSecrets()
    return obj


def replace_args_with_injector(
    args: Sequence, kwargs: Mapping[str, Any]
) -> Tuple[List, Dict[str, Any]]:
    """Switch inputs of injectable types with their injectors."""
    replaced_args = []
    replaced_kwargs = {}
    for arg in args:
        replaced_args.append(_replace(arg))
    for key, arg in kwargs.items():
        replaced_kwargs[key] = _replace(arg)
    return (replaced_args, replaced_kwargs)


def _inject(cls: Type[_T], obj, secrets: Optional[EphemeralSecrets] = None):
    if isinstance(obj, InjectSecrets):
        assert secrets is not None, f"Class {cls} requires secrets, but none provided"
        return secrets
    return obj


def create_obj(
    cls: Type[_T],
    args: Sequence,
    kwargs: Mapping[str, Any],
    secrets: Optional[EphemeralSecrets] = None,
) -> _T:
    """Return cls(*args, **kwargs) after performing dependency injection."""
    replaced_args = []
    replaced_kwargs = {}
    for arg in args:
        replaced_args.append(_inject(cls, arg, secrets=secrets))

    for key, arg in kwargs.items():
        replaced_kwargs[key] = _inject(cls, arg, secrets=secrets)
    return cls(*replaced_args, **replaced_kwargs)
