from typing import Any, Dict, Mapping, Sequence, Tuple, List
from newhelm.general import get_class
from newhelm.secret_values import (
    BaseSecret,
    MissingSecretValues,
    RawSecrets,
    Injector,
    SerializedSecret,
)


def inject_dependencies(
    args: Sequence[Any], kwargs: Mapping[str, Any], secrets: RawSecrets
) -> Tuple[Sequence[Any], Mapping[str, Any], List[Any]]:
    """Replace any arg or kwarg injectors with their concrete values."""

    def process_item(item, secrets):
        """Process an individual item (arg or kwarg)."""
        try:
            replaced_item = _replace_with_injected(item, secrets)
            if isinstance(item, (Injector, SerializedSecret)):
                used_secrets.append(replaced_item)
            return replaced_item, None
        except MissingSecretValues as e:
            return item, e
        # TODO Catch other kinds of missing dependencies

    replaced_args, missing_secrets = [], []
    used_secrets = []

    for arg in args:
        replaced_arg, missing = process_item(arg, secrets)
        replaced_args.append(replaced_arg)
        if missing:
            missing_secrets.append(missing)

    replaced_kwargs = {}
    for key, kwarg in kwargs.items():
        replaced_kwarg, missing = process_item(kwarg, secrets)
        replaced_kwargs[key] = replaced_kwarg
        if missing:
            missing_secrets.append(missing)

    if missing_secrets:
        raise MissingSecretValues.combine(missing_secrets)

    return replaced_args, replaced_kwargs, used_secrets


def _replace_with_injected(value, secrets: RawSecrets):
    if isinstance(value, Injector):
        return value.inject(secrets)
    if isinstance(value, SerializedSecret):
        cls = get_class(value.module, value.qual_name)
        assert issubclass(cls, BaseSecret)
        return cls.make(secrets)
    return value


def serialize_injected_dependencies(
    args: Sequence[Any], kwargs: Mapping[str, Any]
) -> Tuple[Sequence[Any], Mapping[str, Any]]:
    """Replace any injected values with their safe-to-serialize form."""
    replaced_args = []
    for arg in args:
        replaced_args.append(_serialize(arg))
    replaced_kwargs: Dict[str, Any] = {}
    for key, arg in kwargs.items():
        replaced_kwargs[key] = _serialize(arg)
    return replaced_args, replaced_kwargs


def _serialize(arg):
    # TODO Try to make this more generic.
    if isinstance(arg, BaseSecret):
        return SerializedSecret.serialize(arg)
    return arg
