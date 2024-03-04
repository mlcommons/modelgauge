from typing import List, Mapping, Optional

from pydantic import BaseModel


RawSecrets = Mapping[str, Mapping[str, str]]


class EphemeralSecrets:
    def __init__(self, values: RawSecrets):
        self._values: Optional[RawSecrets] = values
        self.requested: List[RequestedSecret] = []

    def get_required(self, scope: str, key: str, instructions: str) -> str:
        assert self._values is not None, "Must call get_required in __init__."
        self.requested.append(
            RequestedSecret(
                scope=scope,
                key=key,
                required=True,
                value_present=True,
                instructions=instructions,
            )
        )
        try:
            return self._values[scope][key]
        except KeyError:
            raise MissingSecretValue(scope, key, instructions)

    def get_optional(self, scope: str, key: str, instructions: str) -> Optional[str]:
        assert self._values is not None, "Must call get_optional in __init__."
        try:
            value = self._values[scope][key]
        except KeyError:
            value = None
        self.requested.append(
            RequestedSecret(
                scope=scope,
                key=key,
                required=False,
                value_present=value is not None,
                instructions=instructions,
            )
        )
        return value

    def invalidate(self):
        self._values = None


class MissingSecretValue(LookupError):
    def __init__(self, scope: str, key: str, instructions: str):
        self.scope = scope
        self.key = key
        self.instructions = instructions

    def __str__(self):
        return (
            f"Missing value for secret scope=`{self.scope}` key=`{self.key}`. "
            f"Instructions for obtaining that value: {self.instructions}"
        )


class RequestedSecret(BaseModel):
    scope: str
    key: str
    required: bool
    value_present: bool
    instructions: str


class InjectSecrets(BaseModel):
    """Placeholder to say we need to inject secrets to this variable."""

    pass
