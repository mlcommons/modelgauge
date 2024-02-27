import threading
from typing import Dict, Mapping, Optional, Sequence
from pydantic import BaseModel


class UseSecret(BaseModel):
    scope: str
    key: str
    instructions: str
    required: bool


RawSecrets = Mapping[str, Mapping[str, str]]


class SecretValues:
    class Stored(BaseModel):
        value: Optional[str]
        required: bool

    def __init__(
        self,
        used_secrets: Sequence[UseSecret],
        value_lookup: RawSecrets,
    ):
        self.secrets: Dict[str, Dict[str, SecretValues.Stored]] = {}
        missing_required = []
        for secret in used_secrets:
            if secret.scope not in self.secrets:
                self.secrets[secret.scope] = {}
            try:
                value = value_lookup[secret.scope][secret.key]
            except KeyError:
                if secret.required:
                    missing_required.append(secret)
                value = None
            self.secrets[secret.scope][secret.key] = SecretValues.Stored(
                value=value, required=secret.required
            )
        # TODO Make a first class exception for this failure
        assert not missing_required

    def get_required(self, scope, key) -> str:
        stored = self._lookup(scope, key)
        # TODO Make this a first class exception
        assert stored.required  # Ensures required are documented as required.
        assert stored.value is not None
        return stored.value

    def get_optional(self, scope, key) -> Optional[str]:
        return self._lookup(scope, key).value

    def _lookup(self, scope, key) -> "SecretValues.Stored":
        # TODO make this be specific about the failures
        return self.secrets[scope][key]


class SecretsMixin:
    def get_used_secrets(self) -> Sequence[UseSecret]:
        return []
