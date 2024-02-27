from typing import Dict, Mapping, Optional, Sequence
from pydantic import BaseModel


class UseSecret(BaseModel):
    """Describe a secret used by an object."""

    scope: str
    key: str
    required: bool
    instructions: str


class SecretsMixin:
    """Any object that uses secrets should inherit from this class."""

    def get_used_secrets(self) -> Sequence[UseSecret]:
        """List all of the secrets used by this object."""
        return []


RawSecrets = Mapping[str, Mapping[str, str]]


class SecretValues:
    """Container for secrets that ensures only known `used_secrets` are accessible."""

    class Stored(BaseModel):
        value: Optional[str]
        declared_as: UseSecret

    def __init__(
        self,
        used_secrets: Sequence[UseSecret],
        known_secrets: RawSecrets,
    ):
        self.secrets: Dict[str, Dict[str, SecretValues.Stored]] = {}
        missing_required = []
        for secret in used_secrets:
            if secret.scope not in self.secrets:
                self.secrets[secret.scope] = {}
            try:
                value = known_secrets[secret.scope][secret.key]
            except KeyError:
                if secret.required:
                    missing_required.append(secret)
                value = None
            to_store = SecretValues.Stored(value=value, declared_as=secret)
            try:
                existing = self.secrets[secret.scope][secret.key]
                if existing.declared_as.required and not to_store.declared_as.required:
                    # Keep the required version, otherwise the last version.
                    to_store = existing
            except KeyError:
                pass
            self.secrets[secret.scope][secret.key] = to_store
        if missing_required:
            raise MissingSecretValue(missing_required, known_secrets)

    def get_required(self, scope: str, key: str) -> str:
        """Retrieve the secret and fail if it isn't provided."""
        stored = self._lookup(scope, key)
        assert stored.declared_as.required, (
            f"The secrets {scope}.{key} needs to be listed as "
            f"required in order to use `get_required`."
        )
        assert stored.value is not None, "This should be unreachable"
        return stored.value

    def get_optional(self, scope: str, key: str) -> Optional[str]:
        """Retrieve the secret if available, else return None."""
        return self._lookup(scope, key).value

    def _lookup(self, scope: str, key: str) -> "SecretValues.Stored":
        try:
            return self.secrets[scope][key]
        except KeyError:
            raise AssertionError(
                f"Secret {scope}.{key} must be listed as used before it can be retrieved."
            )


class MissingSecretValue(AssertionError):
    """Raised if no value was provided for a required secret."""

    def __init__(self, missing_secrets: Sequence[UseSecret], known_secrets: RawSecrets):
        self.missing_secrets = missing_secrets
        self.known_secrets: Dict[str, Sequence[str]] = {}
        for scope, values in known_secrets.items():
            # Strip away the sensitive secret values.
            self.known_secrets[scope] = [key for key in values.keys()]

    def __str__(self):
        text = "Missing the following secrets:\n"
        secret_rows = []
        for secret in self.missing_secrets:
            secret_rows.append(
                f"scope=`{secret.scope}` key=`{secret.key}` "
                f"can be obtained by: {secret.instructions}"
            )
        text += "\n".join(secret_rows)
        if self.known_secrets:
            text += "\nKnown secrets:\n"
            known_rows = []
            for scope, values in self.known_secrets.items():
                known_rows.append(f"scope=`{scope}` has values={values}")
            text += "\n".join(known_rows)
        else:
            text += "\nThere are currently no known secrets."

        return text
