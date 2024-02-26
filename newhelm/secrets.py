import threading
from typing import Dict, Mapping, Optional, Sequence
from pydantic import BaseModel


class UseSecret(BaseModel):
    scope: str
    key: str
    instructions: str
    required: bool


# class SecretsInstructionRegistry:
#     """Store instructions for how to get secrets like api_keys."""

#     def __init__(self) -> None:
#         self._registered: Dict[str, Dict[str, str]] = {}
#         self.lock = threading.Lock()

#     def register(self, scope: str, key: str, instructions: str) -> None:
#         """Record the instructions for obtaining the key."""
#         with self.lock:
#             if scope not in self._registered:
#                 self._registered[scope] = {}
#             previous = self._registered[scope].get(key)
#             if previous is not None:
#                 assert previous == instructions, (
#                     f"The key {key} in {scope} has two different instructions: "
#                     f"{previous} vs {instructions}"
#                 )
#             else:
#                 self._registered[scope][key] = instructions
    
#     def assert_registered(self, scope, key) -> None:
#         error_message = (
#             f"Before you can access the secret `{key}` in `{scope}`, you have to document "
#             "how to obtain the value by calling `register(scope, key, instructions)`."
#         )
#         if not self._registered:
#             raise AssertionError(error_message)
#         elif scope not in self._registered:
#             error_message += (
#                 f" Did you mean one of these scopes? {set(self._registered.keys())}"
#             )
#             raise AssertionError(error_message)
#         elif key not in self._registered[scope]:
#             error_message += f" Did you mean one of these keys in {scope}? {self._registered[scope].keys()}"
#             raise AssertionError(error_message)

#     def get_instructions(self, scope, key) -> str:
#         self.assert_registered(scope, key)
#         return self._registered[scope][key]

class SecretValues:
    class Stored(BaseModel):
        value: Optional[str]
        required: bool

    def __init__(
        self, used_secrets: Sequence[UseSecret], value_lookup: Mapping[str, Mapping[str, str]]
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
        self.registry.assert_registered(scope, key)
        # TODO make this be specific about the failures
        return self.secrets[scope][key]


class SecretsMixin:
    def get_used_secrets(self) -> Sequence[UseSecret]:
        return []
