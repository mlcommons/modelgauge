from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any


class RequiresCredentials(ABC):
    @abstractmethod
    def credential_instructions(self) -> str:
        """Return a string description on how to set up the credentials file."""
        pass

    @abstractmethod
    def load_credentials(self, secrets_dir: str) -> None:
        """Read from the secrets_dir to set variables on `self`."""


def optionally_load_credentials(obj: Any, secrets_dir: str) -> None:
    if not isinstance(obj, RequiresCredentials):
        return
    try:
        obj.load_credentials(secrets_dir)
    except (FileNotFoundError, JSONDecodeError, KeyError) as e:
        raise AssertionError(
            "Instructions for making credential file:\n" + obj.credential_instructions()
        ) from e
