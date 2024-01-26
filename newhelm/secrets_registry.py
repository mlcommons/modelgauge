import threading
from typing import Dict, Optional


class SecretsRegistry:
    """Store secrets like api_keys, while requiring documentation."""

    def __init__(self) -> None:
        self._registered: Dict[str, Dict[str, str]] = {}
        self._values: Optional[Dict[str, Dict[str, str]]] = None
        self.lock = threading.Lock()

    def register(self, scope: str, key: str, instructions: str):
        """Record the instructions for obtaining the key."""
        with self.lock:
            if scope not in self._registered:
                self._registered[scope] = {}
            previous = self._registered[scope].get(key)
            if previous is not None:
                assert previous == instructions, (
                    f"The key {key} in {scope} has two different instructions: "
                    f"{previous} vs {instructions}"
                )
            else:
                self._registered[scope][key] = instructions

    def set_values(self, values: Dict[str, Dict[str, str]]):
        """Set all secret values, like from a file."""
        with self.lock:
            self._values = values

    def get_required(self, scope, key) -> str:
        """Retrieve the secret, failing if it isn't available."""
        with self.lock:
            assert (
                self._values is not None
            ), "Must set values before trying to get values."
            self._assert_registered(scope, key, required=True)
            assert self._values, "The values list is empty"
            assert (
                scope in self._values
            ), f"No values stored for {scope}, did you mean {self._values.keys()}?"
            assert (
                key in self._values[scope]
            ), f"Scope {scope} had no value for {key}, did you mean {self._values[scope].keys()}."
            return self._values[scope][key]

    def get_optional(self, scope, key) -> Optional[str]:
        """Retrieve the secret or None if it isn't available.

        Can still fail if the secret isn't probably documented.
        """
        with self.lock:
            assert (
                self._values is not None
            ), "Must set values before trying to get values."
            self._assert_registered(scope, key, required=False)
            scope_dict: Optional[Dict[str, str]] = self._values.get(scope)
            if not scope_dict:
                return None
            return scope_dict.get(key)

    def _assert_registered(self, scope, key, required: bool) -> None:
        error_message = (
            f"Before you can access the secret {key} in {scope}, you have to document "
            "how to obtain the value by calling `register(scope, key, instructions)`."
        )
        if self._values is None:
            error_message += " You also need to set the values."
        elif scope in self._values and key in self._values[scope]:
            error_message += " Good news is there is a value stored for that key."
        elif required:
            error_message += " You'll also need to add a value for that key."
        else:
            error_message += " Adding a value for that key is optional."

        if scope not in self._registered:
            error_message += (
                f" Did you mean one of these scopes? {self._registered.keys()}"
            )
            raise AssertionError(error_message)
        if key not in self._registered[scope]:
            error_message += f" Did you mean one of these keys in {scope}? {self._registered[scope].keys()}"
            raise AssertionError(error_message)


SECRETS = SecretsRegistry()
