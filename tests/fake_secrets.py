from typing import Optional
from newhelm.ephemeral_secrets import EphemeralSecrets


class FakeSecrets(EphemeralSecrets):
    """Return the same value for all requested secrets."""

    def __init__(self, value="some-value"):
        self.invalidated = False
        self.value = value

    def get_required(self, scope: str, key: str, instructions: str) -> str:
        assert not self.invalidated, "Called after invalidate"
        return self.value

    def get_optional(self, scope: str, key: str, instructions: str) -> Optional[str]:
        assert not self.invalidated, "Called after invalidate"
        return self.value

    def invalidate(self):
        self.invalidated = True
