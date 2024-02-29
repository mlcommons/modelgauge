from dataclasses import dataclass
import threading
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar
from newhelm.dependency_injection import create_obj

from newhelm.ephemeral_secrets import EphemeralSecrets


_T = TypeVar("_T")


@dataclass(frozen=True)
class FactoryEntry(Generic[_T]):
    cls: Type[_T]
    args: Tuple[Any]
    kwargs: Dict[str, Any]

    def __str__(self):
        """Return a string representation of the entry."""
        return f"{self.cls.__name__}(args={self.args}, kwargs={self.kwargs})"

    def make_instance(self, secrets: Optional[EphemeralSecrets] = None) -> _T:
        return create_obj(self.cls, self.args, self.kwargs, secrets=secrets)


class InstanceFactory(Generic[_T]):
    """Generic class that lets you store how to create instances of a given type."""

    def __init__(self) -> None:
        self._lookup: Dict[str, FactoryEntry[_T]] = {}
        self.lock = threading.Lock()

    def register(self, key: str, cls: Type[_T], *args, **kwargs):
        """Add value to the registry, ensuring it has a unique key."""
        with self.lock:
            previous = self._lookup.get(key)
            assert previous is None, (
                f"Factory already contains {key} set to "
                f"{previous.cls.__name__}(args={previous.args}, "
                f"kwargs={previous.kwargs})."
            )
            self._lookup[key] = FactoryEntry[_T](cls, args, kwargs)

    def make_instance(self, key: str, secrets: Optional[EphemeralSecrets] = None) -> _T:
        """Create an instance using the  class and arguments passed to register, raise exception if missing."""
        with self.lock:
            entry: FactoryEntry
            try:
                entry = self._lookup[key]
            except KeyError:
                known_keys = list(self._lookup.keys())
                raise KeyError(f"No registration for {key}. Known keys: {known_keys}")
            return entry.make_instance(secrets=secrets)

    def items(self) -> List[Tuple[str, FactoryEntry[_T]]]:
        """List all items in the registry."""
        with self.lock:
            return list(self._lookup.items())
