from dataclasses import dataclass
import threading
from typing import Any, Dict, Generic, List, Sequence, Tuple, Type, TypeVar
from newhelm.dependency_injection import inject_dependencies

from newhelm.secret_values import MissingSecretValues, RawSecrets
from newhelm.tracked_object import TrackedObject


_T = TypeVar("_T", bound=TrackedObject)


@dataclass(frozen=True)
class FactoryEntry(Generic[_T]):
    cls: Type[_T]
    uid: str
    args: Tuple[Any]
    kwargs: Dict[str, Any]

    def __str__(self):
        """Return a string representation of the entry."""
        return f"{self.cls.__name__}(uid={self.uid}, args={self.args}, kwargs={self.kwargs})"

    def make_instance(self, *, secrets: RawSecrets) -> _T:
        args, kwargs = inject_dependencies(self.args, self.kwargs, secrets=secrets)
        return self.cls(self.uid, *args, **kwargs)  # type: ignore [call-arg]

    def get_missing_dependencies(
        self, *, secrets: RawSecrets
    ) -> Sequence[MissingSecretValues]:
        # TODO: Handle more kinds of dependency failure.
        try:
            inject_dependencies(self.args, self.kwargs, secrets=secrets)
        except MissingSecretValues as e:
            return [e]
        return []


class InstanceFactory(Generic[_T]):
    """Generic class that lets you store how to create instances of a given type."""

    def __init__(self) -> None:
        self._lookup: Dict[str, FactoryEntry[_T]] = {}
        self.lock = threading.Lock()

    def register(self, cls: Type[_T], uid: str, *args, **kwargs):
        """Add value to the registry, ensuring it has a unique key."""
        with self.lock:
            previous = self._lookup.get(uid)
            assert previous is None, (
                f"Factory already contains {uid} set to "
                f"{previous.cls.__name__}(args={previous.args}, "
                f"kwargs={previous.kwargs})."
            )
            self._lookup[uid] = FactoryEntry[_T](cls, uid, args, kwargs)

    def make_instance(self, uid: str, *, secrets: RawSecrets) -> _T:
        """Create an instance using the  class and arguments passed to register, raise exception if missing."""
        entry = self._get_entry(uid)
        return entry.make_instance(secrets=secrets)

    def get_missing_dependencies(
        self, uid: str, *, secrets: RawSecrets
    ) -> Sequence[MissingSecretValues]:
        entry = self._get_entry(uid)
        return entry.get_missing_dependencies(secrets=secrets)

    def _get_entry(self, uid: str) -> FactoryEntry:
        with self.lock:
            entry: FactoryEntry
            try:
                entry = self._lookup[uid]
            except KeyError:
                known_uids = list(self._lookup.keys())
                raise KeyError(f"No registration for {uid}. Known uids: {known_uids}")
        return entry

    def items(self) -> List[Tuple[str, FactoryEntry[_T]]]:
        """List all items in the registry."""
        with self.lock:
            return list(self._lookup.items())
