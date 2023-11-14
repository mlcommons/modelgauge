from dataclasses import dataclass
from newhelm.general import from_json, to_json

from newhelm.schema.traceable_object import TraceableObject


@dataclass(frozen=True)
class SimpleTraceable(TraceableObject):
    """This class is used for testing TraceableObject"""

    value: str


def test_json_round_trip_traceable():
    original = SimpleTraceable("this is the value")
    as_json = to_json(original)
    returned = from_json(SimpleTraceable, as_json)
    assert original == returned
