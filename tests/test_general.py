import datetime
from unittest.mock import patch
from pydantic import AwareDatetime, BaseModel, Field
from newhelm.general import current_local_datetime, get_class, get_unique_id


class NestedClass:
    class Layer1:
        class Layer2:
            value: str

        layer_2: Layer2

    layer_1: Layer1


def test_unique_id():
    value = get_unique_id()
    assert isinstance(value, str)
    assert value != ""


def test_get_class():
    assert get_class("test_general", "NestedClass") == NestedClass


def test_get_class_nested():
    assert (
        get_class("test_general", "NestedClass.Layer1.Layer2")
        == NestedClass.Layer1.Layer2
    )


class PydanticWithDateTime(BaseModel):
    timestamp: AwareDatetime = Field(default_factory=current_local_datetime)


def test_datetime_round_trip():
    original = PydanticWithDateTime()
    as_json = original.model_dump_json()
    returned = PydanticWithDateTime.model_validate_json(as_json, strict=True)
    assert original == returned


@patch("newhelm.general.current_local_datetime")
def test_datetime_serialized(mock_date):
    mock_date.return_value = datetime.datetime(
        2024,
        2,
        28,
        10,
        12,
        34,
        1111,
        tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=61200), "MST"),
    )
    current_local_datetime()
    assert mock_date.call_count == 1
    # assert original.model_dump_json() == """{"timestamp":"2024-02-28T10:22:15.380163-07:00"}"""
