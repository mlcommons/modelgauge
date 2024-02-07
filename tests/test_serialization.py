from abc import ABC
from typing import List
from pydantic import BaseModel


class SomeBase(BaseModel, ABC):
    all_have: int


class Derived1(SomeBase):
    field_1: int


class Derived2(SomeBase):
    field_2: int


class Wrapper(BaseModel):
    elements: List[SomeBase]
    bar: str

def test_pydantic_lack_of_polymorphism_serialize():
    """This test is showing that Pydantic doesn't serialize like we want."""
    wrapper = Wrapper(
        elements=[Derived1(all_have=20, field_1=1), Derived2(all_have=20, field_2=2)],
        bar="foo"
    )
    # This is missing field_1 and field_2
    assert wrapper.model_dump_json() == (
        """{"elements":[{"all_have":20},{"all_have":20}]}"""
    )


def test_pydantic_lack_of_polymorphism_deserialize():
    """This test is showing that Pydantic doesn't deserialize like we want."""

    from_json = Wrapper.model_validate_json(
        """{"elements":[{"all_have":20, "field_1": 1},{"all_have":20, "field_2": 2}]}""",
        strict=True,
    )
    # These should be Derived1 and Derived2
    assert from_json.elements[0].__class__.__name__ == "SomeBase"
    assert from_json.elements[1].__class__.__name__ == "SomeBase"
