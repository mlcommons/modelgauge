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
    elements: List[Derived1 | Derived2]
    any_union: Derived1 | Derived2


def test_pydantic_lack_of_polymorphism_serialize():
    """This test is showing that Pydantic doesn't serialize like we want."""
    wrapper = Wrapper(
        elements=[Derived1(all_have=20, field_1=1), Derived2(all_have=20, field_2=2)],
        any_union=Derived1(all_have=30, field_1=3),
    )
    # This is no longer missing field_1 and field_2 in elements
    assert wrapper.model_dump_json() == (
        """{"elements":[{"all_have":20,"field_1":1},{"all_have":20,"field_2":2}],"any_union":{"all_have":30,"field_1":3}}"""
    )


def test_pydantic_lack_of_polymorphism_deserialize():
    """This test is showing that Pydantic doesn't deserialize like we want."""

    from_json = Wrapper.model_validate_json(
        """{"elements":[{"all_have":20, "field_1": 1},{"all_have":20, "field_2": 2}],"any_union":{"all_have":30,"field_1":3}}""",
        strict=True,
    )
    # These is now Derived1 and Derived2
    assert from_json.elements[0].__class__.__name__ == "Derived1"
    assert from_json.elements[1].__class__.__name__ == "Derived2"
    # This should be Derived1
    assert from_json.any_union.__class__.__name__ == "dict"
