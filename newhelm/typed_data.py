from typing import Any, Dict, Type, TypeVar
from typing_extensions import Self
from pydantic import BaseModel


_BaseModelType = TypeVar("_BaseModelType", bound=BaseModel)


class TypedData(BaseModel):
    """This is a generic container that allows Pydantic to do polymorphic serialization.

    This is useful in situations where you have an unknown set of classes that could be
    used in a particular field.
    """

    type: str
    data: Dict[str, Any]

    @classmethod
    def from_instance(cls, obj: _BaseModelType) -> Self:
        """Convert the object into a TypedData instance."""
        return cls(type=TypedData._get_type(obj.__class__), data=obj.model_dump())

    def to_instance(self, cls: Type[_BaseModelType]) -> _BaseModelType:
        """Convert this data back into its original type."""
        cls_type = TypedData._get_type(cls)
        assert cls_type == self.type, f"Cannot convert {self.type} to {cls_type}."
        return cls.model_validate(self.data)

    @staticmethod
    def _get_type(cls):
        return f"{cls.__module__}.{cls.__qualname__}"
