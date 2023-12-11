from typing import Set, Type, TypeVar
from newhelm.base_test import BaseTest
from newhelm.load_plugins import load_plugins

InT = TypeVar("InT")


def get_all_subclasses(cls: Type[InT]) -> Set[Type[InT]]:
    result = set()
    for subclass in cls.__subclasses__():
        result.add(subclass)
        result.update(get_all_subclasses(subclass))
    return result


if __name__ == "__main__":
    load_plugins()
    for test_cls in get_all_subclasses(BaseTest):  # type: ignore[type-abstract]
        print("Fully qualified name of the test:", test_cls)
        print("Running that test:")
        test_cls().run()
