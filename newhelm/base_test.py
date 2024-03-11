from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Mapping, TypeVar

from pydantic import BaseModel
from newhelm.base_annotator import BaseAnnotator
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData

from newhelm.record_init import record_init
from newhelm.single_turn_prompt_response import (
    TestItemAnnotations,
    MeasuredTestItem,
    TestItem,
)


class TestMetadata(BaseModel):
    """Structured information about the Test which we can use in the UI to list available tests
    as well as for filtering to specific kinds of tests."""

    name: str
    description: str

    # Convince pytest to ignore this class.
    __test__ = False


class BaseTest(ABC):
    """This is the placeholder base class for all tests."""

    @abstractmethod
    def get_metadata(self) -> TestMetadata:
        """Return a description of the test."""
        pass

    @record_init
    def __init__(self):
        """Ensure all Tests default to recording their initialization.

        We want to ensure all Tests record their init to allow us to reconstruct
        their behavior later. If a Test needs to define its own __init__ that is fine,
        it should just include the decorator.
        """
        pass


class Result(BaseModel):
    """The measurement produced by Test."""

    name: str
    value: float


# All ResultTypes should be Pydantic objects.
ResultType = TypeVar("ResultType", bound=BaseModel)


class BasePromptResponseTest(BaseTest, ABC, Generic[ResultType]):
    """This is the base class for all tests that are single turn."""

    @abstractmethod
    def get_dependencies(self) -> Mapping[str, ExternalData]:
        """Return a mapping of external dependency name to how it can be found downloaded."""
        pass

    @abstractmethod
    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        """Generate all data that will eventually go to the SUT."""
        pass

    # TODO: Consider making this method default to returning an empty dict.
    @abstractmethod
    def get_annotators(self) -> Mapping[str, BaseAnnotator]:
        pass

    @abstractmethod
    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        """Use the SUT responses with annotations to determine how well the SUT did on this TestItem."""
        pass

    @abstractmethod
    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> ResultType:
        """Combine the measurements for each TestItem into a test specific ResultType."""
        pass
