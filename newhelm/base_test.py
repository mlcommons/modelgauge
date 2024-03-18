from abc import ABC, abstractmethod
from typing import Dict, List, Mapping

from pydantic import BaseModel
from newhelm.base_annotator import BaseAnnotator
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData
from newhelm.init_hooks import ABCInitHooksMetaclass

from newhelm.record_init import add_initialization_record
from newhelm.single_turn_prompt_response import (
    TestItemAnnotations,
    MeasuredTestItem,
    TestItem,
)
from newhelm.typed_data import Typeable, TypedData


class TestMetadata(BaseModel):
    """Structured information about the Test which we can use in the UI to list available tests
    as well as for filtering to specific kinds of tests."""

    name: str
    description: str

    # Convince pytest to ignore this class.
    __test__ = False


class BaseTest(ABC, metaclass=ABCInitHooksMetaclass):
    """This is the placeholder base class for all tests."""

    @abstractmethod
    def get_metadata(self) -> TestMetadata:
        """Return a description of the test."""
        pass

    def _before_init(self, *args, **kwargs):
        add_initialization_record(self, *args, **kwargs)


class BasePromptResponseTest(BaseTest, ABC):
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
    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> Typeable:
        """Combine the measurements for each TestItem into a test specific Typeable."""
        pass


class TestResult(TypedData):
    """Container for plugin defined Test result data.

    Every Test can return data however it wants, so this generically
    records the Test's return type and data.
    You can use `to_instance` to get back to the original form.
    """

    # Convince pytest to ignore this class.
    __test__ = False
