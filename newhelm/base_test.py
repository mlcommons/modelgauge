from abc import ABC, abstractmethod
from typing import List
from newhelm.annotation import AnnotatedInteraction

from newhelm.placeholders import PromptTemplate, Result


class BaseTest(ABC):
    """This is the placeholder base class for all tests."""

    @abstractmethod
    def run(self):
        pass


class BasePromptResponseTest(ABC):
    @abstractmethod
    def make_prompt_templates(self) -> List[PromptTemplate]:
        pass

    # TODO Insert a method here for how the test can specify what annotators to run.

    @abstractmethod
    def calculate_results(
        self, interactions: List[AnnotatedInteraction]
    ) -> List[Result]:
        pass
