from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from pydantic import BaseModel
from newhelm.secrets import SecretValues, SecretsMixin

from newhelm.single_turn_prompt_response import PromptInteraction


AnnotationType = TypeVar("AnnotationType", bound=BaseModel)


class BaseAnnotator(ABC, Generic[AnnotationType], SecretsMixin):
    """The base class for all annotators."""

    @abstractmethod
    def annotate_test_item(
        self, interactions: List[PromptInteraction]
    ) -> AnnotationType:
        """Returns an annotation for a single TestItem's interactions."""
        pass

    def load(self, secrets: SecretValues) -> None:
        pass
