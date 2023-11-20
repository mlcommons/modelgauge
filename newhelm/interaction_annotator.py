from abc import ABC, abstractmethod
from typing import List

from newhelm.sut_interaction import SUTInteraction


class InteractionAnnotation:
    """Placeholder for data returnd by an annotator"""


class InteractionAnnotator(ABC):
    name: str

    @abstractmethod
    def annotate(
        self, sut_interactions: List[List[SUTInteraction]]
    ) -> List[List[InteractionAnnotation]]:
        pass


class ModelBasedAnnotator(InteractionAnnotator):
    """Placeholder for things like calling Perspective API."""

    def annotate(
        self, sut_interactions: List[List[SUTInteraction]]
    ) -> List[List[InteractionAnnotation]]:
        pass


class HumanAnnotator(InteractionAnnotator):
    """Placeholder for things like interacting with crowd workers."""

    def annotate(
        self, sut_interactions: List[List[SUTInteraction]]
    ) -> List[List[InteractionAnnotation]]:
        pass
