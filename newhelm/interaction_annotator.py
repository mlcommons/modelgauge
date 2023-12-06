from abc import ABC, abstractmethod
from typing import List
from newhelm.sut_interaction import SUTInteraction

from newhelm.task import Task


class InteractionAnnotation:
    """Placeholder for data returnd by an annotator"""


class SimpleAnnotatorClient(ABC):
    """Base class for a client that can annotate a single annotation."""

    @abstractmethod
    def annotate(self, interaction: SUTInteraction) -> List[InteractionAnnotation]:
        pass


class PerspectiveAPIAnnotatorClient(SimpleAnnotatorClient):
    """Stub out the primary example of this from HELM."""

    def annotate(self, interaction: SUTInteraction) -> List[InteractionAnnotation]:
        # Return a dummy annotation.
        return [InteractionAnnotation()]


class InteractionAnnotationTask(Task):
    def __init__(self, annotator, interaction):
        self.annotator = annotator
        self.interaction = interaction

    def execute(self) -> List[InteractionAnnotation]:
        return self.annotator.annotate(self.interaction)


class InteractionAnnotator(ABC):
    """Base class for how to request an external service provide more data about the SUTs' behavior."""

    @abstractmethod
    def get_tasks(
        self, sut_interactions: List[SUTInteraction]
    ) -> List[InteractionAnnotationTask]:
        pass


class SimpleInteractionAnnotator(InteractionAnnotator):
    """Makes a separate task for passing each interaction to the client."""

    def __init__(self, annotator_client):
        self.annotator_client = annotator_client

    def get_tasks(
        self, sut_interactions: List[SUTInteraction]
    ) -> List[InteractionAnnotationTask]:
        return [
            InteractionAnnotationTask(self.annotator_client, interaction)
            for interaction in sut_interactions
        ]
