from abc import ABC, abstractmethod
from typing import List
from newhelm.sut_interaction import SUTInteraction

from newhelm.task import Task


class InteractionAnnotation:
    """Placeholder for data returnd by an annotator"""


class IndependentCallAnnotatorClient:
    """Base class for an annotator that annotates each interaction independently."""

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
    @abstractmethod
    def get_tasks(
        self, sut_interactions: List[SUTInteraction]
    ) -> List[InteractionAnnotationTask]:
        pass


class IndependentCallAnnotator(InteractionAnnotator):
    """Placeholder for things like calling Perspective API."""

    def __init__(self, annotator_client):
        self.annotator_client = annotator_client

    def get_tasks(
        self, sut_interactions: List[SUTInteraction]
    ) -> List[InteractionAnnotationTask]:
        return [
            InteractionAnnotationTask(self.annotator_client, interaction)
            for interaction in sut_interactions
        ]
