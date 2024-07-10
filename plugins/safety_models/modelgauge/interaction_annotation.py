from pydantic import AwareDatetime, BaseModel, Field
from typing import Dict, List, Mapping

from modelgauge.annotation import Annotation
from modelgauge.base_test import TestResult
from modelgauge.general import current_local_datetime
from modelgauge.record_init import InitializationRecord
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.typed_data import TypedData

_Context = TypedData | str | Mapping | None


class SUTInteraction(BaseModel):
    prompt: PromptWithContext
    completion: SUTCompletion


class AnnotationTestItem(BaseModel):
    """The base unit for which an Annotator can be evaluated.

    This test item describes a simple prompt interaction of some SUT.
    The interaction is modeled as a single-turn prompt completion.
    """

    sut_interaction: SUTInteraction

    # TODO: turn context into annotations/labels
    @property
    def context(self):
        """Your test can add one of several serializable types as context, and it will be forwarded."""
        if isinstance(self.context_internal, TypedData):
            return self.context_internal.to_instance()
        return self.context_internal

    context_internal: _Context = None
    """Internal variable for the serialization friendly version of context"""

    def __init__(self, *, sut_interaction, context=None, context_internal=None):
        if context_internal is not None:
            internal = context_internal
        elif isinstance(context, BaseModel):
            internal = TypedData.from_instance(context)
        else:
            internal = context
        super().__init__(sut_interaction=sut_interaction, context_internal=internal)

    # Convince pytest to ignore this class.
    __test__ = False


class MeasuredAnnotationItem(BaseModel):
    test_item: AnnotationTestItem
    measurements: Dict[str, float]


class AnnotatorInteractionRecord(BaseModel):
    """Record of all data relevant to a single annotator test item."""

    test_item: AnnotationTestItem
    annotation: Annotation
    measurements: Dict[str, float]

    __test__ = False


class AnnotatorTestRecord(BaseModel):
    """Record of all data relevant to a single annotator test run."""

    run_timestamp: AwareDatetime = Field(default_factory=current_local_datetime)
    test_uid: str
    test_initialization: InitializationRecord
    dependency_versions: Mapping[str, str]
    annotator_uid: str
    # TODO: Give annotators init. records and uncomment line below
    # annotator_initialization: InitializationRecord
    interaction_records: List[AnnotatorInteractionRecord]
    result: TestResult

    __test__ = False
