from pydantic import BaseModel
from typing import Dict, List, Mapping

from modelgauge.annotator_test import InteractionAnnotationTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData
from modelgauge.interaction_annotation import (
    AnnotationTestItem,
    MeasuredAnnotationItem,
    SUTInteraction,
)
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.test_decorator import modelgauge_test
from tests.fake_annotator import FakeAnnotation


def fake_annotation_test_item(prompt_text, completion_text):
    prompt = PromptWithContext(prompt=TextPrompt(text=prompt_text), source_id=None)
    return AnnotationTestItem(
        sut_interaction=SUTInteraction(
            prompt=prompt, completion=SUTCompletion(text=completion_text)
        )
    )


class FakeTestResult(BaseModel):
    count_test_items: int


@modelgauge_test(requires_sut_capabilities=[])
class FakeAnnotatorTest(InteractionAnnotationTest[FakeAnnotation]):
    """Test that lets the user override almost all of the behavior."""

    def __init__(
        self, uid: str = "test-uid", *, dependencies={}, test_items=[], measurement={}
    ):
        super().__init__(uid)
        self.dependencies = dependencies
        self.test_items = test_items
        self.measurement = measurement

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        return self.dependencies

    def make_test_items(
        self, dependency_helper: DependencyHelper
    ) -> List[AnnotationTestItem]:
        return self.test_items

    def measure_quality(
        self, item: AnnotationTestItem, annotation: FakeAnnotation
    ) -> Dict[str, float]:
        return self.measurement

    def aggregate_measurements(
        self, items: List[MeasuredAnnotationItem]
    ) -> FakeTestResult:
        return FakeTestResult(count_test_items=len(items))
