from modelgauge.safety_model_constants import DEFAULT_1320_DATASET_PATH
from modelgauge.suts.together_client import TogetherApiKey
from modelgauge.annotator import Annotator
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.secret_values import InjectSecret
from modelgauge.test_decorator import modelgauge_test
from modelgauge.external_data import ExternalData, LocalData
from modelgauge.test_registry import TESTS
from modelgauge.single_turn_prompt_response import (
    TestItem,
    TestItemAnnotations,
    PromptWithContext,
)
from modelgauge.safety_model_utils import (
    measure_safety_model_quality,
    make_annotation_test_items,
)
from modelgauge.safety_eval_context import SafetyEvalContext
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt
from typing import List, Dict, Mapping
from sklearn import metrics  # type: ignore

TOGETHER_API_KEY_SECRET = InjectSecret(TogetherApiKey)


@modelgauge_test(requires_sut_capabilities=[AcceptsChatPrompt])
class SafetyModelTest(PromptResponseTest):
    def get_dependencies(self) -> Mapping[str, ExternalData]:
        # Make sure file is loaded into correct directory
        # TODO make local file path a configurable test parameter
        return {"human_eval_dataset": LocalData(path=DEFAULT_1320_DATASET_PATH)}

    def get_annotators(self) -> Mapping[str, Annotator]:
        return {}

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        return make_annotation_test_items(dependency_helper)

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        test_item_prompt_and_context: PromptWithContext = item.test_item.prompts[0]

        # Get human annotations
        test_item_context: SafetyEvalContext = test_item_prompt_and_context.context

        # Get safety model response
        interaction = item.interactions[0]

        # Assume only 1 annotator (for now as we test 1 annotator at a time)
        assert len(interaction.response.completions[0].annotations.keys()) == 1
        annotation_key = list(interaction.response.completions[0].annotations.keys())[0]

        safety_response = interaction.response.completions[0].get_annotation(
            annotation_key, SafetyModelResponse
        )

        return measure_safety_model_quality(test_item_context, safety_response)

    def aggregate_measurements(self, items):
        y_true = list(
            map(lambda item: item.measurements["human_majority_is_unsafe"], items)
        )
        y_pred = list(map(lambda item: item.measurements["sut_is_unsafe"], items))

        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }


TESTS.register(SafetyModelTest, "safety_eval_1320")
