from modelgauge.annotator import Annotator
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
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
    ModelType,
)
from modelgauge.annotators.llama_guard_annotator import (
    LlamaGuardAnnotator,
    LlamaGuardAnnotation,
    LLAMA_GUARD_2_DATA,
    translate_llama_guard_response,
)
from modelgauge.safety_eval_context import SafetyEvalContext
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.safety_model_constants import (
    HUMAN_EVAL_DEPENDENCY_KEY,
    DEFAULT_1320_DATASET_PATH,
)
from modelgauge.sut_capabilities import AcceptsChatPrompt
from typing import List, Dict, Mapping
from sklearn import metrics  # type: ignore
from modelgauge.secret_values import InjectSecret
from modelgauge.suts.together_client import TogetherApiKey

TOGETHER_API_KEY_SECRET = InjectSecret(TogetherApiKey)
ANNOTATOR = "llama_guard_2"


@modelgauge_test(requires_sut_capabilities=[AcceptsChatPrompt])
class SafetyModelLlamaGuard2Test(PromptResponseTest):
    def __init__(self, uid: str, together_api_key: TogetherApiKey):
        super().__init__(uid)
        self.together_api_key = together_api_key

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        # Make sure file is loaded into correct directory
        # TODO make local file path a configurable test parameter
        return {HUMAN_EVAL_DEPENDENCY_KEY: LocalData(path=DEFAULT_1320_DATASET_PATH)}

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        return make_annotation_test_items(dependency_helper)

    def get_annotators(self) -> Mapping[str, Annotator]:
        """Annotations currently read directly from context in measurement stage"""
        return {ANNOTATOR: LlamaGuardAnnotator(self.together_api_key)}

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        test_item_prompt_and_context: PromptWithContext = item.test_item.prompts[0]

        # Get human annotations
        test_item_context: SafetyEvalContext = test_item_prompt_and_context.context

        # Get safety model response
        interaction = item.interactions[0]
        llama_guard_annotation = interaction.response.completions[0].get_annotation(
            ANNOTATOR, LlamaGuardAnnotation
        )

        safety_response = SafetyModelResponse(
            is_safe=llama_guard_annotation.is_safe,
            safety_categories=llama_guard_annotation.violation_categories,
            reasoning=None,
        )

        return measure_safety_model_quality(test_item_context, safety_response)

    def aggregate_measurements(self, items):
        y_true = list(
            map(lambda item: item.measurements["human_majority_is_safe"], items)
        )
        y_pred = list(map(lambda item: item.measurements["sut_is_safe"], items))

        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }


# TODO: uncomment the below line to register the test for use. Make sure you've
# followed the prerequisite steps in the README to make the 1320 dataset
# available locally

TESTS.register(
    SafetyModelLlamaGuard2Test,
    "safety_eval_llama_guard_2_1320",
    TOGETHER_API_KEY_SECRET,
)
