from modelgauge.safety_model_constants import DEFAULT_1320_DATASET_PATH
from modelgauge.annotator_test import InteractionAnnotationTest
from modelgauge.interaction_annotation import AnnotationTestItem
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.test_decorator import modelgauge_test
from modelgauge.external_data import ExternalData, LocalData
from modelgauge.test_registry import TESTS
from modelgauge.safety_model_utils import (
    measure_safety_model_quality,
    make_annotation_test_items,
)
from modelgauge.safety_model_response import SafetyModelResponse
from typing import List, Dict, Mapping
from sklearn import metrics  # type: ignore


@modelgauge_test(requires_sut_capabilities=[])
class SafetyModelTest(InteractionAnnotationTest[SafetyModelResponse]):
    def __init__(self, uid: str):
        super().__init__(uid)

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        # Make sure file is loaded into correct directory
        # TODO make local file path a configurable test parameter
        return {"human_eval_dataset": LocalData(path=DEFAULT_1320_DATASET_PATH)}

    def make_test_items(
        self, dependency_helper: DependencyHelper
    ) -> List[AnnotationTestItem]:
        return make_annotation_test_items(dependency_helper)

    def measure_quality(
        self, item: AnnotationTestItem, annotation: SafetyModelResponse
    ) -> Dict[str, float]:
        return measure_safety_model_quality(item, annotation)

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
