import os
import json
from modelgauge.annotator import Annotator
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.test_decorator import modelgauge_test
from modelgauge.external_data import ExternalData, LocalData
from modelgauge.test_registry import TESTS
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    TestItem,
    TestItemAnnotations,
    PromptWithContext,
)
from modelgauge.safety_model_utils import (
    find_1320_dataset_human_annotator_columns,
    clean_imported_1320_df,
    validate_1320_dataset,
    measure_safety_model_quality,
)
from modelgauge.annotators.llama_guard_annotator import (
    LlamaGuardAnnotation,
    LLAMA_GUARD_2_DATA,
    translate_llama_guard_response,
)
from modelgauge.safety_eval_context import SafetyEvalContext
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.safety_model_constants import MLC_TAXONOMY_0_5, DEFAULT_MAX_TOKENS
from modelgauge.sut_capabilities import AcceptsTextPrompt, AcceptsChatPrompt
from typing import List, Dict, Optional, Mapping
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn import metrics  # type: ignore
import traceback


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SafetyModelLlamaGuard2Test(PromptResponseTest):
    def get_dependencies(self) -> Mapping[str, ExternalData]:
        # Make sure file is loaded into correct directory
        # TODO make local file path a configurable test parameter
        return {"human_eval_dataset": LocalData(path="./1320mlc.csv")}

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        test_items: List[TestItem] = []

        df = pd.read_csv(dependency_helper.get_local_path("human_eval_dataset"))
        assert validate_1320_dataset(df), "Dataset is not valid"

        # Clean dataset for weird nan values
        df = clean_imported_1320_df(df)

        annotator_columns = find_1320_dataset_human_annotator_columns(df)
        for index, row in df.iterrows():
            # TODO parse row using a pydantic class
            try:
                user_prompt = row["prompt"]
                assistant_response = row["response"]
                uid = row["UID"]
                labeled_hazard_category = row["Hazard category"]
                lg_1_is_safe = row["is_safe"]
                lg_1_violation_categories = row["violation_categories"]

                # Parse human annotations
                annotations = {}
                for column in annotator_columns:
                    annotator_name = column.split("_")[0]
                    annotation = row[column]
                    annotations[annotator_name] = annotation

            except Exception as e:
                print(traceback.format_exc())
                print(f"Error reading row {index}: {e}")
                print(f"Row data: {row}")
                continue

            # Create a test item
            prompt = PromptWithContext(
                prompt=TextPrompt(
                    text=f"User: {user_prompt}\nAssistant: {assistant_response}"
                ),
                source_id=uid,
                context=SafetyEvalContext(
                    uid=uid,
                    prompt=user_prompt,
                    response=assistant_response,
                    hazard_category=labeled_hazard_category,
                    annotations=annotations,
                    lg_1_is_safe=lg_1_is_safe,
                    lg_1_violation_categories=lg_1_violation_categories,
                ),
            )
            test_items.append(TestItem(prompts=[prompt]))

        return test_items

    def get_annotators(self) -> Mapping[str, Annotator]:
        """Annotations currently read directly from context in measurement stage"""
        return {}

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        test_item_prompt_and_context: PromptWithContext = item.test_item.prompts[0]

        # Get human annotations
        test_item_context: SafetyEvalContext = test_item_prompt_and_context.context

        # Get safety model response
        interaction = item.interactions[0]
        response_text = interaction.response.completions[0].completion.text
        llama_guard_annotation: LlamaGuardAnnotation = translate_llama_guard_response(
            response_text, LLAMA_GUARD_2_DATA
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

TESTS.register(SafetyModelLlamaGuard2Test, "safety_eval_llama_guard_2_1320")
