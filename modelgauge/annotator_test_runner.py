import os
import random

from modelgauge.safety_eval_context import SafetyEvalContext
from modelgauge.annotation import Annotation
from modelgauge.annotator import CompletionAnnotator
from modelgauge.base_test import PromptResponseTest, TestResult
from modelgauge.caching import Cache, NoCache, SqlDictCache
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.prompt import TextPrompt
from modelgauge.records import TestItemRecord, TestRecord
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptInteractionAnnotations,
    SUTCompletionAnnotations,
    SUTResponseAnnotations,
    TestItem,
    TestItemAnnotations,
)
from modelgauge.sut import SUTCompletion
from modelgauge.sut_capabilities_verification import assert_sut_capabilities
from modelgauge.sut_decorator import assert_is_sut
from modelgauge.test_decorator import assert_is_test
from tqdm import tqdm
from typing import List, Optional


def run_annotator_test(
    test: PromptResponseTest,
    annotator: CompletionAnnotator,
    data_dir: str,
    max_test_items: Optional[int] = None,
    use_caching: bool = True,
    disable_progress_bar: bool = False,
) -> TestRecord:
    """Demonstration for how to run a single Test on a single SUT, all calls serial."""

    assert_is_test(test)

    # Ensure we can record what these objects are
    test_initialization = test.initialization_record
    test_data_path = os.path.join(data_dir, "tests", test.__class__.__name__)

    # Instead of getting annotators, we should be reading it as an arg
    annotator_cache: Cache
    if use_caching:
        annotator_cache = SqlDictCache(
            os.path.join(test_data_path, "annotators"), annotator.uid
        )
    else:
        annotator_cache = NoCache()
    assert isinstance(
        annotator, CompletionAnnotator
    ), "Only know how to do CompletionAnnotator."
    annotator_data = AnnotatorData(annotator.uid, annotator, annotator_cache)

    # This runner just records versions, it doesn't specify a required version.
    dependency_helper = FromSourceDependencyHelper(
        os.path.join(test_data_path, "dependency_data"),
        test.get_dependencies(),
        required_versions={},
    )

    # Still need to create test items with prompt, responses. Just don't need sut for this
    test_items = test.make_test_items(dependency_helper)
    if max_test_items is not None:
        assert max_test_items > 0, f"Cannot run a test using {max_test_items}."
        if max_test_items < len(test_items):
            rng = random.Random()
            rng.seed(0)
            rng.shuffle(test_items)
            test_items = test_items[:max_test_items]
    test_item_records = []
    measured_test_items = []
    desc = f"Processing TestItems for test={test.uid}"
    for test_item in tqdm(test_items, desc=desc, disable=disable_progress_bar):
        test_item_record = _process_test_item(test_item, test, annotator_data)
        test_item_records.append(test_item_record)
        measured_test_items.append(
            MeasuredTestItem(
                test_item=test_item_record.test_item,
                measurements=test_item_record.measurements,
            )
        )
    test_result = TestResult.from_instance(
        test.aggregate_measurements(measured_test_items)
    )
    return TestRecord(
        test_uid=test.uid,
        test_initialization=test_initialization,
        dependency_versions=dependency_helper.versions_used(),
        sut_uid="",
        sut_initialization=None,
        test_item_records=test_item_records,
        result=test_result,
    )


class AnnotatorData:
    """Container to hold data about an annotator."""

    def __init__(self, key: str, annotator: CompletionAnnotator, cache: Cache):
        self.key = key
        self.annotator = annotator
        self.cache = cache


def _process_test_item(
    item: TestItem,
    test: PromptResponseTest,
    annotator_data: AnnotatorData,
) -> TestItemRecord:
    interactions: List[PromptInteractionAnnotations] = []
    for prompt in item.prompts:
        if isinstance(prompt.context, SafetyEvalContext):
            response = prompt.context.response
            completion = SUTCompletion(text=response)
        else:
            raise ValueError("Prompt object does not have SafetyEvalContext object.")

        annotated_completions: List[SUTCompletionAnnotations] = []

        annotations = {}

        annotator = annotator_data.annotator
        try:
            annotator_request = annotator.translate_request(prompt, completion)
            with annotator_data.cache as cache:
                annotator_response = cache.get_or_call(
                    annotator_request, annotator.annotate
                )
            # TODO some check needs to happen here to ensure annotator
            # response is of a certain type. Ideally it should be
            # compile time driven, not runtime
            annotation = annotator.translate_response(
                annotator_request, annotator_response
            )
        except Exception as e:
            raise Exception(
                f"Exception while handling annotation for {annotator_data.key} on {prompt}"
            ) from e

        annotations[annotator_data.key] = Annotation.from_instance(annotation)

        annotated_completions.append(
            SUTCompletionAnnotations(completion=completion, annotations=annotations)
        )

        interactions.append(
            PromptInteractionAnnotations(
                prompt=prompt,
                response=SUTResponseAnnotations(completions=annotated_completions),
            )
        )
    annotated = TestItemAnnotations(
        test_item=item,
        interactions=interactions,
    )
    measurements = test.measure_quality(annotated)

    return TestItemRecord(
        test_item=annotated.test_item,
        interactions=annotated.interactions,
        measurements=measurements,
    )
