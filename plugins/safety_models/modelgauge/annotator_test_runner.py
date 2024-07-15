import os
import random

from modelgauge.annotator_test import InteractionAnnotationTest
from modelgauge.annotation import Annotation
from modelgauge.annotator import CompletionAnnotator
from modelgauge.base_test import TestResult
from modelgauge.caching import Cache, NoCache, SqlDictCache
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.interaction_annotation import (
    AnnotatorInteractionRecord,
    AnnotationTestItem,
    AnnotatorTestRecord,
    MeasuredAnnotationItem,
)

from modelgauge.test_decorator import assert_is_test
from tqdm import tqdm
from typing import List, Optional


def assert_is_annotator(obj):
    """Raise AssertionError if obj is not decorated with @modelgauge_test."""
    if not getattr(obj, "_modelgauge_test", False):
        raise AssertionError(
            f"{obj.__class__.__name__} should be decorated with @modelgauge_test."
        )


def run_annotator_test(
    test: InteractionAnnotationTest,
    annotator: CompletionAnnotator,
    data_dir: str,
    max_test_items: Optional[int] = None,
    use_caching: bool = True,
    disable_progress_bar: bool = False,
) -> AnnotatorTestRecord:
    """Demonstration for how to run a single Interaction Annotation Test on a single Annotator, all calls serial."""

    assert_is_test(test)
    assert isinstance(
        annotator, CompletionAnnotator
    ), "Only know how to do CompletionAnnotator."
    # TODO: check that Annotator's AnnotationType matches test's AnnotationType.

    # Ensure we can record what these objects are
    test_initialization = test.initialization_record
    # TODO: Give annotators init records
    # annotator_initialization = annotator.initialization_record
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

    # This runner just records versions, it doesn't specify a required version.
    dependency_helper = FromSourceDependencyHelper(
        os.path.join(test_data_path, "dependency_data"),
        test.get_dependencies(),
        required_versions={},
    )

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
    desc = f"Processing AnnotationTestItems for test={test.uid}"
    for test_item in tqdm(test_items, desc=desc, disable=disable_progress_bar):
        test_item_record = _process_test_item(
            test_item, test, annotator, annotator_cache
        )
        test_item_records.append(test_item_record)
        measured_test_items.append(
            MeasuredAnnotationItem(
                test_item=test_item_record.test_item,
                measurements=test_item_record.measurements,
            )
        )
    test_result = TestResult.from_instance(
        test.aggregate_measurements(measured_test_items)
    )
    return AnnotatorTestRecord(
        test_uid=test.uid,
        test_initialization=test_initialization,
        dependency_versions=dependency_helper.versions_used(),
        annotator_uid=annotator.uid,
        # annotator_initialization=annotator_initialization,
        interaction_records=test_item_records,
        result=test_result,
    )


def _process_test_item(
    item: AnnotationTestItem,
    test: InteractionAnnotationTest,
    annotator: CompletionAnnotator,
    annotator_cache: Cache,
) -> AnnotatorInteractionRecord:
    interaction = item.sut_interaction
    try:
        annotator_request = annotator.translate_request(
            interaction.prompt, interaction.completion
        )
        with annotator_cache as cache:
            annotator_response = cache.get_or_call(
                annotator_request, annotator.annotate
            )
        # TODO some check needs to happen here to ensure annotator
        # response is of a certain type. Ideally it should be
        # compile time driven, not runtime
        annotation = annotator.translate_response(annotator_request, annotator_response)
    except Exception as e:
        raise Exception(
            f"Exception while handling annotation for {annotator.uid} on {interaction}"
        ) from e

    measurements = test.measure_quality(item, annotation)
    annotation = Annotation.from_instance(annotation)
    return AnnotatorInteractionRecord(
        test_item=item,
        annotation=annotation,
        measurements=measurements,
    )
