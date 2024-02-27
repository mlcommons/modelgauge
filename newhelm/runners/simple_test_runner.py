import os
import random
from typing import Dict, List, Optional
from pydantic import BaseModel
from tqdm import tqdm
from newhelm.annotation import Annotation
from newhelm.base_annotator import BaseAnnotator
from newhelm.base_test import BasePromptResponseTest
from newhelm.caching import BaseCache, NoCache, SqlDictCache
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.prompt import TextPrompt
from newhelm.record_init import get_initialization_record
from newhelm.records import TestItemRecord, TestRecord
from newhelm.single_turn_prompt_response import (
    TestItem,
    TestItemAnnotations,
    MeasuredTestItem,
    PromptInteraction,
    TestItemInteractions,
)
from newhelm.sut import PromptResponseSUT


def run_prompt_response_test(
    test_name: str,
    test: BasePromptResponseTest,
    sut_name: str,
    sut: PromptResponseSUT,
    data_dir: str,
    max_test_items: Optional[int] = None,
    use_caching: Optional[bool] = True,
) -> TestRecord:
    """Demonstration for how to run a single Test on a single SUT, all calls serial."""

    # Ensure we can record what these objects are
    test_initialization = get_initialization_record(test)
    sut_initialization = get_initialization_record(sut)
    test_data_path = os.path.join(data_dir, test.get_metadata().name)

    # This runner just records versions, it doesn't specify a required version.
    dependency_helper = FromSourceDependencyHelper(
        test_data_path,
        test.get_dependencies(),
        required_versions={},
    )

    test_items = test.make_test_items(dependency_helper)
    if max_test_items and max_test_items < len(test_items):
        rng = random.Random()
        rng.seed(0)
        test_items = rng.sample(test_items, max_test_items)
    item_interactions: List[TestItemInteractions] = []
    sut_cache: BaseCache
    if use_caching:
        directory = os.path.join(test_data_path, "cached_responses")
        sut_cache = SqlDictCache(directory, sut_name)
    else:
        sut_cache = NoCache()
    with sut_cache as cache:
        item_interactions = _collect_sut_responses(
            test_name, test_items, sut_name, sut, cache
        )
    annotations_per_annotator: Dict[str, List[Annotation]] = {}
    keyed_annotators = test.get_annotators().items()
    for key, annotator in keyed_annotators:
        annotator_cache: BaseCache
        if use_caching:
            annotator_cache = SqlDictCache(
                os.path.join(test_data_path, "cached_annotations"), key
            )
        else:
            annotator_cache = NoCache()
        with annotator_cache as cache:
            annotations = _collect_annotations(key, annotator, item_interactions, cache)
        annotations_per_annotator[key] = annotations
    # Flatten annotations across annotators
    with_annotations = []
    for i, interactions_for_item in enumerate(item_interactions):
        test_item_annotations = {
            key: annotations_per_annotator[key][i] for key, _ in keyed_annotators
        }
        with_annotations.append(
            TestItemAnnotations(
                test_item=interactions_for_item.test_item,
                interactions=interactions_for_item.interactions,
                annotations=test_item_annotations,
            )
        )

    measured_test_items = []
    test_item_records = []
    for annotated in with_annotations:
        measurements = test.measure_quality(annotated)
        test_item_records.append(
            TestItemRecord(
                test_item=annotated.test_item,
                interactions=annotated.interactions,
                annotations=annotated.annotations,
                measurements=measurements,
            )
        )
        measured_test_items.append(
            MeasuredTestItem(test_item=annotated.test_item, measurements=measurements)
        )
    results = test.aggregate_measurements(measured_test_items)
    return TestRecord(
        test_name=test_name,
        test_initialization=test_initialization,
        dependency_versions=dependency_helper.versions_used(),
        sut_name=sut_name,
        sut_initialization=sut_initialization,
        test_item_records=test_item_records,
        results=results,
    )


def _collect_sut_responses(
    test_name: str,
    test_items: List[TestItem],
    sut_name: str,
    sut: PromptResponseSUT,
    cache: BaseCache,
):
    item_interactions: List[TestItemInteractions] = []
    desc = f"Collecting responses to {test_name} from {sut_name}"
    for item in tqdm(test_items, desc=desc):
        interactions = []
        for prompt in item.prompts:
            if isinstance(prompt.prompt, TextPrompt):
                sut_request = sut.translate_text_prompt(prompt.prompt)
            else:
                sut_request = sut.translate_chat_prompt(prompt.prompt)
            sut_response = cache.get_or_call(sut_request, sut.evaluate)
            response = sut.translate_response(sut_request, sut_response)
            interactions.append(PromptInteraction(prompt=prompt, response=response))
        item_interactions.append(
            TestItemInteractions(interactions=interactions, test_item=item)
        )
    return item_interactions


class AnnotateTestItemRequest(BaseModel):
    """Wrapper to make annotate_test_item's request cacheable."""

    interactions: List[PromptInteraction]


def _collect_annotations(
    key: str,
    annotator: BaseAnnotator,
    item_interactions: List[TestItemInteractions],
    cache: BaseCache,
) -> List[Annotation]:
    annotations: List[Annotation] = []
    desc = f"Collection annotations from {key}"
    for interactions_for_item in tqdm(item_interactions, desc=desc):
        request = AnnotateTestItemRequest(
            interactions=interactions_for_item.interactions
        )

        def _do_annotation(interaction_list: AnnotateTestItemRequest):
            return annotator.annotate_test_item(interaction_list.interactions)

        try:
            annotation = cache.get_or_call(request, _do_annotation)
        except Exception as e:
            raise Exception(f"Exception while handling: {interactions_for_item}") from e
        annotations.append(Annotation.from_instance(annotation))
    return annotations
