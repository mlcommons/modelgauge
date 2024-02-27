import os
import random
from typing import Dict, List, Mapping, Optional
from tqdm import tqdm
from newhelm.annotation import Annotation
from newhelm.base_test import BasePromptResponseTest
from newhelm.caching import SqlDictCache
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.prompt import TextPrompt
from newhelm.record_init import get_initialization_record
from newhelm.records import TestItemRecord, TestRecord
from newhelm.secret_values import RawSecrets, SecretValues
from newhelm.single_turn_prompt_response import (
    TestItem,
    TestItemAnnotations,
    MeasuredTestItem,
    PromptInteraction,
    TestItemInteractions,
)
from newhelm.sut import PromptResponseSUT


def assert_secrets_available(
    test: BasePromptResponseTest,
    sut: PromptResponseSUT,
    raw_secrets: RawSecrets,
):
    used_secrets = list(sut.get_used_secrets())
    for annotator in test.get_annotators().values():
        used_secrets.extend(annotator.get_used_secrets())
    SecretValues(used_secrets, raw_secrets)


def run_prompt_response_test(
    test_name: str,
    test: BasePromptResponseTest,
    sut_name: str,
    sut: PromptResponseSUT,
    data_dir: str,
    raw_secrets: RawSecrets,
    max_test_items: Optional[int] = None,
    use_caching: Optional[bool] = True,
) -> TestRecord:
    """Demonstration for how to run a single Test on a single SUT, all calls serial."""

    # Ensure we can record what these objects are
    test_initialization = get_initialization_record(test)
    sut_initialization = get_initialization_record(sut)
    # Ensure we have all the required secrets
    assert_secrets_available(test, sut, raw_secrets)
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
    if use_caching:
        with SqlDictCache(
            os.path.join(test_data_path, "cached_responses"), sut_name
        ) as cache:
            item_interactions = _collect_sut_responses(
                test_name, test_items, sut_name, sut, raw_secrets, cache
            )
    else:
        item_interactions = _collect_sut_responses(
            test_name, test_items, sut_name, sut, raw_secrets, cache=None
        )
    annotations_per_annotator: Dict[str, List[Annotation]] = {}
    keyed_annotators = test.get_annotators().items()
    for key, annotator in keyed_annotators:
        annotator_secrets = SecretValues(annotator.get_used_secrets(), raw_secrets)
        annotator.load(annotator_secrets)
        annotations: List[Annotation] = []
        desc = f"Collection annotations from {key}"
        for interactions_for_item in tqdm(item_interactions, desc=desc):
            try:
                annotation = annotator.annotate_test_item(
                    interactions_for_item.interactions
                )
            except Exception as e:
                raise Exception(
                    f"Exception while handling: {interactions_for_item}"
                ) from e
            annotations.append(Annotation.from_instance(annotation))
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
    raw_secrets: RawSecrets,
    cache: Optional[SqlDictCache],
):
    sut_secrets = SecretValues(sut.get_used_secrets(), raw_secrets)
    sut.load(sut_secrets)
    item_interactions: List[TestItemInteractions] = []
    desc = f"Collecting responses to {test_name} from {sut_name}"
    for item in tqdm(test_items, desc=desc):
        interactions = []
        for prompt in item.prompts:
            if isinstance(prompt.prompt, TextPrompt):
                sut_request = sut.translate_text_prompt(prompt.prompt)
            else:
                sut_request = sut.translate_chat_prompt(prompt.prompt)
            sut_response = None
            if cache is not None:
                sut_response = cache.get_or_call(sut_request, sut.evaluate)
            else:
                sut_response = sut.evaluate(sut_request)
            response = sut.translate_response(sut_request, sut_response)
            interactions.append(PromptInteraction(prompt=prompt, response=response))
        item_interactions.append(
            TestItemInteractions(interactions=interactions, test_item=item)
        )
    return item_interactions
