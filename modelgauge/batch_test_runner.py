import os
import random
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
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_capabilities_verification import assert_sut_capabilities
from modelgauge.sut_decorator import assert_is_sut
from modelgauge.test_decorator import assert_is_test
from tqdm import tqdm
from typing import Generic, List, Optional

from pydantic import BaseModel
from modelgauge.sut import RequestType, ResponseType


class Job(BaseModel, Generic[RequestType, ResponseType]):
    id: str
    request: RequestType
    result: Optional[ResponseType] = None


def run_batch_prompt_response_test(
    test: PromptResponseTest,
    sut: PromptResponseSUT,
    data_dir: str,
    max_test_items: Optional[int] = None,
    use_caching: bool = True,
    disable_progress_bar: bool = False,
    max_batch_size: int = 20,  # 20 works well as a default against together API rate limits
) -> TestRecord:
    """Runner to execute SUT tests in batch. Annotation and measurement are still called in serial (for now)"""

    assert_is_test(test)
    assert_is_sut(sut)
    assert_sut_capabilities(sut, test)

    # Ensure we can record what these objects are
    test_initialization = test.initialization_record
    sut_initialization = sut.initialization_record
    test_data_path = os.path.join(data_dir, "tests", test.__class__.__name__)

    # TODO leverage the cache for SUT responses
    # sut_cache: Cache
    # if use_caching:
    #     sut_cache = SqlDictCache(os.path.join(data_dir, "suts"), sut.uid)
    # else:
    #     sut_cache = NoCache()

    annotators = []
    for key, annotator in test.get_annotators().items():
        annotator_cache: Cache
        if use_caching:
            annotator_cache = SqlDictCache(
                os.path.join(test_data_path, "annotators"), key
            )
        else:
            annotator_cache = NoCache()
        assert isinstance(
            annotator, CompletionAnnotator
        ), "Only know how to do CompletionAnnotator."
        annotators.append(AnnotatorData(key, annotator, annotator_cache))

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
    desc = f"Processing TestItems for test={test.uid} sut={sut.uid}"

    # Build SUT response joblist by flattening test item prompts into list of individual jobs
    # TODO validate test item ids are all unique
    # TODO validate that all sut request types are the same
    jobs: List[Job] = []
    for test_item in tqdm(test_items, desc=desc, disable=disable_progress_bar):
        assert (
            test_item.id is not None
        ), "For batch jobs, all test items must have an id."

        for index, prompt in enumerate(test_item.prompts):
            job_id = build_job_id(test_item, index)

            if isinstance(prompt.prompt, TextPrompt):
                sut_request = sut.translate_text_prompt(prompt.prompt)
            else:
                sut_request = sut.translate_chat_prompt(prompt.prompt)

            jobs.append(
                Job(
                    id=job_id,
                    request=sut_request,
                )
            )

    job_requests = [job.request for job in jobs]

    # Compute SUT responses in batches
    batch_sut_responses = []
    responses = []
    max_batch_size = len(jobs) if len(jobs) < max_batch_size else max_batch_size
    for i in range(0, len(jobs), max_batch_size):
        batch_sut_responses = sut.batch_evaluate(job_requests[i : i + max_batch_size])
        for response in batch_sut_responses:
            responses.append(response)

    # Translate all the responses and build a lookup
    response_lookup = {}
    for i in range(len(jobs)):
        response = sut.translate_response(job_requests[i], responses[i])
        response_lookup[jobs[i].id] = response

    # Compute annotations and measurements using single item processing (for now)
    for item in test_items:
        interactions: List[PromptInteractionAnnotations] = []
        for index, prompt in enumerate(item.prompts):
            response = response_lookup[build_job_id(item, index)]

            annotated_completions: List[SUTCompletionAnnotations] = []
            for completion in response.completions:
                annotations = {}
                for annotator_data in annotators:
                    annotator = annotator_data.annotator
                    try:
                        annotator_request = annotator.translate_request(
                            prompt, completion
                        )
                        with annotator_data.cache as cache:
                            annotator_response = cache.get_or_call(
                                annotator_request, annotator.annotate
                            )
                        annotation = annotator.translate_response(
                            annotator_request, annotator_response
                        )
                    except Exception as e:
                        raise Exception(
                            f"Exception while handling annotation for {annotator_data.key} on {response}"
                        ) from e

                    annotations[annotator_data.key] = Annotation.from_instance(
                        annotation
                    )
                annotated_completions.append(
                    SUTCompletionAnnotations(
                        completion=completion, annotations=annotations
                    )
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

        # Create and store the record + measurement
        test_item_record = TestItemRecord(
            test_item=annotated.test_item,
            interactions=annotated.interactions,
            measurements=measurements,
        )
        test_item_records.append(test_item_record)
        measured_test_items.append(
            MeasuredTestItem(
                test_item=test_item_record.test_item,
                measurements=test_item_record.measurements,
            )
        )

    # Aggregate measurements into test result
    test_result = TestResult.from_instance(
        test.aggregate_measurements(measured_test_items)
    )

    return TestRecord(
        test_uid=test.uid,
        test_initialization=test_initialization,
        dependency_versions=dependency_helper.versions_used(),
        sut_uid=sut.uid,
        sut_initialization=sut_initialization,
        test_item_records=test_item_records,
        result=test_result,
    )


class AnnotatorData:
    """Container to hold data about an annotator."""

    def __init__(self, key: str, annotator: CompletionAnnotator, cache: Cache):
        self.key = key
        self.annotator = annotator
        self.cache = cache


def build_job_id(test_item: TestItem, prompt_index: int) -> str:
    """Given a test item, and index of a prompt, build a unique job id. Used to tie batched SUT responses to original test items"""
    return f"{test_item.id}-{prompt_index}"
