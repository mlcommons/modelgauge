import pytest

from modelgauge.annotation import Annotation
from modelgauge.batch_test_runner import run_batch_prompt_response_test
from modelgauge.records import TestItemRecord
from modelgauge.single_turn_prompt_response import (
    PromptInteractionAnnotations,
    SUTCompletionAnnotations,
    SUTResponseAnnotations,
)
from modelgauge.sut import SUTCompletion
from modelgauge.sut_capabilities import ProducesPerTokenLogProbabilities
from modelgauge.test_decorator import modelgauge_test
from tests.fake_annotator import FakeAnnotator
from tests.fake_sut import FakeSUT
from tests.fake_test import FakeTest, FakeTestResult, fake_test_item


def test_run_prompt_response_test_output(tmpdir):
    item_1 = fake_test_item("1", id="1")
    item_2 = fake_test_item("2", id="2")
    fake_measurement = {"some-measurement": 0.5}
    record = run_batch_prompt_response_test(
        FakeTest(
            test_items=[item_1, item_2],
            annotators={"some-annotator": FakeAnnotator()},
            measurement=fake_measurement,
        ),
        FakeSUT(),
        tmpdir,
    )

    assert record.test_item_records == [
        TestItemRecord(
            test_item=item_1,
            interactions=[
                PromptInteractionAnnotations(
                    prompt=item_1.prompts[0],
                    response=SUTResponseAnnotations(
                        completions=[
                            SUTCompletionAnnotations(
                                completion=SUTCompletion(text="1"),
                                annotations={
                                    "some-annotator": Annotation(
                                        module="tests.fake_annotator",
                                        class_name="FakeAnnotation",
                                        data={"sut_text": "1"},
                                    )
                                },
                            )
                        ]
                    ),
                )
            ],
            measurements=fake_measurement,
        ),
        TestItemRecord(
            test_item=item_2,
            interactions=[
                PromptInteractionAnnotations(
                    prompt=item_2.prompts[0],
                    response=SUTResponseAnnotations(
                        completions=[
                            SUTCompletionAnnotations(
                                completion=SUTCompletion(text="2"),
                                annotations={
                                    "some-annotator": Annotation(
                                        module="tests.fake_annotator",
                                        class_name="FakeAnnotation",
                                        data={"sut_text": "2"},
                                    )
                                },
                            )
                        ]
                    ),
                )
            ],
            measurements=fake_measurement,
        ),
    ]
    assert record.result.to_instance() == FakeTestResult(count_test_items=2.0)


def fake_run(max_test_items, tmpdir):
    # Lots of test items
    test_items = [fake_test_item(text=str(i), id=str(i)) for i in range(100)]
    fake_measurement = {"some-measurement": 0.5}
    record = run_batch_prompt_response_test(
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": FakeAnnotator()},
            measurement=fake_measurement,
        ),
        FakeSUT(),
        tmpdir,
        # Limit to just 3 test items
        max_test_items=max_test_items,
    )
    return record


def test_run_prompt_response_test_max_test_items(tmpdir):
    max_test_items = 3
    record = fake_run(max_test_items, tmpdir)
    assert len(record.test_item_records) == max_test_items
    assert record.result.to_instance() == FakeTestResult(count_test_items=3.0)


def test_run_prompt_response_test_batched(tmpdir):
    max_test_items = 10000  # Some arbitrary large number
    expected_test_items = 100
    record = fake_run(max_test_items, tmpdir)
    assert len(record.test_item_records) == expected_test_items
    assert record.result.to_instance() == FakeTestResult(
        count_test_items=expected_test_items
    )  # fake_run uses 100 test items


def test_run_prompt_response_test_max_test_items_stable(tmpdir):
    run3 = fake_run(3, tmpdir)
    run4 = fake_run(4, tmpdir)
    prompts3 = [r.test_item.prompts[0].prompt.text for r in run3.test_item_records]
    prompts4 = [r.test_item.prompts[0].prompt.text for r in run4.test_item_records]
    assert len(prompts3) == 3
    assert len(prompts4) == 4

    for p in prompts3:
        assert p in prompts4
