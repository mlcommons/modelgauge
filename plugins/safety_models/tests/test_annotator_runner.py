import pytest

from modelgauge.annotation import Annotation
from modelgauge.annotator_test_runner import run_annotator_test
from modelgauge.interaction_annotation import AnnotatorInteractionRecord
from tests.fake_annotator import FakeAnnotator
from tests.fake_sut import FakeSUT
from fake_annotator_test import ( # type: ignore
    FakeAnnotatorTest,
    FakeTestResult,
    fake_annotation_test_item,
)


def test_run_annotation_test_output(tmpdir):
    item_1 = fake_annotation_test_item("1", "a")
    item_2 = fake_annotation_test_item("2", "b")
    fake_measurement = {"some-measurement": 0.5}
    record = run_annotator_test(
        FakeAnnotatorTest(
            test_items=[item_1, item_2],
            measurement=fake_measurement,
        ),
        FakeAnnotator(),
        tmpdir,
    )

    assert record.interaction_records == [
        AnnotatorInteractionRecord(
            test_item=item_1,
            annotation=Annotation(
                module="tests.fake_annotator",
                class_name="FakeAnnotation",
                data={"sut_text": "a"},
            ),
            measurements=fake_measurement,
        ),
        AnnotatorInteractionRecord(
            test_item=item_2,
            annotation=Annotation(
                module="tests.fake_annotator",
                class_name="FakeAnnotation",
                data={"sut_text": "b"},
            ),
            measurements=fake_measurement,
        ),
    ]
    assert record.result.to_instance() == FakeTestResult(count_test_items=2.0)


class NotATestOrAnnotator:
    pass


def test_run_annotator_test_invalid_test(tmpdir):
    with pytest.raises(AssertionError) as err_info:
        run_annotator_test(
            NotATestOrAnnotator(),
            FakeAnnotator(),
            tmpdir,
        )
    assert (
        str(err_info.value)
        == "NotATestOrAnnotator should be decorated with @modelgauge_test."
    )


def test_run_prompt_response_test_invalid_annotator(tmpdir):
    with pytest.raises(AssertionError) as err_info:
        run_annotator_test(
            FakeAnnotatorTest(),
            NotATestOrAnnotator(),
            tmpdir,
        )
    assert str(err_info.value) == "Only know how to do CompletionAnnotator."
