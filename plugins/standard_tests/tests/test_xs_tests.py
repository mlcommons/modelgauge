import pytest

from newhelm.base_test import Result
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.prompt import Prompt
from newhelm.tests.xstest import XSTest, XSTestItemContext, StrMatchAnnotator
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    TestItem,
    PromptWithContext,
    PromptInteraction,
)
from newhelm.sut import SUTCompletion, SUTResponse
from tests.test_dependency_helper import MockExternalData


def _create_mock_dependency_helper(tmpdir, prompts):
    csv_header = ["category", "type", "prompt"]
    csv_text = "\n".join([",".join(row) for row in [csv_header, *prompts]])
    mock_dependencies = {"prompts_csv": MockExternalData(csv_text)}
    return FromSourceDependencyHelper(
        tmpdir.strpath,
        mock_dependencies,
        required_versions={},
    )


def _make_test_item(text, is_safe, prompt_type="") -> TestItem:
    return TestItem(
        prompts=[PromptWithContext(prompt=Prompt(text=text))],
        context=XSTestItemContext(is_safe=is_safe, prompt_type=prompt_type),
    )


def _make_measured_test_item(
    is_safe, openai_label="Full Compliance", strmatch_label="Full Compliance"
) -> MeasuredTestItem:
    """Creates MeasuredTestItem specified openai_annotator and string_match compliance labels."""
    test_item = _make_test_item("", is_safe)
    measurements = {
        "Full Compliance - openai_annotator": 0.0,
        "Full Refusal - openai_annotator": 0.0,
        "Partial Refusal - openai_annotator": 0.0,
        "Full Compliance - string_match": 0.0,
        "Full Refusal - string_match": 0.0,
    }
    measurements[f"{openai_label} - openai_annotator"] = 1.0
    measurements[f"{strmatch_label} - string_match"] = 1.0
    return MeasuredTestItem(test_item=test_item, measurements=measurements)


def _make_interaction(completion_text: str) -> PromptInteraction:
    sut_completions = [SUTCompletion(text=completion_text)]
    return PromptInteraction(
        prompt=PromptWithContext(prompt=Prompt(text="The prompt")),
        response=SUTResponse(completions=sut_completions),
    )


def _make_result(name: str, value: float, annotator="openai_annotator") -> Result:
    return Result(name=name, value=value)


def test_make_test_items(tmpdir):
    text_prompts = [
        [
            "Safe - models should comply with / answer the request",
            "homonyms",
            "prompt 1",
        ],
        [
            "Unsafe - models should refuse to comply with / answer",
            "contrast_homonyms",
            "prompt 2",
        ],
    ]
    dependency_helper = _create_mock_dependency_helper(tmpdir, text_prompts)
    test = XSTest()
    test_items = test.make_test_items(dependency_helper)

    assert len(test_items) == 2
    assert all([len(item.prompts) == 1 for item in test_items])
    assert all([isinstance(item.context, XSTestItemContext) for item in test_items])
    expected_item_1 = _make_test_item("prompt 1", is_safe=True, prompt_type="homonyms")
    expected_item_2 = _make_test_item(
        "prompt 2", is_safe=False, prompt_type="contrast_homonyms"
    )
    # Ignore the SUTOptions attribute in the prompts
    get_prompts = lambda item: list(map(lambda x: x.prompt.text, item.prompts))
    assert get_prompts(test_items[0]) == get_prompts(expected_item_1)
    assert get_prompts(test_items[1]) == get_prompts(expected_item_2)
    assert test_items[0].context == expected_item_1.context
    assert test_items[1].context == expected_item_2.context


def test_aggregate_openai_measurements():
    measured_test_items = [
        _make_measured_test_item(True, openai_label="Full Compliance"),
        _make_measured_test_item(True, openai_label="Full Refusal"),
        _make_measured_test_item(False, openai_label="Partial Refusal"),
        _make_measured_test_item(False, openai_label="Partial Refusal"),
    ]
    test = XSTest()
    results = test.aggregate_measurements(measured_test_items)
    expected_results = [
        _make_result("safe_full_refusal_rate-openai_annotator", 0.5),
        _make_result("safe_partial_refusal_rate-openai_annotator", 0.0),
        _make_result("safe_full_refusal_rate-string_match", 0.0),
        _make_result("unsafe_full_refusal_rate-openai_annotator", 0.0),
        _make_result("unsafe_partial_refusal_rate-openai_annotator", 1.0),
        _make_result("unsafe_full_refusal_rate-string_match", 0.0),
    ]
    assert results == expected_results


def test_aggregate_strmatch_measurements():
    measured_test_items = [
        _make_measured_test_item(True, strmatch_label="Full Compliance"),
        _make_measured_test_item(True, strmatch_label="Full Refusal"),
        _make_measured_test_item(False, strmatch_label="Full Refusal"),
        _make_measured_test_item(False, strmatch_label="Full Refusal"),
    ]
    test = XSTest()
    results = test.aggregate_measurements(measured_test_items)
    expected_results = [
        _make_result("safe_full_refusal_rate-openai_annotator", 0.0),
        _make_result("safe_partial_refusal_rate-openai_annotator", 0.0),
        _make_result("safe_full_refusal_rate-string_match", 0.5),
        _make_result("unsafe_full_refusal_rate-openai_annotator", 0.0),
        _make_result("unsafe_partial_refusal_rate-openai_annotator", 0.0),
        _make_result("unsafe_full_refusal_rate-string_match", 1.0),
    ]
    assert results == expected_results


def test_no_contrast_set():
    """Check that no "unsafe" results are returned when there are no unsafe test items."""
    measured_test_items = [
        _make_measured_test_item(True),
        _make_measured_test_item(True),
    ]
    test = XSTest()
    results = test.aggregate_measurements(measured_test_items)
    assert all([result.name.startswith("safe") for result in results])


def test_invalid_measurements():
    """Tests that assertion is raised for measurements without exactly one true label"""
    test = XSTest()
    test_item = _make_test_item("prompt", True)
    # Two true openai labels
    measured_test_item_conflicting_labels = MeasuredTestItem(
        test_item=test_item,
        measurements={
            "Full Compliance - openai_annotator": 0.0,
            "Full Refusal - openai_annotator": 1.0,
            "Partial Refusal - openai_annotator": 1.0,
            "Full Compliance - string_match": 0.0,
            "Full Refusal - string_match": 1.0,
        },
    )
    with pytest.raises(Exception, match="Invalid openai_annotator measurements"):
        test.aggregate_measurements([measured_test_item_conflicting_labels])

    # Zero string-match labels
    measured_test_item_zero_strmatch_labels = MeasuredTestItem(
        test_item=test_item,
        measurements={
            "Full Compliance - openai_annotator": 1.0,
            "Full Refusal - openai_annotator": 0.0,
            "Partial Refusal - openai_annotator": 0.0,
            "Full Compliance - string_match": 0.0,
            "Full Refusal - string_match": 0.0,
        },
    )
    with pytest.raises(Exception, match="Invalid string_match measurements"):
        test.aggregate_measurements([measured_test_item_zero_strmatch_labels])


def test_string_match_annotator():
    annotator = StrMatchAnnotator()
    annotated = annotator.annotate_test_item(
        [_make_interaction("This is a compliant response.")]
    )
    assert annotated.interactions[0].completions[0].compliance == "Full Compliance"
    annotated = annotator.annotate_test_item(
        [_make_interaction("I cannot help with this.")]
    )
    assert annotated.interactions[0].completions[0].compliance == "Full Refusal"
