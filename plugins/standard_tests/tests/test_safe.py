import pytest

from newhelm.base_test import Result
from newhelm.prompt import TextPrompt
from newhelm.tests.safe import SAFE, SAFETestItemContext, HAZARDS_MAPPING
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    TestItem,
    PromptWithContext,
)
from tests.fake_dependency_helper import FakeDependencyHelper, make_csv

HAZARD = "cae"


def _fake_dependency_helper(tmpdir, rows):
    csv_header = ["Direct prompt", "Hazard type", "UID"]
    csv_text = make_csv(csv_header, rows)
    return FakeDependencyHelper(tmpdir, {"prompts_csv": csv_text})


def _make_test_item(text, hazard, prompt_uid="") -> TestItem:
    return TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text=text))],
        context=SAFETestItemContext(prompt_uid=prompt_uid, hazard=hazard),
    )


def test_make_test_items(tmpdir):
    hazard_name = HAZARDS_MAPPING[HAZARD]
    text_prompts = [
        ["prompt 1", hazard_name, "id1"],
        ["prompt 2", hazard_name, "id2"],
    ]
    test = SAFE(HAZARD, "")
    dependency_helper = _fake_dependency_helper(tmpdir, text_prompts)
    test_items = test.make_test_items(dependency_helper)

    assert len(test_items) == 2
    assert all([len(item.prompts) == 1 for item in test_items])
    expected_item_1 = _make_test_item("prompt 1", hazard_name, "id1")
    expected_item_2 = _make_test_item("prompt 2", hazard_name, "id2")
    # Ignore the SUTOptions attribute in the prompts
    get_prompts = lambda item: list(map(lambda x: x.prompt.text, item.prompts))
    assert get_prompts(test_items[0]) == get_prompts(expected_item_1)
    assert get_prompts(test_items[1]) == get_prompts(expected_item_2)
    assert test_items[0].context == expected_item_1.context
    assert test_items[1].context == expected_item_2.context


def test_different_hazards(tmpdir):
    """Checks that the context's hazard attribute is the only thing that differs among TestItems produced by different tests."""
    items = []
    for hazard in ["cae", "ter"]:
        test = SAFE(hazard, "")
        dependency_helper = _fake_dependency_helper(
            tmpdir, [["prompt", HAZARDS_MAPPING[hazard], "id"]]
        )
        items.append(test.make_test_items(dependency_helper)[0])
    # All prompts are the same
    assert all(item.prompts == items[0].prompts for item in items)
    # All contexts are the same, exlucding hazard
    assert all(
        {k: v for k, v in item.context_internal.data.items() if k != "hazard"}
        == {k: v for k, v in items[0].context_internal.data.items() if k != "hazard"}
        for item in items
    )
    # All hazards are unique
    assert [item.context.hazard for item in items] == [
        HAZARDS_MAPPING["cae"],
        HAZARDS_MAPPING["ter"],
    ]


def test_unknown_hazard_exception():
    with pytest.raises(Exception, match="Invalid hazard type"):
        SAFE("New Hazard", "")


def test_aggregate_measurements():
    dummy_item = _make_test_item("prompt", HAZARDS_MAPPING[HAZARD])
    # 2 safe, 1 unsafe
    measured_test_items = [
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 0.0}),
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 1.0}),
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 1.0}),
    ]
    test = SAFE(HAZARD, "")
    results = test.aggregate_measurements(measured_test_items)
    assert results == [Result(name="frac_safe", value=2.0 / 3.0)]
