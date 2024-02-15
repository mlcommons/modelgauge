import pytest

from newhelm.base_test import BasePromptResponseTest
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.load_plugins import load_plugins, list_plugins
from newhelm.test_registry import TESTS
from newhelm.single_turn_prompt_response import MeasuredTestItem
from newhelm.typed_data import TypedData
from tests.test_dependency_helper import MockExternalData


def _create_mock_dependency_helper(tmpdir):
    csv_data = [
        ["harm_area", "category", "prompts_final_English"],
        [
            '"Physical harm and Violence"',
            "Information and advice seeking",
            "mock prompt 1",
        ],
        ["Scams and fraud", "Instructions and actions", "mock prompt 2"],
    ]
    csv_text = "\n".join([",".join(row) for row in csv_data]) + "\n"
    mock_dependencies = {"prompts_csv": MockExternalData(csv_text)}
    return FromSourceDependencyHelper(
        tmpdir.strpath,
        mock_dependencies,
        required_versions={},
    )


def _get_mock_test_items(tmpdir):
    test = TESTS.make_instance("simple_safety_tests")
    dependency_helper = _create_mock_dependency_helper(tmpdir)
    return test.make_test_items(dependency_helper)


def test_from_instance_factory():
    load_plugins()
    test = TESTS.make_instance("simple_safety_tests")
    assert isinstance(test, BasePromptResponseTest)


def test_make_test_items(tmpdir):
    test_items = _get_mock_test_items(tmpdir)

    assert len(test_items) == 2
    assert all([len(item.prompts) == 1 for item in test_items])
    assert all([isinstance(item.context, TypedData) for item in test_items])
    assert all([isinstance(item.context, TypedData) for item in test_items])
    assert all([item.context.data["harm_area"] is not None for item in test_items])
    assert all(
        [item.context.data["prompt_category"] is not None for item in test_items]
    )


def test_aggregate_measurements(tmpdir):
    test_items = _get_mock_test_items(tmpdir)
    measured_test_items = [
        MeasuredTestItem(test_item=test_items[0], measurements={"is_safe": 0.0}),
        MeasuredTestItem(test_item=test_items[0], measurements={"is_safe": 1.0}),
    ]

    test = TESTS.make_instance("simple_safety_tests")
    results = test.aggregate_measurements(measured_test_items)
    result = list(filter(lambda r: r.name == "frac_safe", results))
    assert len(result) == 1  # Should only be 1 result with name frac_safe
    assert result[0].value == 0.5
