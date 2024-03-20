import os
from typing import Sequence
import pytest
from newhelm.base_test import BasePromptResponseTest
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.load_plugins import load_plugins
from newhelm.record_init import InitializationRecord
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS
from newhelm.test_specifications import (
    TestSpecification,
    load_test_specification_files,
    register_test_from_specifications,
)
from tests.fake_secrets import fake_all_secrets
from tests.utilities import expensive_tests

# Ensure all the plugins are available during testing.
load_plugins()
_SPECS = load_test_specification_files()
register_test_from_specifications(list(_SPECS.values()), TESTS)
_FAKE_SECRETS = fake_all_secrets()


@pytest.mark.parametrize("test_name", [key for key, _ in TESTS.items()])
def test_all_tests_construct_and_record_init(test_name):
    test = TESTS.make_instance(test_name, secrets=_FAKE_SECRETS)
    assert hasattr(
        test, "initialization_record"
    ), "Test is probably missing @newhelm_test() decorator."
    assert isinstance(test.initialization_record, InitializationRecord)


def _assert_some_contain(values: Sequence[str], target: str):
    if not any(target in value for value in values):
        raise AssertionError(
            f"Expected '{target}' to be part of at least one of the following: {values}."
        )


def test_plugin_test_specifications():
    specifications = load_test_specification_files()
    paths = [spec.source for spec in specifications.values()]
    # Check that it loaded some plugin directories.
    _assert_some_contain(paths, "newhelm/demo_plugin/newhelm/tests/specifications")
    _assert_some_contain(paths, "newhelm/plugins/standard_tests")
    # Check for a specific specification
    assert "demo_04" in specifications
    for uid, spec in specifications.items():
        assert isinstance(
            spec, TestSpecification
        ), f"Expected {uid} to load a TestSpecification."
        assert spec.identity.uid == uid


# Some tests require such large downloads / complex processing
# that we don't want to do that even on expensive_tests.
# If your Test is timing out, consider adding it here.
TOO_SLOW = {
    "real_toxicity_prompts",
    "bbq",
}


@expensive_tests
@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "test_name", [key for key, _ in TESTS.items() if key not in TOO_SLOW]
)
def test_all_tests_make_test_items(test_name, tmpdir):
    test = TESTS.make_instance(test_name, secrets=_FAKE_SECRETS)
    if isinstance(test, BasePromptResponseTest):
        test_data_path = os.path.join(tmpdir, test.get_metadata().name)
        dependency_helper = FromSourceDependencyHelper(
            test_data_path,
            test.get_dependencies(),
            required_versions={},
        )

        test_items = test.make_test_items(dependency_helper)
        assert len(test_items) > 0


@pytest.mark.parametrize("sut_name", [key for key, _ in SUTS.items()])
def test_all_suts_construct_and_record_init(sut_name):
    sut = SUTS.make_instance(sut_name, secrets=_FAKE_SECRETS)
    assert hasattr(
        sut, "initialization_record"
    ), "SUT is probably missing @newhelm_sut() decorator."
    assert isinstance(sut.initialization_record, InitializationRecord)
