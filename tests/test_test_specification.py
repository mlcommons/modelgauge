from typing import Any, Tuple

import pytest
from newhelm.base_test import BaseTest, TestMetadata
from newhelm.instance_factory import InstanceFactory
from newhelm.secret_values import RequiredSecret, SecretDescription
from newhelm.test_decorator import newhelm_test
from newhelm.test_specifications import (
    Definition,
    Identity,
    TestSpecification,
    load_test_specification_files,
    register_test_from_specifications,
)


def _make_spec(source, uid) -> TestSpecification:
    return TestSpecification(
        source=source, identity=Identity(uid=uid, display_name="some display")
    )


def _mock_file(spec: TestSpecification) -> Tuple[str, dict[str, Any]]:
    raw = spec.model_dump()
    source = raw.pop("source")
    return (source, raw)


def test_load_basic():
    expected = _make_spec("some/path", "uid1")

    results = load_test_specification_files([_mock_file(expected)])
    assert results == {"uid1": expected}


def test_load_bad_value():
    expected = _make_spec("some/path", "uid1")
    mocked = _mock_file(expected)
    # Make identity wrong
    mocked[1]["identity"] = "some-value"

    with pytest.raises(AssertionError) as err_info:
        load_test_specification_files([mocked])
    assert str(err_info.value) == "Could not parse some/path into TestSpecification."
    # Ensure it forwards the validation issue.
    assert "valid dictionary or instance of Identity" in str(err_info.value.__cause__)


def test_load_should_not_include_source():
    expected = _make_spec("some/path", "uid1")
    mocked = _mock_file(expected)
    # Make identity wrong
    mocked[1]["source"] = "wrong/path"

    with pytest.raises(AssertionError) as err_info:
        load_test_specification_files([mocked])
    assert (
        str(err_info.value) == "File some/path should not include the "
        "`source` variable as that changes during packaging."
    )


def test_load_multiple():
    expected1 = _make_spec("p1", "uid1")
    expected2 = _make_spec("p2", "uid2")
    expected3 = _make_spec("p3", "uid3")

    results = load_test_specification_files(
        [
            _mock_file(expected1),
            _mock_file(expected2),
            _mock_file(expected3),
        ]
    )
    assert results == {
        "uid1": expected1,
        "uid2": expected2,
        "uid3": expected3,
    }


def test_load_repeated_uid():
    expected1 = _make_spec("p1", "uid1")
    expected2 = _make_spec("p2", "uid2")
    expected1_again = _make_spec("p3", "uid1")

    with pytest.raises(AssertionError) as err_info:
        load_test_specification_files(
            [
                _mock_file(expected1),
                _mock_file(expected2),
                _mock_file(expected1_again),
            ]
        )
    assert str(err_info.value) == (
        "Expected UID to be unique across files, " "but p1 and p3 both have uid=uid1."
    )


def test_load_module_no_error():
    # We don't know what files might exist, so just verify it runs.
    load_test_specification_files()


@newhelm_test()
class SomeRegisteredTest(BaseTest):
    def get_metadata(self) -> TestMetadata:
        raise NotImplementedError()


def test_register_test_from_specifications():
    registry = InstanceFactory[BaseTest]()
    spec = TestSpecification(
        source="some-source",
        identity=Identity(uid="some-test", display_name="Some Test"),
        definition=Definition(class_name="SomeRegisteredTest"),
    )
    register_test_from_specifications([spec], registry)
    result = registry.make_instance("some-test", secrets={})

    assert isinstance(result, SomeRegisteredTest)


@newhelm_test()
class SomeRegisteredTestWithComplexArgs(BaseTest):
    def __init__(self, uid, arg1, the_secret, arg2):
        self.uid = uid
        self.arg1 = arg1
        self.the_secret_value = the_secret.value
        self.arg2 = arg2

    def get_metadata(self) -> TestMetadata:
        raise NotImplementedError()


class SecretForRegistry(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="some-scope", key="some-key", instructions="some-instructions."
        )


def test_register_test_from_specifications_complex_args():
    registry = InstanceFactory[BaseTest]()
    spec = TestSpecification(
        source="some-source",
        identity=Identity(uid="some-test", display_name="Some Test"),
        definition=Definition(
            class_name="SomeRegisteredTestWithComplexArgs",
            keyword_arguments={"arg1": "v1", "arg2": 2},
            secrets={"the_secret": "SecretForRegistry"},
        ),
    )
    register_test_from_specifications([spec], registry)
    result = registry.make_instance(
        "some-test", secrets={"some-scope": {"some-key": "1234"}}
    )

    assert isinstance(result, SomeRegisteredTestWithComplexArgs)
    assert result.uid == "some-test"
    assert result.arg1 == "v1"
    assert result.the_secret_value == "1234"
    assert result.arg2 == 2
