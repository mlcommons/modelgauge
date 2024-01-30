import pytest
from newhelm.secrets_registry import SecretsRegistry, SecretsRegistryMissingValue


def test_get_required_has_value():
    registry = SecretsRegistry()
    registry.register("some-scope", "some-key", "some-instructions")
    registry.set_values({"some-scope": {"some-key": "some-value"}})
    value = registry.get_required("some-scope", "some-key")
    assert value == "some-value"


def test_get_required_no_values():
    registry = SecretsRegistry()
    registry.register("some-scope", "some-key", "some-instructions")
    registry.set_values({})
    with pytest.raises(SecretsRegistryMissingValue) as err_info:
        registry.get_required("some-scope", "some-key")
    err_text = str(err_info.value)
    assert err_text == (
        "Missing value for secret `some-key` in scope `some-scope`. "
        "Known scopes: set(). Known keys in `some-scope`: set(). "
        "Instructions for obtaining that value: some-instructions"
    )


def test_get_required_scope_not_key():
    registry = SecretsRegistry()
    registry.register("some-scope", "some-key", "some-instructions")
    registry.set_values({"some-scope": {"different-key": "key-value"}})
    with pytest.raises(SecretsRegistryMissingValue) as err_info:
        registry.get_required("some-scope", "some-key")
    err_text = str(err_info.value)
    assert err_text == (
        "Missing value for secret `some-key` in scope `some-scope`. "
        "Known scopes: {'some-scope'}. Known keys in `some-scope`: "
        "{'different-key'}. Instructions for obtaining that value: "
        "some-instructions"
    )


# TODO Add a bunch more tests
