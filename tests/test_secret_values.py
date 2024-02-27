import pytest
from newhelm.secret_values import MissingSecretValue, SecretValues, UseSecret

_REQUIRED_SECRET = UseSecret(
    scope="some-scope",
    key="some-key",
    required=True,
    instructions="some-instructions",
)

_OPTIONAL_SECRET = UseSecret(
    scope="optional-scope",
    key="optional-key",
    required=False,
    instructions="optional-instructions",
)


def test_get_required_has_value():
    secrets = SecretValues(
        used_secrets=[_REQUIRED_SECRET],
        known_secrets={"some-scope": {"some-key": "some-value"}},
    )
    assert secrets.get_required("some-scope", "some-key") == "some-value"


def test_init_missing_required():
    with pytest.raises(MissingSecretValue) as err_info:
        SecretValues(used_secrets=[_REQUIRED_SECRET], known_secrets={})
    err_text = str(err_info.value)
    assert err_text == (
        "Missing the following secrets:\n"
        "scope=`some-scope` key=`some-key` can be obtained by: some-instructions\n"
        "There are currently no known secrets."
    )


def test_get_required_not_listed():
    secrets = SecretValues(
        used_secrets=[_REQUIRED_SECRET],
        known_secrets={"some-scope": {"some-key": "some-value"}},
    )
    with pytest.raises(AssertionError) as err_info:
        secrets.get_required("some-scope", "another-key")
    err_text = str(err_info.value)
    assert err_text == (
        "Secret some-scope.another-key must be listed as "
        "used before it can be retrieved."
    )


def test_get_required_listed_optional():
    secrets = SecretValues(
        used_secrets=[_OPTIONAL_SECRET],
        known_secrets={},
    )
    with pytest.raises(AssertionError) as err_info:
        secrets.get_required("optional-scope", "optional-key")
    err_text = str(err_info.value)
    assert err_text == (
        "The secrets optional-scope.optional-key needs to be "
        "listed as required in order to use `get_required`."
    )


def test_get_optional_has_value():
    secrets = SecretValues(
        used_secrets=[_OPTIONAL_SECRET],
        known_secrets={"optional-scope": {"optional-key": "optional-value"}},
    )
    assert secrets.get_optional("optional-scope", "optional-key") == "optional-value"


def test_get_optional_missing_value():
    secrets = SecretValues(
        used_secrets=[_OPTIONAL_SECRET],
        known_secrets={},
    )
    assert secrets.get_optional("optional-scope", "optional-key") is None


def test_get_optional_not_listed():
    secrets = SecretValues(
        used_secrets=[_OPTIONAL_SECRET],
        known_secrets={"optional-scope": {"optional-key": "optional-value"}},
    )
    with pytest.raises(AssertionError) as err_info:
        secrets.get_optional("optional-scope", "another-key")
    err_text = str(err_info.value)
    assert err_text == (
        "Secret optional-scope.another-key must be listed as "
        "used before it can be retrieved."
    )


def test_get_optional_listed_required():
    secrets = SecretValues(
        used_secrets=[_REQUIRED_SECRET],
        known_secrets={"some-scope": {"some-key": "some-value"}},
    )
    # This is allowed
    assert secrets.get_optional("some-scope", "some-key") == "some-value"


def test_get_required_listed_required_and_optional():
    as_optional = UseSecret(
        scope=_REQUIRED_SECRET.scope,
        key=_REQUIRED_SECRET.key,
        required=False,
        instructions="optional-instructions",
    )
    secrets = SecretValues(
        used_secrets=[_REQUIRED_SECRET, as_optional],
        known_secrets={"some-scope": {"some-key": "some-value"}},
    )

    assert secrets.get_required("some-scope", "some-key") == "some-value"
