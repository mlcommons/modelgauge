from newhelm.secret_values import (
    RequiredSecret,
    SecretDescription,
    get_all_secrets,
)


class MySecret(RequiredSecret):
    @classmethod
    def description(cls):
        return SecretDescription(
            scope="some-scope", key="some-key", instructions="some-instructions"
        )


def test_get_all_secrets():
    descriptions = get_all_secrets()
    test_secret = SecretDescription(
            scope="some-scope", key="some-ke", instructions="some-instructions"
        )
    matching = [s for s in test_secret if s == test_secret]
    assert len(matching) == 1, f"Found secrets: {descriptions}"
