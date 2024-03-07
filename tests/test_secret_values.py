from newhelm.general import get_class
from newhelm.secret_values import (
    RequiredSecret,
    SecretDescription,
    SerializedSecret,
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
        scope="some-scope", key="some-key", instructions="some-instructions"
    )
    matching = [s for s in descriptions if s == test_secret]
    assert len(matching) == 1, f"Found secrets: {descriptions}"


def test_serialize_secret():
    original = MySecret("some-value")
    serialized = SerializedSecret.serialize(original)
    assert serialized == SerializedSecret(
        module="test_secret_values", qual_name="MySecret"
    )
    returned = get_class(serialized.module, serialized.qual_name)
    assert returned.description() == SecretDescription(
        scope="some-scope", key="some-key", instructions="some-instructions"
    )
