from dataclasses import dataclass
import pytest
from newhelm.instance_factory import FactoryEntry, InstanceFactory
from newhelm.secret_values import InjectSecret
from newhelm.tracked_object import TrackedObject
from tests.fake_secrets import FakeRequiredSecret


class MockClass(TrackedObject):
    def __init__(self, uid, arg1="1", arg2="2", arg3="3"):
        super().__init__(uid)
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3

    def __eq__(self, other):
        return (
            self.uid == other.uid
            and self.arg1 == other.arg1
            and self.arg2 == other.arg2
            and self.arg3 == other.arg3
        )

    def __repr__(self):
        return f"{self.uid}, {self.arg1}, {self.arg2}, {self.arg3}"


def test_register_and_make():
    factory = InstanceFactory[MockClass]()
    factory.register(MockClass, "key")
    assert factory.make_instance("key", secrets={}) == MockClass("key")


def test_register_and_make_using_args():
    factory = InstanceFactory[MockClass]()
    factory.register(MockClass, "key", "a", "b", "c")
    assert factory.make_instance("key", secrets={}) == MockClass("key", "a", "b", "c")


def test_register_and_make_using_kwargs():
    factory = InstanceFactory[MockClass]()
    factory.register(MockClass, "key", arg1="a", arg2="b", arg3="c")
    assert factory.make_instance("key", secrets={}) == MockClass("key", "a", "b", "c")


def test_register_and_make_using_args_and_kwargs():
    factory = InstanceFactory[MockClass]()
    factory.register(MockClass, "key", "a", "b", arg3="c")
    assert factory.make_instance("key", secrets={}) == MockClass("key", "a", "b", "c")


def test_fails_same_key():
    factory = InstanceFactory[MockClass]()
    factory.register(MockClass, "some-key")
    with pytest.raises(AssertionError) as err_info:
        factory.register(MockClass, "some-key")
    assert (
        "Factory already contains some-key set to MockClass(args=(), kwargs={})."
        in str(err_info)
    )


def test_fails_missing_key():
    factory = InstanceFactory[MockClass]()
    factory.register(MockClass, "some-key")

    with pytest.raises(KeyError) as err_info:
        factory.make_instance("another-key", secrets={})
    assert "No registration for another-key. Known uids: ['some-key']" in str(err_info)


def test_lists_all_items():
    factory = InstanceFactory[MockClass]()
    factory.register(MockClass, "k1", "v1")
    factory.register(MockClass, "k2", "v2")
    factory.register(MockClass, "k3", "v3")
    assert factory.items() == [
        ("k1", FactoryEntry(MockClass, uid="k1", args=("v1",), kwargs={})),
        ("k2", FactoryEntry(MockClass, uid="k2", args=("v2",), kwargs={})),
        ("k3", FactoryEntry(MockClass, uid="k3", args=("v3",), kwargs={})),
    ]


def test_factory_entry_str():
    entry = FactoryEntry(MockClass, uid="k1", args=("v1",), kwargs={"arg2": "v2"})
    assert str(entry) == "MockClass(uid=k1, args=('v1',), kwargs={'arg2': 'v2'})"


class NeedsSecrets(TrackedObject):
    def __init__(self, uid: str, arg1: str, arg2: FakeRequiredSecret):
        super().__init__(uid)
        self.arg1 = arg1
        self.secret = arg2.value


def test_injection():
    factory = InstanceFactory[NeedsSecrets]()
    factory.register(NeedsSecrets, "k1", "v1", InjectSecret(FakeRequiredSecret))
    factory.register(NeedsSecrets, "k2", "v2", arg2=InjectSecret(FakeRequiredSecret))
    secrets = {"some-scope": {"some-key": "some-value"}}
    k1_obj = factory.make_instance("k1", secrets=secrets)
    assert k1_obj.arg1 == "v1"
    assert k1_obj.secret == "some-value"
    k2_obj = factory.make_instance("k2", secrets=secrets)
    assert k2_obj.arg1 == "v2"
    assert k2_obj.secret == "some-value"


class KwargsSecrets(TrackedObject):
    def __init__(self, uid: str, arg1: str, **kwargs):
        super().__init__(uid)
        self.arg1 = arg1
        self.kwargs = kwargs


def test_kwargs_injection():
    factory = InstanceFactory[KwargsSecrets]()
    factory.register(KwargsSecrets, "k1", "v1")
    factory.register(
        KwargsSecrets, "k2", "v2", fake_secret=InjectSecret(FakeRequiredSecret)
    )
    secrets = {"some-scope": {"some-key": "some-value"}}
    k1_obj = factory.make_instance("k1", secrets=secrets)
    assert k1_obj.arg1 == "v1"
    assert k1_obj.kwargs == {}
    k2_obj = factory.make_instance("k2", secrets=secrets)
    assert k2_obj.arg1 == "v2"
    assert k2_obj.kwargs == {"fake_secret": FakeRequiredSecret("some-value")}
