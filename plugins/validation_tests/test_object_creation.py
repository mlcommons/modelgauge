import pytest
from newhelm.load_plugins import load_plugins
from newhelm.record_init import get_initialization_record
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS
from tests.fake_secrets import FakeSecrets

# Ensure all the plugins are available during testing.
load_plugins()


@pytest.mark.parametrize("test_name", [key for key, _ in TESTS.items()])
def test_all_tests_construct_and_record_init(test_name):
    secrets = FakeSecrets()
    test = TESTS.make_instance(test_name, secrets=secrets)
    # This throws if things are set up incorrectly.
    get_initialization_record(test)


@pytest.mark.parametrize("sut_name", [key for key, _ in SUTS.items()])
def test_all_suts_construct_and_record_init(sut_name):
    secrets = FakeSecrets()
    sut = SUTS.make_instance(sut_name, secrets=secrets)
    # This throws if things are set up incorrectly.
    get_initialization_record(sut)
