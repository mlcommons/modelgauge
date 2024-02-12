import os
import pathlib

import pytest


@pytest.fixture
def cmd():
    return pathlib.Path(__file__).parent.parent / "newhelm" / "main.py"


def test_main(cmd):
    assert os.system(f"python {cmd}") == 0


def test_list_plugins(cmd):
    assert os.system(f"python {cmd} list") == 0


@pytest.mark.requires_plugins
@pytest.mark.parametrize("sut", ["DemoMultipleChoiceSUT"])
@pytest.mark.parametrize("test", ["demo_01", "demo_02"])
def test_test_sut_combinations(cmd, test, sut):
    assert (
        os.system(
            f"""python {cmd} run-test \
                --test {test} \
                --sut {sut} \
                --max-test-items 1"""
        )
        == 0
    )
