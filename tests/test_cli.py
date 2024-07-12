import csv
import os
import re

import pytest
from click.testing import CliRunner

from modelgauge import main
from modelgauge.load_plugins import load_plugins
from modelgauge.sut import SUT
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from tests.utilities import expensive_tests


@expensive_tests
def test_main():
    assert os.system("modelgauge") == 0


@expensive_tests
def test_list_plugins():
    assert os.system("modelgauge list") == 0


@expensive_tests
def test_list_secrets():
    assert os.system("modelgauge list-secrets") == 0


@expensive_tests
def test_list_tests():
    assert os.system("modelgauge list-tests") == 0


@expensive_tests
def test_list_suts():
    assert os.system("modelgauge list-suts") == 0


@expensive_tests
@pytest.mark.parametrize(
    "sut",
    [
        "demo_yes_no",
        "demo_random_words",
        "demo_always_angry",
        "demo_always_sorry",
    ],
)
def test_run_sut_demos(sut):
    assert (
        os.system(
            f"""modelgauge run-sut \
                --sut {sut} \
                --prompt "Can you say Hello?" """
        )
        == 0
    )


@expensive_tests
@pytest.mark.parametrize("test", ["demo_01", "demo_02", "demo_03", "demo_04"])
def test_run_test_demos(test):
    assert (
        os.system(
            f"""modelgauge run-test \
                --test {test} \
                --sut demo_yes_no \
                --max-test-items 1"""
        )
        == 0
    )


def test_run_prompts_normal(tmp_path):
    load_plugins()

    in_path = (tmp_path / "input.csv").absolute()

    with open(in_path, "w") as f:
        f.write("UID,Text,Ignored\np1,Say yes,ignored\np2,Refuse,ignored\n")

    runner = CliRunner()
    result = runner.invoke(main.run_prompts, ["--sut", "demo_yes_no", str(in_path)],catch_exceptions=False )

    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.csv", result.stdout)[0]
    with open(in_path.parent / out_path, "r") as f:
        reader = csv.DictReader(f)

        row1 = next(reader)
        assert row1["UID"] == "p1"
        assert row1["Text"] == "Say yes"
        assert row1["demo_yes_no"] == "Yes"

        row2 = next(reader)
        assert row2["UID"] == "p2"
        assert row2["Text"] == "Refuse"
        assert row2["demo_yes_no"] == "No"


@modelgauge_sut(capabilities=[])
class NoReqsSUT(SUT):
    pass


def test_run_prompts_bad_sut(tmp_path):
    load_plugins()

    SUTS.register(NoReqsSUT, "noreqs")

    in_path = (tmp_path / "input.csv").absolute()

    with open(in_path, "w") as f:
        f.write("UID,Text,Ignored\np1,Say yes,ignored\n")

    runner = CliRunner()
    result = runner.invoke(main.run_prompts, ["--sut", "noreqs", str(in_path)], catch_exceptions=False)
    assert result.exit_code == 2
    assert re.search(r"noreqs does not accept text prompts", str(result.output))
