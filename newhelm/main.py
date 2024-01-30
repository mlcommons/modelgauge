import click
from newhelm.benchmark_registry import BENCHMARKS
from newhelm.command_line import SECRETS_FILE_OPTION, SUT_OPTION, newhelm_cli
from newhelm.general import get_or_create_json_file

from newhelm.load_plugins import load_plugins
from newhelm.placeholders import Prompt
from newhelm.secrets_registry import SECRETS
from newhelm.sut import PromptResponseSUT
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
def list():
    print("SUTS:")
    for sut, entry in SUTS.items():
        print("\t", sut, entry)
    print("Tests:")
    for test, entry in TESTS.items():
        print("\t", test, entry)
    print("Benchmarks:")
    for benchmark, entry in BENCHMARKS.items():
        print("\t", benchmark, entry)


# TODO: Consider moving this somewhere else.
@newhelm_cli.command()
@SUT_OPTION
@SECRETS_FILE_OPTION
@click.option("--prompt", help="The full text to send to the SUT.")
def run_sut(sut: str, secrets: str, prompt: str):
    sut_obj = SUTS.make_instance(sut)
    SECRETS.set_values(get_or_create_json_file(secrets))
    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)

    prompt_obj = Prompt(prompt)
    request = sut_obj.translate_request(prompt_obj)
    print(request, "\n")
    response = sut_obj.evaluate(request)
    print(response, "\n")
    result = sut_obj.translate_response(prompt_obj, response)
    print(result, "\n")


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
