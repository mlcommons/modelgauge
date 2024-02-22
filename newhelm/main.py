import json
import click

from newhelm.command_line import (
    SECRETS_FILE_OPTION,
    SUT_OPTION,
    display_header,
    display_list_item,
    newhelm_cli,
)
from newhelm.general import get_or_create_json_file

from newhelm.load_plugins import load_plugins, list_plugins
from newhelm.prompt import Prompt
from newhelm.secrets_registry import SECRETS
from newhelm.sut import PromptResponseSUT
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
def list() -> None:
    """Overview of Plugins, Tests, and SUTs."""
    plugins = list_plugins()
    display_header(f"Plugin Modules: {len(plugins)}")
    for module_name in plugins:
        display_list_item(module_name)
    suts = SUTS.items()
    display_header(f"SUTS: {len(suts)}")
    for sut, sut_entry in suts:
        display_list_item(f"{sut} {sut_entry}")
    tests = TESTS.items()
    display_header(f"Tests: {len(tests)}")
    for test, test_entry in tests:
        display_list_item(f"{test} {test_entry}")


@newhelm_cli.command()
def list_tests() -> None:
    """List details about all registered tests."""
    for test, test_entry in TESTS.items():
        test_obj = test_entry.make_instance()
        metadata = test_obj.get_metadata()
        display_header(metadata.name)
        click.echo(f"Command line key: {test}")
        click.echo(f"Description: {metadata.description}")
        click.echo()


# TODO: Consider moving this somewhere else.
@newhelm_cli.command()
@SUT_OPTION
@SECRETS_FILE_OPTION
@click.option("--prompt", help="The full text to send to the SUT.")
def run_sut(sut: str, secrets: str, prompt: str):
    """Send a prompt from the command line to a SUT."""
    sut_obj = SUTS.make_instance(sut)
    SECRETS.set_values(get_or_create_json_file(secrets))
    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)

    prompt_obj = Prompt(text=prompt)
    request = sut_obj.translate_request(prompt_obj)
    click.echo(f"{request}\n")
    response = sut_obj.evaluate(request)
    click.echo(f"{response}\n")
    result = sut_obj.translate_response(request, response)
    click.echo(f"{result}\n")


@newhelm_cli.command()
@click.option(
    "--scope", help="The scope of the secret. For example, 'together'.", required=True
)
@click.option(
    "--key",
    help="The dictionary key of the secret. For example, 'api_key'",
    required=True,
)
@click.option(
    "--value", help="The secret to add. For example, '8a8adsfnnc898s8'", required=True
)
@SECRETS_FILE_OPTION
def set_secret(scope: str, key: str, value: str, secrets: str):
    """Set the value of a secret in the secrets file."""
    current_secrets = get_or_create_json_file(secrets)
    if scope not in current_secrets:
        current_secrets[scope] = {}
    current_scope = current_secrets[scope]
    current_scope[key] = value
    with open(secrets, "w") as f:
        json.dump(current_secrets, f, indent=4)
        print("", file=f)  # Add newline to end


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
