import click
import datetime
import os
import pathlib
from typing import List, Optional

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.base_test import PromptResponseTest
from modelgauge.command_line import (
    DATA_DIR_OPTION,
    LOCAL_PLUGIN_DIR_OPTION,
    MAX_TEST_ITEMS_OPTION,
    SUT_OPTION,
    display_header,
    display_list_item,
    modelgauge_cli,
)
from modelgauge.config import (
    load_secrets_from_config,
    raise_if_missing_from_config,
    toml_format_secrets,
)
from modelgauge.dependency_injection import list_dependency_usage
from modelgauge.general import normalize_filename
from modelgauge.instance_factory import FactoryEntry
from modelgauge.load_plugins import list_plugins
from modelgauge.pipeline_runner import run_annotator_pipeline, run_prompts_pipeline
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.secret_values import MissingSecretValues, RawSecrets, get_all_secrets
from modelgauge.simple_test_runner import run_prompt_response_test
from modelgauge.sut import PromptResponseSUT, SUT
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS


@modelgauge_cli.command(name="list")
@LOCAL_PLUGIN_DIR_OPTION
def list_command() -> None:
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


def _display_factory_entry(uid: str, entry: FactoryEntry, secrets: RawSecrets):
    def format_missing_secrets(missing):
        """Return formatted string for missing secrets."""
        return "\n".join(
            f"Scope: '{secret['scope']}', Key: '{secret['key']}', Instructions: '{secret['instructions']}'"
            for secret in missing
        )

    used, missing = list_dependency_usage(entry.args, entry.kwargs, secrets)
    missing = format_missing_secrets(missing)

    display_header(uid)
    click.echo(f"Class: {entry.cls.__name__}")
    click.echo(f"Args: {entry.args}")
    click.echo(f"Kwargs: {entry.kwargs}")

    if used:
        click.echo("Used Secrets:")
        click.echo(used)
    else:
        click.echo("No Secrets Used.")

    if missing:
        click.echo("Missing Secrets:")
        click.echo(missing)

    click.echo()


@modelgauge_cli.command()
@LOCAL_PLUGIN_DIR_OPTION
def list_tests() -> None:
    """List details about all registered tests."""
    secrets = load_secrets_from_config()
    for test_uid, test_entry in TESTS.items():
        _display_factory_entry(test_uid, test_entry, secrets)


@modelgauge_cli.command()
@LOCAL_PLUGIN_DIR_OPTION
def list_suts():
    """List details about all registered SUTs (System Under Test)."""
    secrets = load_secrets_from_config()
    for sut_uid, sut in SUTS.items():
        _display_factory_entry(sut_uid, sut, secrets)


@modelgauge_cli.command()
@LOCAL_PLUGIN_DIR_OPTION
def list_secrets() -> None:
    """List details about secrets modelgauge might need."""
    descriptions = get_all_secrets()
    if descriptions:
        display_header("Here are the known secrets modelgauge might use.")
        click.echo(toml_format_secrets(descriptions))
    else:
        display_header("No secrets used by any installed plugin.")


@modelgauge_cli.command()
@LOCAL_PLUGIN_DIR_OPTION
@SUT_OPTION
@click.option("--prompt", help="The full text to send to the SUT.")
@click.option(
    "--num-completions",
    default=None,
    type=click.IntRange(1),
    help="How many different completions to generation.",
)
@click.option(
    "--max-tokens",
    default=None,
    type=click.IntRange(1),
    help="How many tokens to generate for each completion.",
)
@click.option(
    "--top-logprobs",
    type=click.IntRange(1),
    help="How many log probabilities to report for each token position.",
)
def run_sut(
    sut: str,
    prompt: str,
    num_completions: Optional[int],
    max_tokens: Optional[int],
    top_logprobs: Optional[int],
):
    """Send a prompt from the command line to a SUT."""
    secrets = load_secrets_from_config()
    try:
        sut_obj = SUTS.make_instance(sut, secrets=secrets)
    except MissingSecretValues as e:
        raise_if_missing_from_config([e])

    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)

    options = SUTOptions()
    if num_completions:
        options.num_completions = num_completions
    if max_tokens:
        options.max_tokens = max_tokens
    if top_logprobs:
        options.top_logprobs = top_logprobs
    prompt_obj = TextPrompt(text=prompt, options=options)
    request = sut_obj.translate_text_prompt(prompt_obj)
    click.echo(f"Native request: {request}\n")
    response = sut_obj.evaluate(request)
    click.echo(f"Native response: {response}\n")
    result = sut_obj.translate_response(request, response)
    click.echo(f"Normalized response: {result.model_dump_json(indent=2)}\n")


@modelgauge_cli.command()
@click.option("--test", help="Which registered TEST to run.", required=True)
@LOCAL_PLUGIN_DIR_OPTION
@SUT_OPTION
@DATA_DIR_OPTION
@MAX_TEST_ITEMS_OPTION
@click.option(
    "--output-file",
    help="If specified, will override the default location for outputting the TestRecord.",
)
@click.option(
    "--no-caching",
    is_flag=True,
    show_default=True,
    default=False,
    help="Disable caching.",
)
@click.option(
    "--no-progress-bar",
    is_flag=True,
    show_default=True,
    default=False,
    help="Disable displaying the 'Processing TestItems' progress bar.",
)
def run_test(
    test: str,
    sut: str,
    data_dir: str,
    max_test_items: int,
    output_file: Optional[str],
    no_caching: bool,
    no_progress_bar: bool,
):
    """Run the Test on the desired SUT and output the TestRecord."""
    secrets = load_secrets_from_config()
    # Check for missing secrets without instantiating any objects
    missing_secrets: List[MissingSecretValues] = []
    missing_secrets.extend(TESTS.get_missing_dependencies(test, secrets=secrets))
    missing_secrets.extend(SUTS.get_missing_dependencies(sut, secrets=secrets))
    raise_if_missing_from_config(missing_secrets)

    test_obj = TESTS.make_instance(test, secrets=secrets)
    sut_obj = SUTS.make_instance(sut, secrets=secrets)

    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)
    assert isinstance(test_obj, PromptResponseTest)

    if output_file is None:
        os.makedirs("output", exist_ok=True)
        output_file = os.path.join(
            "output", normalize_filename(f"record_for_{test}_{sut}.json")
        )
    test_record = run_prompt_response_test(
        test_obj,
        sut_obj,
        data_dir,
        max_test_items,
        use_caching=not no_caching,
        disable_progress_bar=no_progress_bar,
    )
    with open(output_file, "w") as f:
        print(test_record.model_dump_json(indent=4), file=f)
    # For displaying to the screen, clear out the verbose test_item_records
    test_record.test_item_records = []
    print(test_record.model_dump_json(indent=4))
    print("Full TestRecord json written to", output_file)


@modelgauge_cli.command()
@click.option(
    "sut_uids",
    "-s",
    "--sut",
    help="Which registered SUT(s) to run.",
    multiple=True,
    required=True,
)
@click.option(
    "annotator_uids",
    "-a",
    "--annotator",
    help="Which registered annotator(s) to run",
    multiple=True,
    required=False,
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of worker threads, default is 10 * number of SUTs.",
)
@click.option(
    "sut_cache_dir",
    "--cache",
    help="Directory to cache model answers (only applies to SUTs).",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--debug", is_flag=True, help="Show internal pipeline debugging information."
)
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
)
def run_prompts(sut_uids, annotator_uids, workers, sut_cache_dir, debug, input_path):
    """Take a CSV of prompts and run them through SUTs and/or annotators.

    The CSV file must have 'UID' and 'Text' columns.
    """
    secrets = load_secrets_from_config()
    try:
        suts: dict[str, SUT] = {
            sut_uid: SUTS.make_instance(sut_uid, secrets=secrets)
            for sut_uid in sut_uids
        }
    except MissingSecretValues as e:
        raise_if_missing_from_config([e])

    if sut_cache_dir:
        print(f"Creating cache dir {sut_cache_dir}")
        sut_cache_dir.mkdir(exist_ok=True)

    for sut_uid in suts:
        sut = suts[sut_uid]
        if not AcceptsTextPrompt in sut.capabilities:
            raise click.BadParameter(f"{sut_uid} does not accept text prompts")

    output_path = input_path.parent / pathlib.Path(
        input_path.stem
        + "-responses-"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".csv"
    )

    annotators = None
    if len(annotator_uids):
        try:
            annotators = {
                annotator_uid: ANNOTATORS.make_instance(annotator_uid, secrets=secrets)
                for annotator_uid in annotator_uids
            }
        except MissingSecretValues as e:
            raise_if_missing_from_config([e])

        output_path = input_path.parent / pathlib.Path(
            input_path.stem
            + "-annotated-responses"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            + ".jsonl"
        )

    run_prompts_pipeline(
        suts, workers, sut_cache_dir, debug, input_path, output_path, annotators
    )


@modelgauge_cli.command()
@click.option(
    "annotator_uids",
    "-a",
    "--annotator",
    help="Which annotator to run",
    multiple=True,
    required=True,
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of worker threads, default is 10 * number of SUTs.",
)
@click.option(
    "--debug", is_flag=True, help="Show internal pipeline debugging information."
)
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
)
def run_annotators(annotator_uids, workers, input_path, debug):
    """Take a CSV of prompts and responses and run them through annotators.
    The CSV file must contain UID, Prompt, SUT, and Response columns.
    Outputs a jsonl file."""

    secrets = load_secrets_from_config()

    try:
        annotators = {
            annotator_uid: ANNOTATORS.make_instance(annotator_uid, secrets=secrets)
            for annotator_uid in annotator_uids
        }
    except MissingSecretValues as e:
        raise_if_missing_from_config([e])

    output_path = input_path.parent / pathlib.Path(
        input_path.stem
        + "-annotations-"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".jsonl"
    )

    run_annotator_pipeline(annotators, workers, debug, input_path, output_path)


def main():
    modelgauge_cli()


if __name__ == "__main__":
    main()
