import click


@click.group()
def newhelm_cli():
    """To add a command, decorate your function with @newhelm_cli.command()."""
    pass


def display_header(text):
    """Echo the text, but in bold!"""
    click.echo(click.style(text, bold=True))


def display_list_item(text):
    click.echo(f"\t{text}")


# Define some reusable options
DATA_DIR_OPTION = click.option(
    "--data-dir",
    default="run_data",
    help="Where to store the auxiliary data produced during the run.",
)

SECRETS_FILE_OPTION = click.option(
    "--secrets", default="secrets/default.json", help="File containing needed secrets."
)

SUT_OPTION = click.option("--sut", help="Which registered SUT to run.", required=True)
