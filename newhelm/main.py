import click

from newhelm.benchmark_registry import BENCHMARKS
from newhelm.command_line import newhelm_cli

from newhelm.load_plugins import load_plugins, list_plugins
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
def list():
    plugins = list_plugins()
    click.echo(click.style(f"Plugin Modules: {len(plugins)}", bold=True))
    for module_name in plugins:
        click.echo("\t", nl=False)
        click.echo(module_name)
    click.echo(click.style("SUTS:", bold=True))
    for sut, entry in SUTS.items():
        click.echo("\t", nl=False)
        click.echo(f"{sut} {entry.cls.__name__} {entry.args}")
    click.echo(click.style("Tests:", bold=True))
    for test, entry in TESTS.items():
        click.echo("\t", nl=False)
        click.echo(f"{test} {entry.cls.__name__} {entry.args}")
    click.echo(click.style("Benchmarks:", bold=True))
    for benchmark, entry in BENCHMARKS.items():
        click.echo("\t", nl=False)
        click.echo(f"{benchmark} {entry.cls.__name__} {entry.args}")


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
