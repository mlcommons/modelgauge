# Developer Quick Start

To get started run the following at the top level directory of the checked out repository:

```
poetry install
```

This will instruct [poetry](https://python-poetry.org/docs/) to install the default dependencies into this project's environment. After you install, future `poetry run` commands will use that environment.

> [!WARNING]
> Poetry and other python virtual environment tooling [may not play nicely together](https://github.com/orgs/python-poetry/discussions/7767). As such we recommend you let Poetry manage the venv, and not try to run it within a venv.

For example, you can run our command line tool with:

```
poetry run newhelm
```

That should provide you with a list of all commands available. A useful command to run is `list`, which will show you all known Tests, System Under Tests (SUTs), and installed plugins.

```
poetry run newhelm list
```

NewHELM uses a [plugin architecture](plugins.md), so by default the list should be pretty empty. To see this in action, we can instruct poetry to install the `demo` plugin:

```
poetry install --extras demo
poetry run newhelm list
```

You should now see a list of all the modules in the `demo_plugin/` directory. For more info on the demo see [here](tutorial.md). The `plugins/` directory contains many useful plugins. However, those have a lot of transitive dependencies, so they can take a while to install. To install them all:

```
poetry install --extras all_plugins
poetry run newhelm list
```

Finally note that any extras not listed in a `poetry install` call will be uninstalled.

## Running a Test

Here is an example of running a Test, using the `demo` plugin:

```
poetry run newhelm run-test --sut demo_yes_no --test demo_01
```

If you want additional information about existing tests, you can run:

```
poetry run newhelm list-tests
```

To obtain detailed information about the existing Systems Under Test (SUTs) in your setup, you can execute the following command:
```
poetry run newhelm list-suts
```

# Further Questions

If you have any further questions, please feel free to ask them in the #engineering discord / file a github issue. Also if you see a way to make our documentation better, please submit a pull request. We'd love your help!
