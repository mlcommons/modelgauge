# Developer Quick Start

> [!NOTE]
> This guide assumes you want to edit ModelGauge source. If you want to use it as a library, you can `pip install modelgauge` instead.

## Prerequisites
- **Python 3.10**: It is recommended to use Python version 3.10 with ModelGauge.
- **Poetry**: ModelGauge uses [Poetry](https://python-poetry.org/) for dependency management. [Install](https://python-poetry.org/docs/#installation) it if it's not already on your machine.

**Note:** This quick start is all your need if you want to run ModelGauge and / or write a plugin to add your own SUTs and tests. If you intend to make changes to ModelGauge and open pull requests on the ModelGauge git repository, please read the [Contributor Workflow guide](contributor_workflow.md) first.

## Prerequisites

- **Python 3.10**: It is reccomended to use Python version 3.10 with ModelGauge.
- **Python virtual environment**: It is recommended to install ModelGauge inside of a virtual environment using a system such as Poetry, pyenv, virtualenv or Conda.

## Installation

Run the following (ideally inside your Python virtual environment):

```sh
pip install modelgauge
```

## Getting Started

You can run our command line tool with:

```sh
modelgauge
```

That should provide you with a list of all commands available. A useful command to run is `list`, which will show you all known Tests, System Under Tests (SUTs), and installed plugins.

```sh
modelgauge list
```

ModelGauge uses a [plugin architecture](plugins.md), so by default the list should be pretty empty. To see this in action, we can instruct poetry to install the `demo` plugin:

```sh
pip install modelgauge-demo-plugin@git+https://github.com/mlcommons/modelgauge#subdirectory=demo_plugin
```

After installing the demo plugin, run `modelgauge list` again. You should now see a list of all the modules in from the `demo` plugin.

The `plugins/` directory contains many useful plugins. However, those have a lot of transitive dependencies, so they can take a while to install. Here is a list of officially supported plugins, as well as the commands to install them:

```sh
# OpenAI SUTs
pip install modelgauge-openai@git+https://github.com/mlcommons/modelgauge#subdirectory=plugins/openai

# Hugging Face SUTs
pip install modelgauge-huggingface@git+https://github.com/mlcommons/modelgauge#subdirectory=plugins/huggingface

# Together SUTs
pip install modelgauge-together@git+https://github.com/mlcommons/modelgauge#subdirectory=plugins/together

# Perspective API
pip install modelgauge-perspective-api@git+https://github.com/mlcommons/modelgauge#subdirectory=plugins/perspective_api

# Tests used by AI Safety
pip install modelgauge-standard-tests@git+https://github.com/mlcommons/modelgauge#subdirectory=plugins/standard_tests
```

After installing these plugins, running `modelgauge list` again will display a list of all installed modules.

## Running a Test

Here is an example of running a Test, using the `demo` plugin:

```sh
modelgauge run-test --sut demo_yes_no --test demo_01
```

If you want additional information about existing tests, you can run:

```sh
modelgauge list-tests
```

To obtain detailed information about the existing Systems Under Test (SUTs) in your setup, you can execute the following command:
```sh
modelgauge list-suts
```

## Next steps



# Further Questions

If you have any further questions, please feel free to ask them in the #engineering discord / file a github issue. Also if you see a way to make our documentation better, please submit a pull request. We'd love your help!
