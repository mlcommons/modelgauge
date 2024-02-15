# Demo Plugin

This directory contains worked examples for how to define your own Tests, and SUTs in NewHelm. For a more detailed tutorial, see the docs. To see this code in action, from the top level git directory:

```
poetry install --extras demo
poetry run python newhelm/main.py run-test --test demo_01 --sut DemoMultipleChoiceSUT
```
