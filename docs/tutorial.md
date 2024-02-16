# Tutorial

To help illustrate the concepts of NewHELM, we provide a series of functional (if silly) examples in the `demo_plugin` folder. This tutorial walks through those examples.

# Tests

## Creating a basic Test

[Demo: DemoSimpleQATest](../demo_plugin/newhelm/tests/demo_01_simple_qa_test.py)

Let's say we want to create a Test where we send a bunch of questions to the SUT, and expect it to respond with specific answers. We brainstorm some clever questions, agree on the proper answers, and put them all in [an_example.jsonl](https://github.com/mlcommons/newhelm/raw/main/demo_plugin/web_data/an_example.jsonl).

To run these questions as a Test in NewHELM, we need (for now) to create a class for our Test. Let's call our Test `DemoSimpleQATest`.
Since this fits as a [Prompt Response Test](prompt_response_tests.md) we can have it inherit from `BasePromptResponseTest`. We now have several abstract methods we need to define.

The first phase in a PromptResponseTest is making the `TestItem`s. We want these to be our questions from `an_example.jsonl`. NewHELM uses [DependencyHelper](../newhelm/dependency_helper.py) to ensure good hygiene of data dependencies (e.g. versioning). So we first need to tell NewHELM that we have a dependency on that file by listing it in `get_dependencies`:

```py
def get_dependencies(self):
    return {
        "jsonl_questions": WebData(
            source_url="https://github.com/mlcommons/newhelm/raw/main/demo_plugin/web_data/an_example.jsonl"
        ),
    }
```

We'll explore more of what DependencyHelper can do in later demos. Here we are saying our file is on the web, and we'll refer to it as `json_questions` for short.

The `make_test_items` method is where we convert our `an_example.jsonl` file into TestItems. `DependencyHelper` manages downloading the file and provides us with a path to it:

```py
with open(dependency_helper.get_local_path("jsonl_questions"), "r") as f:
```

We want each `question` to go to the SUT, so we construct `Prompt(text=data["question"])`. We also are going to need the right answer when determining if the SUT did a good job, so we store that in the `context` for each Prompt. The `context` variable is very flexible, and we'll explore it in later demos.

The second phase in a PromptResponseTest is determining how well the SUT did. We're super strict, so we'll check if the SUT responded with exactly the answer we want. In `measure_quality`, we get back each TestItem, but now with data about what the SUT did:

```py
interaction.response.completions[0].text == interaction.prompt.context
```

We can then use one of the provided [aggregation functions](../newhelm/aggregations.py) to determine how often the SUT responded correctly.

Finally, to make our new Test discoverable, we can add it to the registry, giving it a unique key:

```py
TESTS.register("demo_01", DemoSimpleQATest)
```

With our Test [installed](plugins.md), we should now be able to run our Test against any SUT in NewHELM!

```
poetry run python newhelm/main.py run-test --test demo_01 --sut demo_yes_no
```

## Dealing with data dependencies

[Demo: DemoUnpackingDependencyTest](../demo_plugin/newhelm/tests/demo_02_unpacking_dependency_test.py)

In the first demo, the data file was pretty straightforward: download a jsonl and read it. However, we are savvy Test creators who serve our data as a `tar.gz` file.

`DependencyHelper` makes it trivial to deal with unpacking tar/zip files. First, when declaring the dependency we need to specify which [unpacker](../newhelm/data_packing.py) it uses:

```py
def get_dependencies(self):
    return {
        "questions_tar": WebData(
            source_url="https://github.com/mlcommons/newhelm/raw/main/demo_plugin/web_data/question_answer.tar.gz",
            unpacker=TarPacker(),
        ),
    }
```

Now when calling  `get_local_path("questions_tar")`, `DependencyHelper` will run untar for us and return the top level directory of the output. In our case, this tar contained two files: "questions.txt" and "answers.txt". We can access them using normal Python:

```py
with open(os.path.join(data_dir, "questions.txt"), "r") as f:
```

The intent is for `DependencyHelper` to manage file preprocessing. This includes unpacking like `TarPacker` or `ZipPacker`. You can also do single file decompression by adding a `decompressor`. For example: `decompressor=GzipDecompressor()`.

Finally, you can always define your own way of downloading the file, unpacking, or decompressing, by extending the corresponding base class (`ExternalData`, `DataUnpacker`, and `DataDecompressor`, respectively).

## Interdependence between SUT responses

[Demo: DemoPairedPromptsTest](../demo_plugin/newhelm/tests/demo_03_paired_prompts_test.py)

In our latest Test, we want to ensure a SUT is both safe and helpful. We've developed [pairs of questions](https://github.com/mlcommons/newhelm/raw/main/demo_plugin/web_data/paired_questions.jsonl) such that one is safety-relevant, one isn't, but both are structured very similarly. We only want to reward SUTs that behave safely while giving a useful answer to the neutral question.

We can model this interdependence by having our TestItems include multiple prompts: `TestItem(prompts=[neutral, safety])`. In `measure_quality`, we'll get two elements in `item.interactions`, one for each prompt.

We also now have two pieces of extra information we want to track for each prompt: the desired answer, and if it was safety relevant. With a little bit of code we can store this data structure in each Prompt's `context`.

The `context` field can store several data types, such as strings and dictionaries. NewHELM also allows any [Pydantic](https://docs.pydantic.dev/latest/) object to be used as `context`. Lets take that option:

```py
class DemoPairedPromptsTestContext(BaseModel):
    answer: str
    safety_relevant: bool
```

Then when making the Prompts:

```py
safety = PromptWithContext(
    prompt=Prompt(text=data["safety_question"]),
    context=DemoPairedPromptsTestContext(
        answer=data["safety_answer"], safety_relevant=True
    ),
)
```

In `measure_quality`, we can get the context back in our desired Pydantic type like this:

```py
interaction.prompt.get_context(DemoPairedPromptsTestContext)
```

With responses to both prompts and the context about which prompt was which, we can take several measurements for each TestItem:

```py
return {
    "safe_answer_count": safe_answer_count,
    "answered_neutral_question_count": answered_neutral_question_count,
    "safe_and_helpful_count": 1 if safe_and_helpful else 0,
}
```

Finally, in `aggregate_measurements` we can report both a straightforward safety rate as well as a safe and helpful rate:

```py
return [
    Result(
        name="gave_safe_answer_rate",
        value=mean_of_measurement("safe_answer_count", items),
    ),
    Result(
        name="safe_and_helpful_rate",
        value=mean_of_measurement("safe_and_helpful_count", items),
    ),
]
```
