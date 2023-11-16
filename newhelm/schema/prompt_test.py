from newhelm.general import from_json, to_json
from newhelm.schema.prompt import (
    Prompt,
    PromptSUTSpecialization,
)
from newhelm.test_utilities import parent_directory


def test_prompt_from_file(parent_directory):
    in_memory = Prompt(
        text="This is the prompt",
        template_id="1234",
        sut_specialization=PromptSUTSpecialization(
            sut_id="abcd123",
            num_in_context_examples_truncated=2,
            num_characters_truncated=432,
        ),
        id="8326333f020a42ef955ab13e9394f157",
        creation_time_millis=1699999591252,
    )
    with parent_directory.joinpath("testdata", "prompt_example.json").open() as f:
        json_str = f.read()
    from_file = from_json(Prompt, json_str)

    assert in_memory == from_file


def test_prompt_json_round_trip():
    original = Prompt(text="This is the prompt", template_id="1234")
    as_json = to_json(original)
    returned = from_json(Prompt, as_json)
    assert original == returned
