from newhelm.general import from_json, to_json
from newhelm.schema.instance import (
    InstancePromptBlock,
    InstancePromptTemplate,
    Instance,
    Reference,
)
from newhelm.test_utilities import parent_directory


def test_instance_from_file(parent_directory):
    in_memory = Instance(
        text="What is 2+2?",
        references=[
            Reference("7"),
            Reference("4"),
        ],
        split="TEST",
        id="0040d080d022464a8184c8c49b02bdea",
        creation_time_millis=1700148080049,
    )
    with parent_directory.joinpath("testdata", "instance_example.json").open() as f:
        json_str = f.read()
    from_file = from_json(Instance, json_str)

    assert in_memory == from_file


def test_instance_json_round_trip():
    original = Instance(
        text="This is the song that never ends,",
        references=[],
        split="TRAIN",
    )
    as_json = to_json(original)
    returned = from_json(Instance, as_json)
    assert original == returned


def test_instance_prompt_template_from_file(parent_directory):
    in_memory = InstancePromptTemplate(
        eval_instance_block=InstancePromptBlock(
            "The color of the sky is: blue", instance_id="1111"
        ),
        reference_index=3,
        train_instances=[
            InstancePromptBlock(
                "The color of a polar bear is: white", instance_id="123"
            ),
            InstancePromptBlock("The color of space is: black", instance_id="987"),
        ],
        id="025d360242974681aa092624bf3391bf",
        creation_time_millis=1699998606466,
    )
    with parent_directory.joinpath(
        "testdata", "instance_prompt_template_example.json"
    ).open() as f:
        json_str = f.read()
    from_file = from_json(InstancePromptTemplate, json_str)

    assert in_memory == from_file


def test_instance_prompt_template_json_round_trip():
    original = InstancePromptTemplate(
        eval_instance_block=InstancePromptBlock(
            "The color of the sky is: blue", instance_id="1111"
        ),
    )
    as_json = to_json(original)
    returned = from_json(InstancePromptTemplate, as_json)
    assert original == returned
