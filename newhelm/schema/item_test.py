from newhelm.general import from_json, to_json
from newhelm.schema.item import ItemPromptBlock, ItemPromptTemplate
from newhelm.test_utilities import parent_directory


def test_item_prompt_template_from_file(parent_directory):
    in_memory = ItemPromptTemplate(
        eval_item_block=ItemPromptBlock(
            "The color of the sky is: blue", item_id="1111"
        ),
        reference_index=3,
        train_items=[
            ItemPromptBlock("The color of a polar bear is: white", item_id="123"),
            ItemPromptBlock("The color of space is: black", item_id="987"),
        ],
        id="025d360242974681aa092624bf3391bf",
        creation_time_millis=1699998606466,
    )
    with parent_directory.joinpath(
        "testdata", "item_prompt_template_example.json"
    ).open() as f:
        json_str = f.read()
    from_file = from_json(ItemPromptTemplate, json_str)

    assert in_memory == from_file


def test_item_prompt_template_json_round_trip():
    original = ItemPromptTemplate(
        eval_item_block=ItemPromptBlock(
            "The color of the sky is: blue", item_id="1111"
        ),
    )
    as_json = to_json(original)
    returned = from_json(ItemPromptTemplate, as_json)
    assert original == returned
