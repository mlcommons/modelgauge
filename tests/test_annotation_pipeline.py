import itertools
import jsonlines
import pytest
import time
from unittest.mock import MagicMock

from modelgauge.annotation_pipeline import (
    SutInteraction,
    AnnotatorInput,
    AnnotatorSource,
    AnnotatorAssigner,
    AnnotatorWorkers,
    AnnotatorSink,
    CsvAnnotatorInput,
    JsonlAnnotatorOutput,
)
from modelgauge.pipeline import Pipeline
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import (
    PromptOutput,
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
)
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from tests.fake_annotator import (
    FakeAnnotation,
    FakeAnnotator,
)
from tests.fake_sut import FakeSUT
from tests.test_prompt_pipeline import FakePromptInput


class FakeAnnotatorInput(AnnotatorInput):
    def __init__(self, items: list[dict], delay=None):
        super().__init__()
        self.items = items
        self.delay = itertools.cycle(delay or [0])

    def __iter__(self):
        for row in self.items:
            time.sleep(next(self.delay))
            prompt = PromptWithContext(
                prompt=TextPrompt(text=row["Prompt"]),
                source_id=row["UID"],
                context=row,
            )
            response = SUTCompletion(text=row["Response"])
            yield SutInteraction(prompt, row["SUT"], response)


class FakeAnnotatorOutput(PromptOutput):
    def __init__(self):
        self.output = {}

    def write(self, item, annotations):
        self.output[item] = annotations


def make_sut_interaction(source_id, prompt, sut_uid, response):
    return SutInteraction(
        PromptWithContext(source_id=source_id, prompt=TextPrompt(text=prompt)),
        sut_uid,
        SUTCompletion(text=response),
    )


def sut_interactions_is_equal(a, b):
    """Equality check that ignores the prompt's context attribute."""
    return (
        a.prompt.source_id == b.prompt.source_id
        and a.prompt.prompt.text == b.prompt.prompt.text
        and a.sut_uid == b.sut_uid
        and a.response == b.response
    )


def test_csv_annotator_input(tmp_path):
    file_path = tmp_path / "input.csv"
    file_path.write_text('UID,Prompt,SUT,Response\n"1","a","s","b"')
    input = CsvAnnotatorInput(file_path)

    assert len(input) == 1
    item: SutInteraction = next(iter(input))
    assert sut_interactions_is_equal(item, make_sut_interaction("1", "a", "s", "b"))


def test_json_annotator_output(tmp_path):
    file_path = tmp_path / "output.jsonl"
    with JsonlAnnotatorOutput(file_path) as output:
        output.write(make_sut_interaction("1", "a", "sut1", "b"), {"fake": "x"})
        output.write(make_sut_interaction("2", "c", "sut2", "d"), {"fake": "y"})

    with jsonlines.open(file_path) as reader:
        items: list[dict] = [i for i in reader]
        assert len(items) == 2
        assert items[0] == {
            "UID": "1",
            "Prompt": "a",
            "SUT": "sut1",
            "Response": "b",
            "Annotations": {"fake": "x"},
        }
        assert items[1] == {
            "UID": "2",
            "Prompt": "c",
            "SUT": "sut2",
            "Response": "d",
            "Annotations": {"fake": "y"},
        }


def test_json_annotator_output_different_annotation_types(tmp_path):
    file_path = tmp_path / "output.jsonl"
    annotations = {
        "fake1": {"sut_text": "a"},
        "fake2": {"sut_text": "b", "num": 0},
        "fake3": "c",
    }
    with JsonlAnnotatorOutput(file_path) as output:
        output.write(make_sut_interaction("1", "a", "s", "b"), annotations)

    with jsonlines.open(file_path) as reader:
        assert reader.read()["Annotations"] == annotations


@pytest.fixture
def annotators():
    annotator_pydantic = FakeAnnotator()
    annotator_dict = FakeAnnotator()
    # Return the same annotation but as a dict.
    annotator_dict.translate_response = MagicMock(
        side_effect=lambda *args: annotator_pydantic.translate_response(
            *args
        ).model_dump()
    )
    annotator_dummy = FakeAnnotator()
    annotator_dummy.translate_response = MagicMock(return_value="d")
    return {
        "annotator_pydantic": annotator_pydantic,
        "annotator_dict": annotator_dict,
        "dummy": annotator_dummy,
    }


@pytest.mark.parametrize(
    "annotator_uid,annotation",
    [
        ("annotator_pydantic", FakeAnnotation(sut_text="response")),
        ("annotator_dict", {"sut_text": "response"}),
        ("dummy", "d"),
    ],
)
def test_annotator_worker_normal(annotators, annotator_uid, annotation):
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    w = AnnotatorWorkers(annotators)
    result = w.handle_item((sut_interaction, annotator_uid))

    assert result[0] == sut_interaction
    assert result[1] == annotator_uid
    assert result[2] == annotation


def test_annotator_worker_cache(annotators, tmp_path):
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    w = AnnotatorWorkers(annotators, cache_path=tmp_path)

    for _ in range(2):
        _, _, annotation = w.handle_item((sut_interaction, "annotator_pydantic"))
        assert annotation == FakeAnnotation(sut_text="response")
        assert annotators["annotator_pydantic"].annotate_calls == 1

        _, _, annotation = w.handle_item((sut_interaction, "annotator_pydantic"))
        assert annotation == FakeAnnotation(sut_text="response")
        assert annotators["annotator_pydantic"].annotate_calls == 1


def test_full_run(annotators):
    input = FakeAnnotatorInput(
        [
            {"UID": "1", "Prompt": "a", "Response": "b", "SUT": "s"},
            {"UID": "2", "Prompt": "c", "Response": "d", "SUT": "s"},
        ]
    )
    output = FakeAnnotatorOutput()
    p = Pipeline(
        AnnotatorSource(input),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=1),
        AnnotatorSink(annotators, output),
        debug=True,
    )
    p.run()

    assert len(output.output) == len(input.items)
    interactions = sorted(list(output.output.keys()), key=lambda o: o.prompt.source_id)
    assert sut_interactions_is_equal(
        interactions[0], make_sut_interaction("1", "a", "s", "b")
    )
    assert output.output[interactions[0]] == {
        "annotator_pydantic": {"sut_text": "b"},
        "annotator_dict": {"sut_text": "b"},
        "dummy": "d",
    }

    assert sut_interactions_is_equal(
        interactions[1], make_sut_interaction("2", "c", "s", "d")
    )
    assert output.output[interactions[1]] == {
        "annotator_pydantic": {"sut_text": "d"},
        "annotator_dict": {"sut_text": "d"},
        "dummy": "d",
    }


def test_progress(annotators):
    input = FakeAnnotatorInput(
        [
            {"UID": "1", "Prompt": "a", "Response": "b", "SUT": "s"},
            {"UID": "2", "Prompt": "c", "Response": "d", "SUT": "s"},
        ]
    )
    output = FakeAnnotatorOutput()

    def track_progress(data):
        progress_items.append(data.copy())

    p = Pipeline(
        AnnotatorSource(input),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=1),
        AnnotatorSink(annotators, output),
        progress_callback=track_progress,
    )
    progress_items = []

    p.run()

    assert progress_items[0]["completed"] == 0
    assert progress_items[-1]["completed"] == len(annotators) * len(input.items)


@pytest.mark.parametrize(
    "sut_worker_count,annotator_worker_count",
    [(1, 1), (2, 2), (8, 8), (1, 5), (5, 1), (3, 9), (9, 3)],
)
def test_prompt_response_annotation_pipeline(
    annotators, sut_worker_count, annotator_worker_count
):
    input = FakePromptInput(
        [
            {"UID": "1", "Text": "a"},
            {"UID": "2", "Text": "b"},
        ]
    )
    output = FakeAnnotatorOutput()

    suts = {"sut1": FakeSUT(), "sut2": FakeSUT()}
    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=sut_worker_count),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=annotator_worker_count),
        AnnotatorSink(annotators, output),
    )
    p.run()

    assert len(output.output) == len(input.items) * len(suts)
    interactions = sorted(
        list(output.output.keys()), key=lambda o: (o.prompt.source_id, o.sut_uid)
    )
    for interaction, prompt_sut in zip(
        interactions, itertools.product(input.items, suts)
    ):
        prompt, sut = prompt_sut
        assert sut_interactions_is_equal(
            interaction,
            make_sut_interaction(prompt["UID"], prompt["Text"], sut, prompt["Text"]),
        )
        annotation = {"sut_text": prompt["Text"]}
        assert output.output[interaction] == {
            "annotator_pydantic": annotation,
            "annotator_dict": annotation,
            "dummy": "d",
        }
