import itertools
import jsonlines
import signal
import time
from csv import DictReader
from typing import List

import pytest

from modelgauge.annotation_pipeline import (
    AnnotatorInputSample,
    AnnotatorInput,
    JsonlAnnotatorOutput,
    CsvAnnotatorInput,
    AnnotatorSource,
    AnnotatorAssigner,
    AnnotatorWorkers,
    AnnotatorSink,
)
from modelgauge.pipeline import PipelineSegment, Pipeline
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import PromptOutput

# from modelgauge.prompt_pipeline import (
#     PromptOutput,
#     PromptInput,
#     CsvPromptInput,
#     CsvPromptOutput,
# )
# from modelgauge.prompt_pipeline import (
#     PromptSource,
#     PromptSutAssigner,
#     PromptSutWorkers,
#     PromptSink,
# )
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from tests.fake_annotator import (
    FakeAnnotation,
    FakeAnnotator,
    FakeAnnotatorRequest,
    FakeAnnotatorResponse,
)


class timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError(f"took more than {self.seconds}s to run")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


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
            yield AnnotatorInputSample(prompt, row["SUT"], response)


class FakePromptOutput(PromptOutput):
    def __init__(self):
        self.output = []

    def write(self, item, results):
        self.output.append({"item": item, "results": results})


# TODO
# class FakeAnnotatorWithDelay(FakeAnnotator):
#     def __init__(self, uid: str = "fake-sut", delay=None):
#         self.delay = itertools.cycle(delay or [0])
#         super().__init__(uid)
#
#     def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
#         time.sleep(next(self.delay))
#         return super().evaluate(request)


@pytest.fixture
def annotators():
    return {"fake1": FakeAnnotator(), "fake2": FakeAnnotator()}


@pytest.fixture
def input_sample():
    return AnnotatorInputSample(
        PromptWithContext(source_id="1", prompt=TextPrompt(text="a")),
        "sut_uid",
        SUTCompletion(text="b"),
    )


def test_csv_annotator_input(tmp_path):
    file_path = tmp_path / "input.csv"
    file_path.write_text('UID,Prompt,SUT,Response\n"1","a","sut_uid","b"')
    input = CsvAnnotatorInput(file_path)

    assert len(input) == 1
    item: AnnotatorInputSample = next(iter(input))
    assert item.prompt.prompt == TextPrompt(text="a")
    assert item.prompt.source_id == "1"
    assert item.sut_uid == "sut_uid"
    assert item.response == SUTCompletion(text="b")


def test_json_annotator_output(tmp_path, annotators, input_sample):
    file_path = tmp_path / "output.jsonl"
    with JsonlAnnotatorOutput(file_path, annotators) as output:
        output.write(input_sample, {"fake1": "a1", "fake2": "a2"})
        output.write(
            AnnotatorInputSample(
                PromptWithContext(source_id="2", prompt=TextPrompt(text="a2")),
                "sut_uid2",
                SUTCompletion(text="b2"),
            ),
            {"fake1": "a12", "fake2": "a22"},
        )

    with jsonlines.open(file_path) as reader:
        items: list[dict] = [i for i in reader]
        assert len(items) == 2
        assert items[0] == {
            "UID": "1",
            "Prompt": "a",
            "SUT": "sut_uid",
            "Response": "b",
            "Annotations": {"fake1": "a1", "fake2": "a2"},
        }
        assert items[1] == {
            "UID": "2",
            "Prompt": "a2",
            "SUT": "sut_uid2",
            "Response": "b2",
            "Annotations": {"fake1": "a12", "fake2": "a22"},
        }


def test_json_annotator_output_dict_annotation(tmp_path, annotators, input_sample):
    file_path = tmp_path / "output.jsonl"

    with JsonlAnnotatorOutput(file_path, annotators) as output:
        output.write(
            input_sample,
            {
                "fake1": FakeAnnotation(sut_text="a1").model_dump(),
                "fake2": FakeAnnotation(sut_text="a2").model_dump(),
            },
        )
    with jsonlines.open(file_path) as reader:
        items: list[dict] = [i for i in reader]
        assert items[0]["Annotations"] == {
            "fake1": {"sut_text": "a1"},
            "fake2": {"sut_text": "a2"},
        }


def test_full_run(annotators):
    input = FakeAnnotatorInput(
        [
            {"UID": "1", "Prompt": "a", "Response": "b", "SUT": "s"},
            {"UID": "2", "Prompt": "c", "Response": "d", "SUT": "s"},
        ]
    )
    output = FakePromptOutput()

    p = Pipeline(
        AnnotatorSource(input),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=1),
        AnnotatorSink(annotators, output),
        debug=True,
    )

    p.run()

    assert len(output.output) == len(input.items)
    assert sorted([r["item"].prompt.source_id for r in output.output]) == [
        i["UID"] for i in input.items
    ]
    row1 = output.output[0]
