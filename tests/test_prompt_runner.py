from csv import DictReader

from modelgauge.prompt_runner import PromptOutput, PromptInput, ParallelPromptRunner, PromptItem, CsvPromptInput, \
    CsvPromptOutput
from tests.fake_sut import FakeSUT


class FakePromptInput(PromptInput):

    def __init__(self, items: list[dict]):
        super().__init__()
        self.items = items

    def __iter__(self):
        for item in self.items:
            yield PromptItem(item)


class FakePromptOutput(PromptOutput):

    def __init__(self):
        self.output = []

    def write(self, item, results):
        self.output.append({"item": item, "results": results})


def test_csv_prompt_input(tmp_path):
    file_path = tmp_path / "input.csv"
    file_path.write_text('UID,Text\n"1","a"')
    input = CsvPromptInput(file_path)

    assert len(input) == 1
    items = [i for i in input]
    assert items[0].uid() == "1"
    assert items[0].prompt() == "a"
    assert len(items) == 1


def test_csv_prompt_output(tmp_path):
    file_path = tmp_path / "output.csv"
    suts = {"fake1": FakeSUT(), "fake2": FakeSUT()}

    with CsvPromptOutput(file_path, suts) as output:
        output.write(PromptItem({"UID": "1", "Text": "a"}), {"fake1": "a1", "fake2": "a2"})

    with open(file_path, "r", newline='') as f:
        # noinspection PyTypeChecker
        items: list[dict] = [i for i in (DictReader(f))]
        assert len(items) == 1
        assert items[0]["UID"] == "1"
        assert items[0]["Text"] == "a"
        assert items[0]["fake1"] == "a1"
        assert items[0]["fake2"] == "a2"


def test_full_run():
    input = FakePromptInput([
        {"UID": "1", "Text": "a"},
        {"UID": "2", "Text": "b"},
    ])
    output = FakePromptOutput()
    ppr = ParallelPromptRunner(input, output, {"fake1": FakeSUT(), "fake2": FakeSUT()}, worker_count=2)

    ppr.run()

    assert len(output.output) == len(input.items)
    assert sorted([r["item"]["UID"] for r in output.output]) == [i["UID"] for i in input.items]
    row1 = output.output[0]
    assert "fake1" in row1["results"]
    assert "fake2" in row1["results"]


def test_progress():
    input = FakePromptInput([
        {"UID": "1", "Text": "a"},
        {"UID": "2", "Text": "b"},
    ])
    output = FakePromptOutput()
    ppr = ParallelPromptRunner(input, output, {"fake1": FakeSUT(), "fake2": FakeSUT()}, worker_count=2)
    progress_items = []

    def track_progress(data):
        progress_items.append(data.copy())

    ppr.run(progress=track_progress)
    assert progress_items[0]["completed"] == 0
    assert progress_items[-1]["completed"] == 2
    print(progress_items)
