import csv
import sys
import traceback
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from typing import Any, Dict

from modelgauge.pipeline import Source, Pipe, Sink
from modelgauge.prompt import TextPrompt
from modelgauge.sut import PromptResponseSUT


class PromptItem(dict):
    def __init__(self, row):
        super().__init__()
        self.update(row)

    def uid(self):
        try:
            return self["UID"]
        except KeyError:
            raise ValueError(f"Missing UID key in {self.keys()}")

    def prompt(self):
        try:
            return self["Text"]
        except KeyError:
            raise ValueError(f"Missing Text key in {self.keys()}")

    def __hash__(self):
        return hash(self.uid())


class PromptInput(metaclass=ABCMeta):
    @abstractmethod
    def __iter__(self):
        pass

    def __len__(self):
        count = 0
        for prompt in self:
            count += 1
        return count


class CsvPromptInput(PromptInput):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def __iter__(self):
        with open(self.path, newline="") as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                yield PromptItem(row)


class PromptOutput(metaclass=ABCMeta):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def write(self, item, results):
        pass


class CsvPromptOutput(PromptOutput):
    def __init__(self, path, suts):
        super().__init__()
        self.path = path
        self.suts = suts
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file, quoting=csv.QUOTE_ALL)
        headers = ["UID", "Text"]
        self.writer.writerow(headers + [s for s in self.suts.keys()])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def write(self, item: PromptItem, results):
        row = [item.uid(), item.prompt()]
        for k in self.suts:
            if k in results:
                row.append(results[k])
            else:
                row.append("")
        self.writer.writerow(row)


class PromptSource(Source):
    def __init__(self, input: PromptInput):
        super().__init__()
        self.input = input

    def new_item_iterable(self):
        return self.input


class PromptSutAssigner(Pipe):
    def __init__(self, suts: dict[str, PromptResponseSUT]):
        super().__init__()
        self.suts = suts

    def handle_item(self, item: PromptItem):
        for sut_uid in self.suts:
            self.downstream_put((item, sut_uid))


class PromptSutWorkers(Pipe):
    def __init__(self, suts: dict[str, PromptResponseSUT], workers=None):
        if workers is None:
            workers = 8
        super().__init__(thread_count=workers)
        self.suts = suts

    def handle_item(self, item):
        prompt_item, sut_uid = item
        try:
            sut = self.suts[sut_uid]
            request = sut.translate_text_prompt(TextPrompt(text=prompt_item.prompt()))
            response = sut.evaluate(request)
            result = sut.translate_response(request, response)
            return (prompt_item, sut_uid, result.completions[0].text)
        except:
            print(
                f"unexpected failure processing {item} for {sut_uid}", file=sys.stderr
            )
            traceback.print_exc(file=sys.stderr)


class PromptSink(Sink):
    unfinished: defaultdict[PromptItem, dict[str, str]]

    def __init__(self, suts: dict[str, PromptResponseSUT], writer: PromptOutput):
        super().__init__()
        self.suts = suts
        self.writer = writer
        self.unfinished = defaultdict(lambda: dict())

    def run(self):
        with self.writer:
            super().run()

    def handle_item(self, item):
        prompt_item, sut_key, response = item
        self.unfinished[prompt_item][sut_key] = response
        if len(self.unfinished[prompt_item]) == len(self.suts):
            self.writer.write(prompt_item, self.unfinished[prompt_item])
            self._debug(f"wrote {prompt_item}")
            del self.unfinished[prompt_item]
