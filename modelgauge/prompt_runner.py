import csv
import datetime
import pathlib
import queue
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from threading import Event, Thread
from typing import Callable

import click

from modelgauge.config import load_secrets_from_config, raise_if_missing_from_config
from modelgauge.load_plugins import load_plugins
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import MissingSecretValues
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_registry import SUTS


class PromptItem(dict):
    def __init__(self, row):
        super().__init__()
        self.update(row)

    def uid(self):
        return self["UID"]

    def prompt(self):
        return self["Text"]

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


def do_nothing(*args):
    pass


class ParallelPromptRunner:
    def __init__(
        self,
        input: PromptInput,
        output: PromptOutput,
        suts: dict[str, PromptResponseSUT],
        worker_count: int = None,
    ):
        super().__init__()
        self.input = input
        self.output = output
        self.suts = suts
        if worker_count:
            self._worker_count = worker_count
        else:
            self._worker_count = 10 * len(suts)

        # queues
        self.to_process = queue.Queue(maxsize=self._worker_count * 4)
        self.to_output = queue.Queue(maxsize=self._worker_count * 2)

        # data
        self.unfinished = defaultdict(lambda: dict())
        self.stats = defaultdict(lambda: 0)

        # flags
        self.all_input_read = Event()

    def done_processing(self):
        return self.all_input_read.is_set() and self.to_process.empty()

    def done_outputting(self):
        return self.done_processing() and self.to_output.empty()

    def _read(self):
        for item in self.input:
            for sut_key in self.suts:
                self.to_process.put((item, sut_key))
            self.stats["read"] += 1
        self.all_input_read.set()

    def _process(self):
        while not self.done_processing():
            item, sut_key = self.to_process.get()
            request = self.suts[sut_key].translate_text_prompt(
                TextPrompt(text=item.prompt())
            )
            response = self.suts[sut_key].evaluate(request)
            result = self.suts[sut_key].translate_response(request, response)

            self.to_output.put((item, sut_key, result.completions[0].text))
            self.stats["process"] += 1

            self.to_process.task_done()

    def _handle_output(self):
        with self.output as output:
            while not self.done_outputting():
                item, sut_key, response = self.to_output.get()
                self.unfinished[item][sut_key] = response
                if len(self.unfinished[item]) == len(self.suts):
                    output.write(item, self.unfinished[item])
                    del self.unfinished[item]
                    self.to_output.task_done()
                    self.stats["completed"] += 1

    def run(self, progress: Callable = do_nothing):
        self.all_input_read.clear()
        progress(self.stats)

        input_worker = Thread(target=self._read)
        input_worker.start()
        process_workers = [
            Thread(target=self._process) for n in range(self._worker_count)
        ]
        for worker in process_workers:
            worker.start()
        output_worker = Thread(target=self._handle_output)
        output_worker.start()

        while not self.done_outputting():
            time.sleep(0.1)
            progress(self.stats)

        input_worker.join()
        for worker in process_workers:
            worker.join()
        output_worker.join()
        progress(self.stats)
