import concurrent
import csv
import datetime
import pathlib
import queue
import time
from collections import defaultdict
from threading import Event, Thread

import click

from modelgauge.config import load_secrets_from_config, raise_if_missing_from_config
from modelgauge.load_plugins import load_plugins
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import MissingSecretValues
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


class PromptInput:
    def __init__(self, path):
        super().__init__()
        self.path = path

    def __iter__(self):
        with open(self.path, newline="") as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                yield PromptItem(row)


class PromptOutput:
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
        pass

    def write(self, item: PromptItem, results):
        row = [item.uid(), item.prompt()]
        for k in self.suts:
            if k in results:
                row.append(results[k])
            else:
                row.append("")
        self.writer.writerow(row)


class ParallelPromptRunner:
    def __init__(self, input, output, suts):
        super().__init__()
        self.input = input
        self.output = output
        self.suts = suts
        self._worker_count = 25 * len(suts)

    def run(self):
        to_process = queue.Queue(maxsize=self._worker_count * 4)
        to_output = queue.Queue(maxsize=self._worker_count * 2)
        unfinished = defaultdict(lambda: dict())
        input_done = Event()

        def read():
            for item in self.input:
                for sut_key in self.suts:
                    to_process.put((item, sut_key))
            input_done.set()

        def process():
            while (not input_done.is_set()) or (not to_process.empty()):
                item, sut_key = to_process.get()
                request = self.suts[sut_key].translate_text_prompt(
                    TextPrompt(text=item.prompt())
                )
                response = self.suts[sut_key].evaluate(request)
                result = self.suts[sut_key].translate_response(request, response)

                to_output.put((item, sut_key, result.completions[0].text))
                to_process.task_done()

        def handle_output():
            with self.output as output:
                while (
                    (not input_done.is_set())
                    or (not to_process.empty())
                    or not (to_output.empty())
                ):
                    item, sut_key, response = to_output.get()
                    print(f"outputting {item}")
                    unfinished[item][sut_key] = response
                    if len(unfinished[item]) == len(self.suts):
                        output.write(item, unfinished[item])
                        del unfinished[item]
                        to_output.task_done()
            print(
                f"worker done: (not {input_done.is_set()}) and (not {to_process.empty()}) and not ({to_output.empty()})"
            )

        input_worker = Thread(target=read)
        input_worker.start()
        process_workers = [Thread(target=process) for n in range(self._worker_count)]
        for worker in process_workers:
            worker.start()
        output_worker = Thread(target=handle_output)
        output_worker.start()

        def status():
            return f"to_process: {to_process.qsize()} to_output: {to_output.qsize()} unfinished: {len(unfinished)} workers: {input_worker} {process_workers} {output_worker}"

        print(f"off to the races: {status()}")
        while not input_done.wait(1):
            print(f"working: {status()}")
            time.sleep(0.5)
        print(f"input all loaded: {status()}")
        input_worker.join()
        print(f"input finished: {status()}")

        for worker in process_workers:
            worker.join()
            print(f"{worker} done")
        output_worker.join()
        print("output complete")


input


@click.command()
@click.option(
    "sut_names",
    "-s",
    "--sut",
    help="Which registered SUT to run.",
    multiple=True,
    required=True,
)
@click.argument("filename", type=click.Path(exists=True))
def cli(sut_names, filename):
    load_plugins()
    secrets = load_secrets_from_config()

    try:
        suts = {
            sut_name: SUTS.make_instance(sut_name, secrets=secrets)
            for sut_name in sut_names
        }
    except MissingSecretValues as e:
        raise_if_missing_from_config([e])

    path = pathlib.Path(filename)
    input = PromptInput(path)
    output = PromptOutput(
        pathlib.Path(
            path.stem
            + "-responses-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            + ".csv"
        ),
        suts,
    )
    runner = ParallelPromptRunner(input, output, suts)
    runner.run()


@click.command()
@click.option("--sut", help="Which registered SUT to run.", required=True)
@click.argument("filename", type=click.Path(exists=True))
def fnord(sut, filename):
    load_plugins()
    secrets = load_secrets_from_config()

    # TODO Add multiple SUTS and correlating output
    # TODO Retry on error
    # TODO maybe convert to an object so state is less insane?
    try:
        sut_obj = SUTS.make_instance(sut, secrets=secrets)
    except MissingSecretValues as e:
        raise_if_missing_from_config([e])

    def process_prompt(prompt):
        request = sut_obj.translate_text_prompt(TextPrompt(text=prompt))
        response = sut_obj.evaluate(request)
        result = sut_obj.translate_response(request, response)
        return result.completions[0].text

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        future_to_row = {}
        with open(filename, newline="") as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                future_to_row[executor.submit(process_prompt, row["Text"])] = row
        with open("result.csv", "w", newline="") as f:
            csvwriter = csv.writer(f, quoting=csv.QUOTE_ALL)
            csvwriter.writerow(["UID", "Text", sut])
            for future in concurrent.futures.as_completed(future_to_row):
                row = future_to_row[future]

                try:
                    response = future.result()
                except Exception as e:
                    print(f"Unexpected failure for {row['UID']}: {e}")
                else:
                    csvwriter.writerow([row["UID"], row["Text"], response])
                    print(f"{row['UID']}: {row['Text']} -> {response}")


if __name__ == "__main__":
    cli()
