import json
import os
from typing import List, Mapping
from newhelm.aggregations import mean_of_measurement
from newhelm.base_test import BasePromptResponseTest, TestMetadata
from newhelm.data_packing import ZipPacker
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData, WebData
from newhelm.placeholders import Measurement, Prompt, Result
from newhelm.single_turn_prompt_response import (
    AnnotatedTestItem,
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)


class DemoSimpleQATest(BasePromptResponseTest):
    @classmethod
    def get_metadata(cls) -> TestMetadata:
        return TestMetadata(
            "DemoSimpleQATest",
            "This test is a demonstration of how to create very simple "
            + "question and answer based Test using external data.",
        )

    @classmethod
    def get_dependencies(cls) -> Mapping[str, ExternalData]:
        """Specify all the external dependencies needed to run this Test."""
        return {
            # The keys can be arbitrary, they are used to decide where to store
            # the dependency locally and when you look up the dependency in make_test_items.
            "jsonl_questions": WebData(
                source_url="https://storage.googleapis.com/mlc-aisafety-scratch-dev-7292542/an_example.jsonl"
            ),
            "zipped_questions": WebData(
                source_url="https://storage.googleapis.com/mlc-aisafety-scratch-dev-7292542/question_answer.zip",
                unpacker=ZipPacker(),
            ),
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        """Read questions from our two dependencies and convert them into TestItems."""
        test_items: List[TestItem] = []
        # Read some questions from a jsonl file.
        with open(dependency_helper.get_local_path("jsonl_questions"), "r") as f:
            for line in f.readlines():
                if not line.strip():
                    # Skip empty lines
                    continue
                data = json.loads(line)
                prompt = PromptWithContext(
                    # The Prompt will go to the SUT, the context will not.
                    Prompt(data["question"]),
                    context=data["safe_answer"],
                )
                test_items.append(TestItem([prompt]))
        # Read some more questions, this time from a zip file containing separate question
        # and answer files. Because we specified an unpacker get_dependencies,
        # this points at a directory.
        zip_dir = dependency_helper.get_local_path("zipped_questions")
        with open(os.path.join(zip_dir, "questions.txt"), "r") as f:
            questions = f.readlines()
        with open(os.path.join(zip_dir, "answers.txt"), "r") as f:
            answers = f.readlines()
        for question, answer in zip(questions, answers):
            if not question.strip() or not answer.strip():
                # Skip empty lines
                continue
            prompt = PromptWithContext(Prompt(question), context=answer)
            test_items.append(TestItem([prompt]))
        return test_items

    def measure_quality(self, item: AnnotatedTestItem) -> List[Measurement]:
        """Use the TestItem context to report how well the SUT did."""
        # This Test only uses a single Prompt per TestItem, so only 1 interaction.
        interaction = item.item_with_interactions.interactions[0]
        # For simplicity, this demo assumes there is only one safe answer.
        gave_safe_answer = interaction.response.completion == interaction.prompt.context
        return [Measurement("gave_safe_answer", 1 if gave_safe_answer else 0)]

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        """Combine the Measurements from measure_quality into Results."""
        return [
            Result(
                "gave_safe_answer_rate",
                value=mean_of_measurement("gave_safe_answer", items),
            )
        ]
