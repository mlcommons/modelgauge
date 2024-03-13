import json
from newhelm.annotators.llama_guard_annotator import LlamaGuardAnnotation
from pydantic import BaseModel
from typing import List, Dict, Mapping

from newhelm.aggregations import mean_of_measurement
from newhelm.base_annotator import BaseAnnotator
from newhelm.base_test import BasePromptResponseTest, Result, TestMetadata
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData, WebData
from newhelm.prompt import TextPrompt, SUTOptions
from newhelm.record_init import record_init
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItemAnnotations,
    TestItem,
)
from newhelm.test_registry import TESTS


class DiscrimEvalTestItemContext(BaseModel):
    age: int
    gender: str
    race: str


class DiscrimEval(BasePromptResponseTest):
    @record_init
    def __init__(self, prompt_type: str):
        assert prompt_type in ["explicit", "implicit"], f"Invalid prompt_type"
        self.prompt_type = prompt_type

    def get_metadata(self) -> TestMetadata:
        return TestMetadata(
            name=f"DiscrimEval-{self.prompt_type}",
            description=f"Evaluate potential discriminatory impact with {self.prompt_type} demographics.",
        )

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        return {
            f"prompts_jsonl": WebData(
                source_url=f"https://huggingface.co/datasets/Anthropic/discrim-eval/raw/main/{self.prompt_type}.jsonl"
            ),
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:

        test_items: List[TestItem] = []
        data_file = dependency_helper.get_local_path("prompts_jsonl")
        with open(data_file, "r") as f:
            for line in f:
                parsed_line = json.loads(line)
                prompt = PromptWithContext(
                    prompt=TextPrompt(
                        text=parsed_line["filled_template"],
                        options=SUTOptions(max_tokens=1),
                    ),
                    source_id=parsed_line["decision_question_id"],
                )
                test_items.append(
                    TestItem(
                        prompts=[prompt],
                        context=DiscrimEvalTestItemContext(
                            age=parsed_line["age"],
                            gender=parsed_line["gender"],
                            race=parsed_line["race"],
                        ),
                    )
                )
        return test_items

    def get_annotators(self) -> Mapping[str, BaseAnnotator]:
        return {}

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        # TODO: Replace placeholders
        logit_yes = 0.0
        logit_no = 0.0
        return {"logit_yes": logit_yes, "logit_no": logit_no}

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        # TODO
        return []


TESTS.register(
    "discrim_eval-explicit",
    DiscrimEval,
    "explicit",
)
TESTS.register(
    "discrim_eval-implicit",
    DiscrimEval,
    "implicit",
)
