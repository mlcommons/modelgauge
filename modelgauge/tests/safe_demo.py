"""
Each test registered in this file corresponds to a different hazard in the MLCommons taxonomy of hazards.

Version 0.5 hazards and tests:
    The hazards taxonomy is described in detail here: https://drive.google.com/file/d/1V8KFfk8awaAXc83nZZzDV2bHgPT8jbJY/view.
    Test UIDs follow the form "safe-{ hazard acronym }" (e.g. safe-vcr). The hazard acronyms are as follows:
    - vcr: Violent Crimes
    - nvc: Non-Violent Crimes
    - scr: Sex-Related Crimes
    - cae: Child Sexual Exploitation
    - cbr: Indiscriminate Weapons (CBRNE)
    - ssh: Suicide & Self-Harm
    - hat: Hate


"""

import csv
from enum import Enum
from pydantic import BaseModel
from typing import Dict, List, Mapping

from modelgauge.aggregations import get_measurement_stats_by_key
from modelgauge.annotator import Annotator
from modelgauge.annotators.ensemble_annotator import EnsembleAnnotator
from modelgauge.annotators.llama_3_70b_chat_annotator import LLAMA_3_70B_CONFIG
from modelgauge.annotators.mistral_8x22b_instruct_annotator import MISTRAL_8x22B_CONFIG
from modelgauge.annotators.prompt_engineered_annotator import PromptEngineeredAnnotator
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.safety_model_response import EnsembleSafetyModelResponse
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
    TestItemAnnotations,
)
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.suts.together_client import TogetherApiKey
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS
from modelgauge.tests.safe import (
    SafeTestItemContext,
    SafeTestResult,
    SafePersonas,
    PersonaResult,
)


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SafeTestDemo(PromptResponseTest):
    """Transitional Safe Test using v0.5 prompts and v1.0 annotators."""

    hazards = ["cae", "cbr", "nvc"]

    def __init__(
        self,
        uid: str,
        hazard: str,
        together_api_key: TogetherApiKey,
        persona_types: List[SafePersonas],
    ):
        super().__init__(uid)
        assert hazard in self.hazards, f"Invalid hazard type {hazard}."
        self.hazard = hazard
        self.together_api_key = together_api_key
        assert len(persona_types), "At least one persona type must be specified."
        assert len(set(persona_types)) == len(
            persona_types
        ), f"Must specify a unique set of persona types, but got {persona_types}"
        self.persona_types = persona_types

    def _persona_dependency_key(self, persona: SafePersonas) -> str:
        return f"data_04apr2024_{self.hazard}_{persona.value}"

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        modellab_base_download_url = (
            "https://ai-safety-dev.uc.r.appspot.com/files/download"
        )
        dependencies = {}
        for persona in self.persona_types:
            dependencies[self._persona_dependency_key(persona)] = WebData(
                source_url=f"{modellab_base_download_url}/mls_ais_wg_safe_{self.hazard}_{persona.value}-user.csv",
            )
        return dependencies

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        sut_options = SUTOptions(
            max_tokens=500,
            temperature=0.01,
        )
        test_items: List[TestItem] = []
        for persona in self.persona_types:
            data_file = dependency_helper.get_local_path(
                self._persona_dependency_key(persona)
            )
            with open(data_file, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    text = row["Prompt"].strip()
                    if not text:
                        continue
                    file_hazard = row["Hazard category shorthand"]
                    assert (
                        file_hazard == self.hazard
                    ), f"Expected {data_file} to have {self.hazard}, but had {file_hazard}."
                    prompt = PromptWithContext(
                        prompt=TextPrompt(text=text, options=sut_options),
                        source_id=row["UID"],
                    )
                    test_items.append(
                        TestItem(
                            prompts=[prompt],
                            context=SafeTestItemContext(persona_type=persona),
                        ),
                    )
        return test_items

    def get_annotators(self) -> Mapping[str, Annotator]:
        a1 = PromptEngineeredAnnotator(MISTRAL_8x22B_CONFIG.name, MISTRAL_8x22B_CONFIG)
        a2 = PromptEngineeredAnnotator(LLAMA_3_70B_CONFIG.name, LLAMA_3_70B_CONFIG)
        return {
            "ensemble": EnsembleAnnotator("ensemble", [a1, a2]),
        }

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        llama_guard_completion = (
            item.interactions[0]
            .response.completions[0]
            .get_annotation("ensemble", EnsembleSafetyModelResponse)
        )
        return {"is_safe": float(llama_guard_completion.is_safe)}

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> SafeTestResult:
        """Returns a SafeTestResult that maps each persona_type's subset of prompts to its corresponding measurements.

        Persona types that do not have any measured test items are not included.
        """
        measurement_stats = get_measurement_stats_by_key(
            "is_safe", items, key=lambda item: item.test_item.context.persona_type
        )
        persona_results = {}
        for persona in self.persona_types:
            if persona in measurement_stats:
                persona_results[persona.value] = PersonaResult(
                    frac_safe=measurement_stats[persona].mean,
                    num_items=measurement_stats[persona].count,
                )
        return SafeTestResult(persona_results=persona_results)


# TODO: VLLM secret. Maybe put to env variable.
API_KEY_SECRET = InjectSecret(TogetherApiKey)

TESTS.register(
    SafeTestDemo,
    "safe-cae-demo",
    "cae",
    API_KEY_SECRET,
    persona_types=[SafePersonas.TYPICAL, SafePersonas.MALICIOUS],
)
TESTS.register(
    SafeTestDemo,
    "safe-cbr-demo",
    "cbr",
    API_KEY_SECRET,
    persona_types=[SafePersonas.TYPICAL, SafePersonas.MALICIOUS],
)
TESTS.register(
    SafeTestDemo,
    "safe-nvc-demo",
    "nvc",
    API_KEY_SECRET,
    persona_types=[SafePersonas.TYPICAL, SafePersonas.MALICIOUS],
)
