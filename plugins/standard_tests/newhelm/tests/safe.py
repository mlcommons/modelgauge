import csv
from typing import List, Dict, Mapping

from newhelm.annotators.llama_guard_annotator import (
    LlamaGuardAnnotation,
    LlamaGuardAnnotator,
)
from pydantic import BaseModel
from newhelm.aggregations import mean_of_measurement
from newhelm.base_annotator import BaseAnnotator
from newhelm.base_test import BasePromptResponseTest, TestMetadata
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData, GDriveData
from newhelm.prompt import TextPrompt, SUTOptions
from newhelm.secret_values import InjectSecret
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItemAnnotations,
    TestItem,
)
from newhelm.suts.together_client import TogetherApiKey
from newhelm.test_decorator import newhelm_test
from newhelm.test_registry import TESTS


HAZARDS_MAPPING = {
    "cae": "Child sexual exploitation",
    "cbr": "Chemical, biological, radioloigcal, nuclear and high-yield explosives weapons",
    "hat": "Hate",
    "nvc": "Non-violent crimes",
    "sc": "Sexual content",
    "ssh": "Suicide & self-harm",
    "ter": "Terrorism",
    "vio": "Violent crimes",
}
_PERSONA_TYPES = ["typical", "malicious", "vulnerable"]


class SafeTestResult(BaseModel):
    frac_safe: float


@newhelm_test()
class SafeTest(BasePromptResponseTest):
    def __init__(
        self,
        uid: str,
        hazard: str,
        folder_name: str,
        together_api_key: TogetherApiKey,
        persona_type: str = "typical",
    ):
        super().__init__(uid)
        assert hazard in HAZARDS_MAPPING, f"Invalid hazard type {hazard}."
        assert persona_type in _PERSONA_TYPES, f"Invalid persona type {persona_type}."
        self.hazard = hazard
        self.hazard_name = HAZARDS_MAPPING[hazard]
        self.folder_name = folder_name
        self.together_api_key = together_api_key
        self.persona_type = persona_type
        self.dependency_key = f"{self.folder_name}_{self.hazard}_{self.persona_type}"

    def get_metadata(self) -> TestMetadata:
        return TestMetadata(
            name=f"SafeTest",
            description=f"Safe AI with Flexible Evaluation for {self.hazard_name} with {self.persona_type} prompts",
        )

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        return {
            f"{self.dependency_key}": GDriveData(
                data_source="https://drive.google.com/drive/folders/1FNpd9KQ1MvylgVNuWx7O0byWoFDyLBeS",
                file_path=f"{self.folder_name}/mls_ais_wg_safe_{self.hazard}_{self.persona_type}-user.csv",
            ),
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        sut_options = SUTOptions(
            max_tokens=500,
            temperature=0.01,
        )
        test_items: List[TestItem] = []
        data_file = dependency_helper.get_local_path(self.dependency_key)
        with open(data_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                text = row["Prompt"].strip()
                if not text:
                    continue
                file_hazard = row["Hazard type shorthand"]
                assert (
                    file_hazard == self.hazard
                ), f"Expected {data_file} to have {self.hazard}, but had {file_hazard}."
                prompt = PromptWithContext(
                    prompt=TextPrompt(text=text, options=sut_options),
                    source_id=row["UID"],
                )
                test_items.append(TestItem(prompts=[prompt]))
        return test_items

    def get_annotators(self) -> Mapping[str, BaseAnnotator]:
        return {"llama_guard": LlamaGuardAnnotator(self.together_api_key)}

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        llama_guard_completion = (
            item.get_annotation("llama_guard", LlamaGuardAnnotation)
            .interactions[0]
            .completions[0]
        )
        return {"is_safe": float(llama_guard_completion.is_safe)}

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> SafeTestResult:
        """
        Returns the fraction [0.0, 1.0] of responses that were evaluated as safe.
        """
        return SafeTestResult(frac_safe=mean_of_measurement("is_safe", items))


FOLDER_NAME = "data_22mar2024"
API_KEY_SECRET = InjectSecret(TogetherApiKey)
TESTS.register(
    SafeTest,
    "safe-cae",
    "cae",
    FOLDER_NAME,
    API_KEY_SECRET,
)
TESTS.register(
    SafeTest,
    "safe-cae-mal",
    "cae",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_type="malicious",
)
TESTS.register(
    SafeTest,
    "safe-cbr-mal",
    "cbr",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_type="malicious",
)
TESTS.register(
    SafeTest,
    "safe-hat",
    "hat",
    FOLDER_NAME,
    API_KEY_SECRET,
)
TESTS.register(
    SafeTest,
    "safe-hat-mal",
    "hat",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_type="malicious",
)
TESTS.register(
    SafeTest,
    "safe-nvc",
    "nvc",
    FOLDER_NAME,
    API_KEY_SECRET,
)
TESTS.register(
    SafeTest,
    "safe-nvc-mal",
    "nvc",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_type="malicious",
)
TESTS.register(
    SafeTest,
    "safe-sc",
    "sc",
    FOLDER_NAME,
    API_KEY_SECRET,
)
TESTS.register(
    SafeTest,
    "safe-sc-mal",
    "sc",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_type="malicious",
)
TESTS.register(
    SafeTest,
    "safe-ssh",
    "ssh",
    FOLDER_NAME,
    API_KEY_SECRET,
)
TESTS.register(
    SafeTest,
    "safe-ssh-vul",
    "ssh",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_type="vulnerable",
)
TESTS.register(
    SafeTest,
    "safe-ter",
    "ter",
    FOLDER_NAME,
    API_KEY_SECRET,
)
TESTS.register(
    SafeTest,
    "safe-ter-mal",
    "ter",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_type="malicious",
)
TESTS.register(
    SafeTest,
    "safe-vio-mal",
    "vio",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_type="malicious",
)
