import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.llama_guard_2_lora_annotator import (
    LlamaGuard2LoRAAnnotator,
    LlamaGuard2LoRAConfig,
)
from modelgauge.annotators.llama_guard_annotator import SafetyCategory

# Arbitrary name to register annotator with modelgauge
MODELGAUGE_ANNOTATOR_NAME = "sample-lg2-lora"
# Update the following endpoint to point to the deployed vllm
DEPLOYED_VLLM_ENDPOINT_URL = "https://change.me"
# This value is NOT arbitrary. The docker image sets this value
DEPLOYED_LORA_MODEL_NAME = "llamaguard-lora"
# Define Llama Guard 2 style safety categories
SAFETY_CATEGORIES = [
    SafetyCategory("Violent Crimes. ", ""),
    SafetyCategory("Non-Violent Crimes. ", ""),
    SafetyCategory("Sex Crimes. ", ""),
    SafetyCategory("Child Exploitation. ", ""),
    SafetyCategory("Specialized Advice. ", ""),
    SafetyCategory("Privacy. ", ""),
    SafetyCategory("Intellectual Property. ", ""),
    SafetyCategory("Indiscriminate Weapons. ", ""),
    SafetyCategory("Hate. ", ""),
    SafetyCategory("Self-Harm. ", ""),
    SafetyCategory("Sexual Content. ", ""),
]

config = LlamaGuard2LoRAConfig(
    name=MODELGAUGE_ANNOTATOR_NAME,
    model=DEPLOYED_LORA_MODEL_NAME,
    base_url=DEPLOYED_VLLM_ENDPOINT_URL,
    api_key=os.getenv("VLLM_LOCAL_API_KEY", ""),
    safety_categories=SAFETY_CATEGORIES,
)

ANNOTATORS.register(LlamaGuard2LoRAAnnotator, config.name, config)
