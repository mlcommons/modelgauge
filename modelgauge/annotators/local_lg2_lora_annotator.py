import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.llama_guard_2_lora_annotator import (
    LlamaGuard2LoRAAnnotator,
    LlamaGuard2LoRAConfig,
)
from modelgauge.safety_model_constants import DEFAULT_MINIMAL_SAFETY_CATEGORIES

# Arbitrary name to register annotator with modelgauge
MODELGAUGE_ANNOTATOR_NAME = "local-lg2-lora"
# Update the following endpoint to point to the deployed vllm
DEPLOYED_VLLM_ENDPOINT_URL = "http://localhost:8000/v1"
# This value is NOT arbitrary. The docker image sets this value
DEPLOYED_LORA_MODEL_NAME = "llamaguard-lora"

config = LlamaGuard2LoRAConfig(
    name=MODELGAUGE_ANNOTATOR_NAME,
    model=DEPLOYED_LORA_MODEL_NAME,
    base_url=DEPLOYED_VLLM_ENDPOINT_URL,
    api_key=os.getenv("VLLM_LOCAL_API_KEY", ""),
    safety_categories=DEFAULT_MINIMAL_SAFETY_CATEGORIES,
)

ANNOTATORS.register(LlamaGuard2LoRAAnnotator, config.name, config)
