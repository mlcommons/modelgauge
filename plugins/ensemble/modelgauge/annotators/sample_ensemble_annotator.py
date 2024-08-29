import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.ensemble_annotator import EnsembleAnnotator
from modelgauge.annotators.llama_3_70b_chat_annotator import LLAMA_3_70B_CONFIG
from modelgauge.annotators.llama_guard_2_lora_annotator import (
    LlamaGuard2LoRAAnnotator,
    LlamaGuard2LoRAConfig,
)
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotator
from modelgauge.annotators.mistral_8x22b_instruct_annotator import MISTRAL_8x22B_CONFIG
from modelgauge.annotators.prompt_engineered_annotator import PromptEngineeredAnnotator
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole
from modelgauge.safety_model_constants import DEFAULT_MINIMAL_SAFETY_CATEGORIES

a1 = PromptEngineeredAnnotator(MISTRAL_8x22B_CONFIG.name, MISTRAL_8x22B_CONFIG)
a2 = PromptEngineeredAnnotator(LLAMA_3_70B_CONFIG.name, LLAMA_3_70B_CONFIG)

ANNOTATORS.register(EnsembleAnnotator, "ensemble", [a1, a2])
