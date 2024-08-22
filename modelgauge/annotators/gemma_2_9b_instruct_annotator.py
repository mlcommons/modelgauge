import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.prompt_engineered_annotator import (
    PromptEngineeredAnnotator,
    PromptEngineeredAnnotatorConfig,
    TogetherModelConfig,
)
from modelgauge.safety_model_constants import ASSISTANT_ROLE
from modelgauge.safety_model_prompts import (
    LABELER_SYS_PROMPT,
    LABELER_USER_PROMPT,
    MLC_TAXONOMY_0_5,
)
from modelgauge.suts.together_client import TogetherApiKey

GEMMA_2_9B_CONFIG = PromptEngineeredAnnotatorConfig(
    name="gemma_2_9b",
    llm_config=TogetherModelConfig(
        path="google/gemma-2-9b-it",
        api_key=TogetherApiKey(os.environ.get("TOGETHER_API_KEY")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT,
    labeler_user_prompt=LABELER_USER_PROMPT,
    taxonomy=MLC_TAXONOMY_0_5,
    role_to_measure=ASSISTANT_ROLE,
)

ANNOTATORS.register(
    PromptEngineeredAnnotator, GEMMA_2_9B_CONFIG.name, GEMMA_2_9B_CONFIG
)
