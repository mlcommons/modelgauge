import json
import os
from typing import Callable, Dict, Optional

from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.prompt_engineered_annotator import (
    OpenAIModelConfig,
    PromptEngineeredAnnotator,
    PromptEngineeredAnnotatorConfig,
)
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, SUTOptions, TextPrompt
from modelgauge.safety_model_constants import ASSISTANT_ROLE
from modelgauge.safety_model_prompts import (
    LABELER_SYS_PROMPT,
    LABELER_USER_PROMPT,
    MLC_TAXONOMY_0_5,
)
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.safety_model_utils import flatten_conversation_to_str
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.suts.openai_client import (
    OpenAIApiKey,
    OpenAIChat,
    OpenAIChatRequest,
    OpenAIOrgId,
)

# FIXME - To use the openai GPT-4 annotator, add the "ChatRole.system: _SYSTEM_ROLE", role into the _ROLE_MAP in modelgauge/plugins/openai/modelgauge/suts/openai_client.py:
# LINE 32:
#
# _ROLE_MAP = {
#     ChatRole.user: _USER_ROLE,
#     ChatRole.sut: _ASSISTANT_ROLE,
#     ChatRole.system: _SYSTEM_ROLE,   ##### This line needs to be added into _ROLE_MAP
# }

GPT_4_CONFIG = PromptEngineeredAnnotatorConfig(
    name="gpt-4",
    llm_config=OpenAIModelConfig(
        path="gpt-4",
        api_key=OpenAIApiKey(os.environ.get("OPENAI_API_KEY")),
        org_id=OpenAIOrgId(os.environ.get("OPENAI_ORG_ID")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT,
    labeler_user_prompt=LABELER_USER_PROMPT,
    taxonomy=MLC_TAXONOMY_0_5,
    role_to_measure=ASSISTANT_ROLE,
)


ANNOTATORS.register(PromptEngineeredAnnotator, GPT_4_CONFIG.name, GPT_4_CONFIG)
