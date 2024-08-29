import json
from typing import Union

from pydantic import BaseModel

from modelgauge.annotator import CompletionAnnotator
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, SUTOptions, TextPrompt
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
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherChatRequest,
    TogetherChatResponse,
    TogetherChatSUT,
)

# FIXME - To use the openai GPT-4 annotator, add the "ChatRole.system: _SYSTEM_ROLE", role into the _ROLE_MAP in modelgauge/plugins/openai/modelgauge/suts/openai_client.py:
# LINE 32:
#
# _ROLE_MAP = {
#     ChatRole.user: _USER_ROLE,
#     ChatRole.sut: _ASSISTANT_ROLE,
#     ChatRole.system: _SYSTEM_ROLE,   ##### This line needs to be added into _ROLE_MAP
# }


class BaseLLMConfig(BaseModel):
    path: str
    max_tokens: int = 4000


class TogetherModelConfig(BaseLLMConfig):
    api_key: TogetherApiKey


class OpenAIModelConfig(BaseLLMConfig):
    api_key: OpenAIApiKey
    org_id: OpenAIOrgId


class PromptEngineeredAnnotatorConfig(BaseModel):
    name: str
    # FIXME - in the future, model_config should be a union eg Union[ConfigA, ConfigB]
    llm_config: Union[TogetherModelConfig, OpenAIModelConfig]
    labeler_system_prompt: str
    labeler_user_prompt: str
    taxonomy: str
    role_to_measure: str
    pass


class PromptEngineeredAnnotator(CompletionAnnotator[SafetyModelResponse]):
    """Annotator using prompt engineered Together model for hazard and safety classification"""

    def __init__(
        self,
        uid: str,
        config: PromptEngineeredAnnotatorConfig,
    ):
        super().__init__(uid)
        self.config = config

        if isinstance(self.config.llm_config, TogetherModelConfig):
            self.model = TogetherChatSUT(
                uid, self.config.llm_config.path, self.config.llm_config.api_key
            )
        elif isinstance(self.config.llm_config, OpenAIModelConfig):
            self.model = OpenAIChat(
                "annotator",
                "gpt-4",
                self.config.llm_config.api_key,
                self.config.llm_config.org_id,
            )
        else:
            raise ValueError(
                f"Unsupported LLM config type: {type(self.config.llm_config)}"
            )

    def translate_request(self, prompt: PromptWithContext, completion: SUTCompletion):
        """Convert the prompt+completion into the native representation for this annotator."""
        system_prompt = ChatMessage(
            text=self.config.labeler_system_prompt.format(
                taxonomy=self.config.taxonomy
            ),
            role=ChatRole.system,
        )

        # We have a list of interaction messages, simplest option is to turn into 'role': 'message' block of text
        if isinstance(prompt.prompt, ChatPrompt):
            chat_interactions_to_annotate = flatten_conversation_to_str(prompt.prompt)
        elif isinstance(prompt.prompt, TextPrompt):
            assert prompt.prompt.text is not None, "TextPrompt must have text."
            assert completion.text is not None, "SUTCompletion must have text."
            chat_interactions_to_annotate = (
                f"User: {prompt.prompt.text}\n\nAssistant: {completion.text}"
            )
        else:
            raise ValueError(
                f"Can only handle ChatPrompt, TextPrompt types. Type given: {type(prompt.prompt)}"
            )

        request = self.model.translate_chat_prompt(
            prompt=ChatPrompt(
                messages=[
                    system_prompt,
                    ChatMessage(
                        text=self.config.labeler_user_prompt.format(
                            messages=chat_interactions_to_annotate,
                            role_to_measure=self.config.role_to_measure,
                        ),
                        role=ChatRole.user,
                    ),
                ],
                options=SUTOptions(max_tokens=self.config.llm_config.max_tokens),
            )
        )

        return request

    def annotate(self, annotation_request):
        """Perform annotation and return the raw response from the annotator."""
        result = self.model.evaluate(annotation_request)
        return result

    def translate_response(
        self, request: TogetherChatRequest, response: TogetherChatResponse
    ) -> SafetyModelResponse:
        """Convert the raw response into the form read by Tests."""

        response_text = response.choices[0].message.content

        # Parse response
        try:
            response_json = json.loads(response_text)
            result = SafetyModelResponse(**response_json)
            result.total_tokens_used = response.usage.total_tokens
            return result
        except Exception as e:
            print(f"Error parsing response JSON: {e}")
            return SafetyModelResponse(
                is_safe=False,
                safety_categories=[],
                is_valid=False,
            )
