import json
from typing import List
from modelgauge.safety_model_constants import DEFAULT_MAX_CHAT_TOKENS
from modelgauge.safety_model_prompts import (
    LABELER_SYS_PROMPT,
    LABELER_USER_PROMPT,
    MLC_TAXONOMY_0_5,
)
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherChatRequest,
    TogetherChatResponse,
    TogetherChatSUT,
)
from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, SUTOptions
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.sut import SUTCompletion

ANNOTATOR_UID = "llama_3_70b_chat_safety_annotator"
LLAMA_3_70B_MODEL_PATH = "meta-llama/Llama-3-70b-chat-hf"


class Llama3_70bChatAnnotator(CompletionAnnotator[SafetyModelResponse]):
    """Annotator simply uses a prompt with Llama 3 70b to do hazard and safety classification"""

    def __init__(
        self,
        uid: str,
        together_api_key: TogetherApiKey,
    ):
        super().__init__(uid)
        self.model_name = None
        self.model = TogetherChatSUT(uid, LLAMA_3_70B_MODEL_PATH, together_api_key)

    def translate_request(self, prompt: PromptWithContext, completion: SUTCompletion):
        """Convert the prompt+completion into the native representation for this annotator."""

        system_prompt = ChatMessage(
            text=LABELER_SYS_PROMPT.format(taxonomy=MLC_TAXONOMY_0_5),
            role=ChatRole.system,
        )

        # We have a list of interaction messages, simplest option is to turn into 'role': 'message' block of text
        if isinstance(prompt.prompt, ChatPrompt):
            chat_interactions_to_annotate = self._messages_to_str(
                prompt.prompt.messages
            )
        else:
            raise ValueError(
                f"Can only handle ChatPrompt type. Type given: {type(prompt.prompt)}"
            )

        request = self.model.translate_chat_prompt(
            prompt=ChatPrompt(
                messages=[
                    system_prompt,
                    ChatMessage(
                        text=chat_interactions_to_annotate,
                        role=ChatRole.user,
                    ),
                ],
                options=SUTOptions(max_tokens=DEFAULT_MAX_CHAT_TOKENS),
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
            return SafetyModelResponse(**response_json)
        except Exception as e:
            print(f"Error parsing response JSON: {e}")
            return SafetyModelResponse(
                is_safe=False,
                safety_categories=[],
                is_valid=False,
            )

    def _messages_to_str(self, messages: List[ChatMessage]) -> str:
        def _role_to_str(role: ChatRole) -> str:
            if role == ChatRole.user:
                return "User"
            elif role == ChatRole.sut:
                return "Assistant"
            else:
                raise ValueError(f"Unknown or invalid role: {role}")

        return "\n".join([f"{_role_to_str(m.role)}: {m.text}" for m in messages])


ANNOTATORS.register(
    Llama3_70bChatAnnotator, ANNOTATOR_UID, InjectSecret(TogetherApiKey)
)
