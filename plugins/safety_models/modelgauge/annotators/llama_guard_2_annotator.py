from typing import List
from modelgauge.annotators.llama_guard_annotator import (
    LlamaGuardAnnotator,
    LlamaGuardData,
    LlamaGuardVersion,
    translate_llama_guard_response,
)
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherCompletionsRequest,
    TogetherCompletionsResponse,
)
from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.prompt import ChatMessage, ChatPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion


class LlamaGuard2Annotator(CompletionAnnotator[SafetyModelResponse]):
    def __init__(
        self,
        uid: str,
        together_api_key: TogetherApiKey,
    ):
        super().__init__(uid)
        self.llama_guard_annotator = LlamaGuardAnnotator(
            uid=uid,
            together_api_key=together_api_key,
            llama_guard_version=LlamaGuardVersion.VERSION_2_APRIL_2024,
        )

    def translate_request(self, prompt: PromptWithContext, completion: SUTCompletion):
        """Convert the prompt+completion into the native representation for this annotator."""
        if isinstance(prompt.prompt, ChatPrompt):
            return TogetherCompletionsRequest(
                prompt=_formatter(
                    prompt.prompt, self.llama_guard_annotator.llama_guard_data
                ),
                model=self.llama_guard_annotator.model_name,
                # This might need to be dynamic if the decoder is complicated.
                max_tokens=20,
                n=1,
            )
        else:
            raise ValueError(
                f"Unsupported prompt type. Can only support ChatPrompt. Type provided: {type(prompt.prompt)}"
            )

    def annotate(self, annotation_request: TogetherCompletionsRequest):
        """Perform annotation and return the raw response from the annotator."""
        return self.llama_guard_annotator.annotate(annotation_request)

    def translate_response(
        self, request: TogetherCompletionsRequest, response: TogetherCompletionsResponse
    ) -> SafetyModelResponse:
        result = self.llama_guard_annotator.translate_response(request, response)

        safety_response = SafetyModelResponse(
            is_safe=result.is_safe,
            safety_categories=result.violation_categories,
            reasoning=None,
        )

        return safety_response


def _formatter(chat_prompt: ChatPrompt, llama_guard: LlamaGuardData):
    """WARNING this formatter is currently not compatible with the original LlamaGuardAnnotator.

    This formatter takes several messages whereas the original LlamaGuardAnnotator only takes one message.
    TODO add support for multiple messages in the original LlamaGuardAnnotator and consolidate implementations.
    """
    categories_str = "\n".join(
        [
            f"{shorthand}: {category.name}"
            for shorthand, category in llama_guard.category_map.items()
        ]
    )

    conversations_str = format_chat(chat_prompt, user_role="User", sut_role="Agent")
    return llama_guard.template.substitute(
        agent_type=llama_guard.agent,
        categories=categories_str,
        conversations=conversations_str,
    )


ANNOTATORS.register(LlamaGuard2Annotator, "llama_guard_2", InjectSecret(TogetherApiKey))
