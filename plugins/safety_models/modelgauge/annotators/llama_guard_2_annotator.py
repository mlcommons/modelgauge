from modelgauge.annotators.llama_guard_annotator import (
    LlamaGuardAnnotator,
    LlamaGuardData,
    LlamaGuardVersion,
)
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.safety_model_utils import flatten_conversation_to_str
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherCompletionsRequest,
    TogetherCompletionsResponse,
)
from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.prompt import ChatPrompt, TextPrompt
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
                prompt=_chat_prompt_formatter(
                    prompt.prompt, self.llama_guard_annotator.llama_guard_data
                ),
                model=self.llama_guard_annotator.model_name,
                # This might need to be dynamic if the decoder is complicated.
                max_tokens=20,
                n=1,
            )
        elif isinstance(prompt.prompt, TextPrompt):
            assert prompt.prompt.text is not None, "TextPrompt must have text."
            assert completion.text is not None, "SUTCompletion must have text."
            return TogetherCompletionsRequest(
                prompt=_text_prompt_formatter(
                    prompt=prompt.prompt.text,
                    completion=completion.text,
                    llama_guard=self.llama_guard_annotator.llama_guard_data,
                ),
                model=self.llama_guard_annotator.model_name,
                # This might need to be dynamic if the decoder is complicated.
                max_tokens=20,
                n=1,
            )
        else:
            raise ValueError(
                f"Unsupported prompt type. Can only support ChatPrompt or TextPrompt. Type provided: {type(prompt.prompt)}"
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


def _chat_prompt_formatter(chat_prompt: ChatPrompt, llama_guard: LlamaGuardData):
    """WARNING this formatter is currently not compatible with the original LlamaGuardAnnotator.

    This formatter takes several messages whereas the original LlamaGuardAnnotator only takes one message.
    TODO add support for multiple messages in the original LlamaGuardAnnotator and consolidate implementations.
    """
    conversations_str = flatten_conversation_to_str(
        chat_prompt, user_role="User", sut_role="Agent"
    )
    return llama_guard.template.substitute(
        agent_type=llama_guard.agent,
        categories=_get_categories_str(llama_guard),
        conversations=conversations_str,
    )


def _text_prompt_formatter(prompt: str, completion: str, llama_guard: LlamaGuardData):
    conversations_str = f"User: {prompt}\n\nAgent: {completion}"
    return llama_guard.template.substitute(
        agent_type=llama_guard.agent,
        categories=_get_categories_str(llama_guard),
        conversations=conversations_str,
    )


def _get_categories_str(llama_guard: LlamaGuardData):
    return "\n".join(
        [
            f"{shorthand}: {category.name}"
            for shorthand, category in llama_guard.category_map.items()
        ]
    )


ANNOTATORS.register(LlamaGuard2Annotator, "llama_guard_2", InjectSecret(TogetherApiKey))
