import os
from modelgauge.prompt import ChatPrompt, TextPrompt, ChatMessage, ChatRole
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.annotators.llama_guard_annotator import (
    LlamaGuardAnnotator,
    LlamaGuardAnnotation,
)
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherCompletionsResponse,
    TogetherCompletionsRequest,
)
from typing import List


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class LlamaGuard2SUT(
    PromptResponseSUT[TogetherCompletionsRequest, TogetherCompletionsResponse]
):
    """This SUT demonstrates the bare minimum behavior of a SUT: Use the input Prompt to determine the response."""

    def __init__(self, uid: str, together_api_key: TogetherApiKey):
        super().__init__(uid)
        self.llama_guard_client = LlamaGuardAnnotator(together_api_key)

    def translate_text_prompt(self, prompt: TextPrompt) -> TogetherCompletionsRequest:
        no_op_prompt = PromptWithContext(prompt=TextPrompt(text=""), source_id="")
        return self.llama_guard_client.translate_request(
            prompt=no_op_prompt, completion=SUTCompletion(text=prompt.text)
        )

    def translate_chat_prompt(self, prompt: ChatPrompt) -> TogetherCompletionsRequest:
        completion_input = self._messages_to_str(prompt.messages)
        no_op_prompt = PromptWithContext(prompt=TextPrompt(text=""), source_id="")
        return self.llama_guard_client.translate_request(
            prompt=no_op_prompt, completion=SUTCompletion(text=completion_input)
        )

    def evaluate(
        self, request: TogetherCompletionsRequest
    ) -> TogetherCompletionsResponse:
        return self.llama_guard_client.annotate(request)

    def translate_response(
        self, request: TogetherCompletionsRequest, response: TogetherCompletionsResponse
    ) -> SUTResponse:
        return SUTResponse(completions=[SUTCompletion(text=response.choices[0].text)])

    def _messages_to_str(self, messages: List[ChatMessage]) -> str:
        return "\n".join([self._combine_role_and_text(m) for m in messages])

    def _combine_role_and_text(self, m: ChatMessage) -> str:
        role_name = ""

        if m.role == ChatRole.user:
            role_name = "User"
        elif m.role == ChatRole.sut:
            role_name = "Assistant"
        else:
            raise ValueError(f"Bad role: {m.role}. Valid roles are 'user' and 'sut'.")

        return f"{role_name}: {m.text}"


# TODO move the os env var into a constant
SUTS.register(
    LlamaGuard2SUT,
    "lg2",
    together_api_key=InjectSecret(TogetherApiKey),
)
