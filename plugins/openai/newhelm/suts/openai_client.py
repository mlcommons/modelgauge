from typing import Dict, List, Optional, Sequence, Union

from pydantic import BaseModel
from newhelm.prompt import ChatPrompt, ChatRole, SUTOptions, TextPrompt
from newhelm.record_init import record_init
from newhelm.secret_values import SecretValues, UseSecret
from newhelm.secrets_registry import SECRETS
from newhelm.sut import SUTCompletion, PromptResponseSUT, SUTResponse
from openai import OpenAI
from openai.types.chat import ChatCompletion

from newhelm.sut_registry import SUTS

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"
_TOOL_ROLE = "tool_call_id"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
}


class OpenAIChatMessage(BaseModel):
    content: str
    role: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[OpenAIChatMessage]
    model: str
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    # How many chat completion choices to generate for each input message.
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List] = None
    tool_choice: Optional[Union[str, Dict]] = None
    user: Optional[str] = None


class OpenAIChat(PromptResponseSUT[OpenAIChatRequest, ChatCompletion]):
    """
    Documented at https://platform.openai.com/docs/api-reference/chat/create
    """

    @record_init
    def __init__(self, model: str):
        self.model = model
        self.client: Optional[OpenAI] = None

    def get_used_secrets(self) -> Sequence[UseSecret]:
        return [
            UseSecret(
                scope="openai",
                key="api_key",
                required=True,
                instructions="See https://platform.openai.com/api-keys",
            ),
            UseSecret(
                scope="openai",
                key="org_id",
                required=False,
                instructions="See https://platform.openai.com/account/organization",
            ),
        ]

    def load(self, secrets: SecretValues):
        self.client = OpenAI(
            api_key=secrets.get_required("openai", "api_key"),
            organization=secrets.get_optional("openai", "org_id"),
        )

    def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIChatRequest:
        messages = [OpenAIChatMessage(content=prompt.text, role=_USER_ROLE)]
        return self._translate_request(messages, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(
                OpenAIChatMessage(content=message.text, role=_ROLE_MAP[message.role])
            )
        return self._translate_request(messages, prompt.options)

    def _translate_request(
        self, messages: List[OpenAIChatMessage], options: SUTOptions
    ):
        return OpenAIChatRequest(
            messages=messages,
            model=self.model,
            frequency_penalty=options.frequency_penalty,
            max_tokens=options.max_tokens,
            n=options.num_completions,
            presence_penalty=options.presence_penalty,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
        )

    def evaluate(self, request: OpenAIChatRequest) -> ChatCompletion:
        assert self.client is not None, "Must call load() before evaluate()"
        request_dict = request.model_dump(exclude_none=True)
        return self.client.chat.completions.create(**request_dict)

    def translate_response(
        self, request: OpenAIChatRequest, response: ChatCompletion
    ) -> SUTResponse:
        completions = []
        for choice in response.choices:
            text = choice.message.content
            assert text is not None
            completions.append(SUTCompletion(text=text))
        return SUTResponse(completions=completions)


SUTS.register("gpt-3.5-turbo", OpenAIChat, "gpt-3.5-turbo")
