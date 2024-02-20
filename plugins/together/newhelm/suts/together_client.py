from typing import Any, List, Optional
from pydantic import BaseModel, Field
import requests
from together.utils import response_status_exception  # type: ignore
from newhelm.prompt import ChatPrompt, ChatRole, SUTOptions, TextPrompt
from newhelm.prompt_formatting import format_chat
from newhelm.record_init import record_init
from newhelm.secrets_registry import SECRETS
from newhelm.sut import PromptResponseSUT, SUTCompletion, SUTResponse

from newhelm.sut_registry import SUTS

SECRETS.register(
    "together", "api_key", "See https://api.together.xyz/settings/api-keys"
)

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
}


class TogetherCompletionsRequest(BaseModel):
    # https://docs.together.ai/reference/completions
    model: str
    prompt: str
    max_tokens: int
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    n: Optional[int] = None  # How many completions.


class TogetherCompletionsResponse(BaseModel):
    # https://docs.together.ai/reference/completions
    class Choice(BaseModel):
        text: str

    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    id: str
    choices: List[Choice]
    usage: Usage
    created: int
    model: str
    object: str


class TogetherCompletionsSUT(
    PromptResponseSUT[TogetherCompletionsRequest, TogetherCompletionsResponse]
):
    _URL = "https://api.together.xyz/v1/completions"

    @record_init
    def __init__(self, model):
        self.model = model

    def translate_text_prompt(self, prompt: TextPrompt) -> TogetherCompletionsRequest:
        return self._translate_request(prompt.text, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> TogetherCompletionsRequest:
        return self._translate_request(
            format_chat(prompt, user_text=_USER_ROLE, sut_text=_ASSISTANT_ROLE),
            prompt.options,
        )

    def _translate_request(self, text, options):
        return TogetherCompletionsRequest(
            model=self.model,
            prompt=text,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            repetition_penalty=options.frequency_penalty,
            n=options.num_completions,
        )

    def evaluate(
        self, request: TogetherCompletionsRequest
    ) -> TogetherCompletionsResponse:
        api_key = SECRETS.get_required("together", "api_key")
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = requests.post(self._URL, headers=headers, json=as_json)
        response_status_exception(response)
        return TogetherCompletionsResponse.model_validate(response.json(), strict=True)

    def translate_response(
        self, request: TogetherCompletionsRequest, response: TogetherCompletionsResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.choices:
            assert choice.text is not None
            sut_completions.append(SUTCompletion(text=choice.text))
        return SUTResponse(completions=sut_completions)


class TogetherChatRequest(BaseModel):
    # https://docs.together.ai/reference/chat-completions
    class Message(BaseModel):
        role: str
        content: str

    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    n: Optional[int] = None


class TogetherChatResponse(BaseModel):
    # https://docs.together.ai/reference/chat-completions
    class Choice(BaseModel):
        class Message(BaseModel):
            role: str
            content: str

        message: Message

    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    id: str
    choices: List[Choice]
    usage: Usage
    created: int
    model: str
    object: str


class TogetherChatSUT(PromptResponseSUT[TogetherChatRequest, TogetherChatResponse]):
    _URL = "https://api.together.xyz/v1/chat/completions"

    @record_init
    def __init__(self, model):
        self.model = model

    def translate_text_prompt(self, prompt: TextPrompt) -> TogetherChatRequest:
        return self._translate_request(
            [TogetherChatRequest.Message(content=prompt.text, role=_USER_ROLE)],
            prompt.options,
        )

    def translate_chat_prompt(self, prompt: ChatPrompt) -> TogetherChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(
                TogetherChatRequest.Message(
                    content=message.text, role=_ROLE_MAP[message.role]
                )
            )
        return self._translate_request(messages, prompt.options)

    def _translate_request(
        self, messages: List[TogetherChatRequest.Message], options: SUTOptions
    ):
        return TogetherChatRequest(
            model=self.model,
            messages=messages,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            repetition_penalty=options.frequency_penalty,
            n=options.num_completions,
        )

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        api_key = SECRETS.get_required("together", "api_key")
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = requests.post(self._URL, headers=headers, json=as_json)
        response_status_exception(response)
        return TogetherChatResponse.model_validate(response.json(), strict=True)

    def translate_response(
        self, request: TogetherChatRequest, response: TogetherChatResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.choices:
            text = choice.message.content
            assert text is not None
            sut_completions.append(SUTCompletion(text=text))
        return SUTResponse(completions=sut_completions)


class TogetherInferenceRequest(BaseModel):
    # https://docs.together.ai/reference/inference
    model: str
    # prompt is documented as required, but you can pass messages instead,
    # which is not documented.
    prompt: Optional[str] = None
    messages: Optional[List[TogetherChatRequest.Message]] = None
    max_tokens: int
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    safety_model: Optional[str] = None
    n: Optional[int] = None


class TogetherInferenceResponse(BaseModel):
    class Args(BaseModel):
        model: str
        prompt: Optional[str] = None
        temperature: float
        top_p: float
        top_k: float
        max_tokens: int

    status: str
    prompt: List[str]
    model: str
    # Pydantic uses "model_" as the prefix for its methods, so renaming
    # here to get out of the way.
    owner: str = Field(alias="model_owner")
    tags: Optional[Any] = None
    num_returns: int
    args: Args
    subjobs: List

    class Output(BaseModel):
        class Choice(BaseModel):
            finish_reason: str
            index: Optional[int] = None
            text: str

        choices: List[Choice]
        raw_compute_time: Optional[float] = None
        result_type: str

    output: Output


class TogetherInferenceSUT(
    PromptResponseSUT[TogetherInferenceRequest, TogetherInferenceResponse]
):
    _URL = "https://api.together.xyz/inference"

    @record_init
    def __init__(self, model):
        self.model = model

    def translate_text_prompt(self, prompt: TextPrompt) -> TogetherInferenceRequest:
        return self._translate_request(prompt.text, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> TogetherInferenceRequest:
        return self._translate_request(
            format_chat(prompt, user_text=_USER_ROLE, sut_text=_ASSISTANT_ROLE),
            prompt.options,
        )

    def _translate_request(self, text: str, options: SUTOptions):
        return TogetherInferenceRequest(
            model=self.model,
            prompt=text,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            repetition_penalty=options.frequency_penalty,
            n=options.num_completions,
        )

    def evaluate(self, request: TogetherInferenceRequest) -> TogetherInferenceResponse:
        api_key = SECRETS.get_required("together", "api_key")
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = requests.post(self._URL, headers=headers, json=as_json)
        response_status_exception(response)
        return TogetherInferenceResponse(**response.json())

    def translate_response(
        self, request: TogetherInferenceRequest, response: TogetherInferenceResponse
    ) -> SUTResponse:
        for p in response.prompt:
            print(p)
        sut_completions = []
        for choice in response.output.choices:
            assert choice.text is not None
            sut_completions.append(SUTCompletion(text=choice.text))
        return SUTResponse(completions=sut_completions)


# Language
SUTS.register("llama-2-7b", TogetherCompletionsSUT, "togethercomputer/llama-2-7b")
SUTS.register("llama-2-70b", TogetherCompletionsSUT, "togethercomputer/llama-2-70b")
SUTS.register("falcon-40b", TogetherCompletionsSUT, "togethercomputer/falcon-40b")
SUTS.register("llama-2-13b", TogetherCompletionsSUT, "togethercomputer/llama-2-13b")
SUTS.register("flan-t5-xl", TogetherCompletionsSUT, "google/flan-t5-xl")


# Chat
SUTS.register("llama-2-7b-chat", TogetherChatSUT, "togethercomputer/llama-2-7b-chat")
SUTS.register("llama-2-70b-chat", TogetherChatSUT, "togethercomputer/llama-2-70b-chat")
SUTS.register("zephyr-7b-beta", TogetherChatSUT, "HuggingFaceH4/zephyr-7b-beta")
SUTS.register("vicuna-13b-v1.5", TogetherChatSUT, "lmsys/vicuna-13b-v1.5")
SUTS.register(
    "Mistral-7B-Instruct-v0.2", TogetherChatSUT, "mistralai/Mistral-7B-Instruct-v0.2"
)
SUTS.register("WizardLM-13B-V1.2", TogetherChatSUT, "WizardLM/WizardLM-13B-V1.2")
SUTS.register(
    "oasst-sft-4-pythia-12b-epoch-3.5",
    TogetherChatSUT,
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
)
SUTS.register("dolly-v2-12b", TogetherChatSUT, "databricks/dolly-v2-12b")
