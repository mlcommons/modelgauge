from typing import Any, List, Optional
from pydantic import BaseModel
import requests
from together.utils import response_status_exception  # type: ignore
from newhelm.placeholders import Prompt
from newhelm.secrets_registry import SECRETS
from newhelm.sut import PromptResponseSUT, SUTCompletion, SUTResponse

from newhelm.sut_registry import SUTS

SECRETS.register(
    "together", "api_key", "See https://api.together.xyz/settings/api-keys"
)


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

    def __init__(self, model):
        self.model = model

    def translate_request(self, prompt: Prompt) -> TogetherCompletionsRequest:
        return TogetherCompletionsRequest(
            model=self.model,
            prompt=prompt.text,
            max_tokens=prompt.options.max_tokens,
            stop=prompt.options.stop_sequences,
            temperature=prompt.options.temperature,
            top_p=prompt.options.top_p,
            top_k=prompt.options.top_k_per_token,
            repetition_penalty=prompt.options.frequency_penalty,
            n=prompt.options.num_completions,
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
        return TogetherCompletionsResponse(**response.json())

    def translate_response(
        self, prompt: Prompt, response: TogetherCompletionsResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.choices:
            assert choice.text is not None
            sut_completions.append(SUTCompletion(choice.text))
        return SUTResponse(sut_completions)


class TogetherChatRequest(BaseModel):
    # https://docs.together.ai/reference/chat-completions
    class Message(BaseModel):
        role: str = "user"
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

    def __init__(self, model):
        self.model = model

    def translate_request(self, prompt: Prompt) -> TogetherChatRequest:
        return TogetherChatRequest(
            model=self.model,
            messages=[TogetherChatRequest.Message(content=prompt.text)],
            max_tokens=prompt.options.max_tokens,
            stop=prompt.options.stop_sequences,
            temperature=prompt.options.temperature,
            top_p=prompt.options.top_p,
            top_k=prompt.options.top_k_per_token,
            repetition_penalty=prompt.options.frequency_penalty,
            n=prompt.options.num_completions,
        )

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        api_key = SECRETS.get_required("together", "api_key")
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = requests.post(self._URL, headers=headers, json=as_json)
        response_status_exception(response)
        return TogetherChatResponse(**response.json())

    def translate_response(
        self, prompt: Prompt, response: TogetherChatResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.choices:
            text = choice.message.content
            assert text is not None
            sut_completions.append(SUTCompletion(text))
        return SUTResponse(sut_completions)


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
    model_owner: str
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
    # This is needed because model_owner makes Pydantic raise a warning that
    # we are using something in the model_ namespace.
    model_config = {"protected_namespaces": ()}


class TogetherInferenceSUT(
    PromptResponseSUT[TogetherInferenceRequest, TogetherInferenceResponse]
):
    _URL = "https://api.together.xyz/inference"

    def __init__(self, model):
        self.model = model

    def translate_request(self, prompt: Prompt) -> TogetherInferenceRequest:
        return TogetherInferenceRequest(
            model=self.model,
            prompt=prompt.text,
            # TODO: Add messages here once issue #56 is resolved.
            max_tokens=prompt.options.max_tokens,
            stop=prompt.options.stop_sequences,
            temperature=prompt.options.temperature,
            top_p=prompt.options.top_p,
            top_k=prompt.options.top_k_per_token,
            repetition_penalty=prompt.options.frequency_penalty,
            n=prompt.options.num_completions,
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
        self, prompt: Prompt, response: TogetherInferenceResponse
    ) -> SUTResponse:
        for p in response.prompt:
            print(p)
        sut_completions = []
        for choice in response.output.choices:
            assert choice.text is not None
            sut_completions.append(SUTCompletion(choice.text))
        return SUTResponse(sut_completions)


SUTS.register("llama-2-7b", TogetherCompletionsSUT, "togethercomputer/llama-2-7b")
SUTS.register("llama-2-70b", TogetherCompletionsSUT, "togethercomputer/llama-2-70b")
SUTS.register("llama-2-7b-chat", TogetherChatSUT, "togethercomputer/llama-2-7b-chat")
SUTS.register("llama-2-70b-chat", TogetherChatSUT, "togethercomputer/llama-2-70b-chat")
