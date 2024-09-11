from huggingface_hub import (
    ChatCompletionOutput,
    InferenceClient,
    InferenceEndpointStatus,
    get_inference_endpoint,
)
from huggingface_hub.utils import HfHubHTTPError
from pydantic import BaseModel
from typing import List, Optional

from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


class ChatMessage(BaseModel):
    content: str
    role: str


class HuggingFaceInferenceChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class HuggingFaceInferenceToken(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="hugging_face",
            key="token",
            instructions="You can create tokens at https://huggingface.co/settings/tokens.",
        )


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class HuggingFaceInferenceSUT(
    PromptResponseSUT[HuggingFaceInferenceChatRequest, ChatCompletionOutput]
):
    """A Hugging Face SUT that is hosted on a dedicated inference endpoint."""

    def __init__(
        self, uid: str, inference_endpoint: str, token: HuggingFaceInferenceToken
    ):
        super().__init__(uid)
        self.token = token
        self.endpoint = get_inference_endpoint(inference_endpoint, token=token.value)
        self.client = None

    def _create_client(self):
        timeout = 60 * 6
        if self.endpoint.status in [
            InferenceEndpointStatus.PENDING,
            InferenceEndpointStatus.INITIALIZING,
            InferenceEndpointStatus.UPDATING,
        ]:
            print(
                f"Endpoint starting. Status: {self.endpoint.status}. Waiting up to {timeout}s to start."
            )
            self.endpoint.wait(timeout)
        elif self.endpoint.status == InferenceEndpointStatus.SCALED_TO_ZERO:
            print("Endpoint scaled to zero... requesting to resume.")
            try:
                self.endpoint.resume(running_ok=True)
            except HfHubHTTPError:
                raise ConnectionError(
                    "Failed to resume endpoint. Please resume manually."
                )
            print(f"Requested resume. Waiting up to {timeout}s to start.")
            self.endpoint.wait(timeout)
        elif self.endpoint.status != InferenceEndpointStatus.RUNNING:
            raise ConnectionError(
                "Endpoint is not running: Please contact admin to ensure endpoint is starting or running"
            )

        self.client = InferenceClient(
            base_url=self.endpoint.url, token=self.token.value
        )

    def translate_text_prompt(
        self, prompt: TextPrompt
    ) -> HuggingFaceInferenceChatRequest:
        return HuggingFaceInferenceChatRequest(
            messages=[ChatMessage(role="user", content=prompt.text)],
            **prompt.options.model_dump(),
        )

    def evaluate(
        self, request: HuggingFaceInferenceChatRequest
    ) -> ChatCompletionOutput:
        if self.client is None:
            self._create_client()

        request_dict = request.model_dump(exclude_none=True)
        return self.client.chat_completion(**request_dict)

    def translate_response(
        self, request: HuggingFaceInferenceChatRequest, response: ChatCompletionOutput
    ) -> SUTResponse:
        completions = []
        for choice in response.choices:
            text = choice.message.content
            assert text is not None
            completions.append(SUTCompletion(text=text))
        return SUTResponse(completions=completions)


SUTS.register(
    HuggingFaceInferenceSUT,
    "gemma-9b-it-hf",
    "gemma-2-9b-it-qfa",
    InjectSecret(HuggingFaceInferenceToken),
)
