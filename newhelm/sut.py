from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.record_init import InitializationRecord
from newhelm.sut_capabilities import SUTCapability
from newhelm.tracked_object import TrackedObject

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class TokenProbability(BaseModel):
    """Probability assigned to a given token."""

    token: str
    logprob: float


class TopTokens(BaseModel):
    """List of most likely tokens and their probabilities."""

    top_tokens: Sequence[TokenProbability]


class SUTCompletion(BaseModel):
    """All data about a single completion in the response."""

    text: str
    top_logprobs: Optional[Sequence[TopTokens]] = None
    """For each position, list the probabilities for each of the most likely tokens.

    A Test will see this field as None unless it specifies it requires
    ProducesPerTokenLogProbabilities. SUTs that set this value must specify
    they have the ProducesPerTokenLogProbabilities capability. This value
    may be conditional on SUTOptions.top_logprobs being not None.
    """


class SUTResponse(BaseModel):
    """The data that came out of the SUT."""

    completions: List[SUTCompletion]


class SUT(TrackedObject):
    """Base class for all SUTs. There is no guaranteed interface between SUTs, so no methods here."""

    # Set automatically by @newhelm_sut()
    capabilities: Sequence[Type[SUTCapability]]

    def __init__(self, uid: str):
        super().__init__(uid)
        # The initialization record is set automatically by @newhelm_sut()
        self.initialization_record: InitializationRecord


class PromptResponseSUT(SUT, ABC, Generic[RequestType, ResponseType]):
    """The base class for any SUT that is designed for handling a single-turn."""

    @abstractmethod
    def translate_text_prompt(self, prompt: TextPrompt) -> RequestType:
        pass

    @abstractmethod
    def translate_chat_prompt(self, prompt: ChatPrompt) -> RequestType:
        pass

    @abstractmethod
    def evaluate(self, request: RequestType) -> ResponseType:
        pass

    @abstractmethod
    def translate_response(
        self, request: RequestType, response: ResponseType
    ) -> SUTResponse:
        pass
