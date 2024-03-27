from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

from newhelm.not_implemented import not_implemented
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
    # This is only accessible via SUT capabilities
    top_logprobs: Optional[Sequence[TopTokens]] = None


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

    @not_implemented
    def translate_text_prompt(self, prompt: TextPrompt) -> RequestType:
        raise NotImplementedError(
            f"SUT {self.__class__.__name__} does not implement translate_text_prompt."
        )

    @not_implemented
    def translate_chat_prompt(self, prompt: ChatPrompt) -> RequestType:
        raise NotImplementedError(
            f"SUT {self.__class__.__name__} does not implement translate_chat_prompt."
        )

    @abstractmethod
    def evaluate(self, request: RequestType) -> ResponseType:
        pass

    @abstractmethod
    def translate_response(
        self, request: RequestType, response: ResponseType
    ) -> SUTResponse:
        pass
