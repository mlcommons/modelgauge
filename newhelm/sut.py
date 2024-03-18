from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from pydantic import BaseModel
from newhelm.init_hooks import ABCInitHooksMetaclass

from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.record_init import add_initialization_record

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class SUTCompletion(BaseModel):
    """All data about a single completion in the response."""

    text: str


class SUTResponse(BaseModel):
    """The data that came out of the SUT."""

    completions: List[SUTCompletion]


class SUT(ABC, metaclass=ABCInitHooksMetaclass):
    """Base class for all SUTs. There is no guaranteed interface between SUTs, so no methods here."""

    def _before_init(self, *args, **kwargs):
        add_initialization_record(self, *args, **kwargs)


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
