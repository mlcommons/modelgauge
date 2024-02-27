from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from pydantic import BaseModel

from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.record_init import record_init
from newhelm.secret_values import SecretValues, SecretsMixin

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class SUTCompletion(BaseModel):
    """All data about a single completion in the response."""

    text: str


class SUTResponse(BaseModel):
    """The data that came out of the SUT."""

    completions: List[SUTCompletion]


class SUT(ABC, SecretsMixin):
    """Base class for all SUTs. There is no guaranteed interface between SUTs, so no methods here."""

    @record_init
    def __init__(self):
        """Ensure all SUTs default to recording their initialization.

        We want to ensure all SUTs record their init to allow us to reconstruct
        their behavior later. If a SUT needs to define its own __init__ that is fine,
        it should just include the decorator.
        """
        pass


class PromptResponseSUT(SUT, ABC, Generic[RequestType, ResponseType]):
    """The base class for any SUT that is designed for handling a single-turn."""

    def load(self, secrets: SecretValues) -> None:
        """Method for performing expensive initialization.

        If your SUT needs secret keys or to perform expensive
        initialization, override this method.
        """
        pass

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
