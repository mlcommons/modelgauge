from typing import List, TypedDict
from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.prompt_formatting import format_chat
from newhelm.sut import SUTCompletion, SUTResponse, PromptResponseSUT
from newhelm.sut_registry import SUTS


class DemoYesNoSUTRequest(TypedDict):
    """The behavior of this sut only depends on the Prompt text."""

    text: str


class DemoYesNoSUTResponse(TypedDict):
    """This SUT is only capable of returning text."""

    text: str


class DemoYesNoSUT(PromptResponseSUT[DemoYesNoSUTRequest, DemoYesNoSUTResponse]):
    """This SUT demonstrates the bare minimum behavior of a SUT: Use the input Prompt to determine the response."""

    def translate_text_prompt(self, prompt: TextPrompt) -> DemoYesNoSUTRequest:
        return {"text": prompt.text}

    def translate_chat_prompt(self, prompt: ChatPrompt) -> DemoYesNoSUTRequest:
        return {"text": format_chat(prompt)}

    def evaluate(self, request: DemoYesNoSUTRequest) -> DemoYesNoSUTResponse:
        # Return Yes if the input is an even number of words
        number_of_words = len(request["text"].split())
        answer = "Yes" if number_of_words % 2 == 0 else "No"
        return {"text": answer}

    def translate_response(
        self, prompt: TextPrompt, response: DemoYesNoSUTResponse
    ) -> SUTResponse:
        return SUTResponse(completions=[SUTCompletion(text=response["text"])])


SUTS.register("demo_yes_no", DemoYesNoSUT)
