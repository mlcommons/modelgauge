from pydantic import BaseModel
from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.prompt_formatting import format_chat
from newhelm.sut import SUTCompletion, SUTResponse, PromptResponseSUT
from newhelm.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from newhelm.sut_decorator import newhelm_sut
from newhelm.sut_registry import SUTS
from newhelm.secret_values import SecretDescription, RequiredSecret, InjectSecret

import cohere

class CohereAPIKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="cohere", key="api_key", instructions="You have to get it from cohere.com"
        )

class KurtRequest(BaseModel):
    text: str

class KurtResponse(BaseModel):
    text: str

@newhelm_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class KurtSUT(PromptResponseSUT[KurtRequest, KurtResponse ]):

    def __init__(self, uid: str, api_key: CohereAPIKey):
        super().__init__(uid)
        self.api_key = api_key.value
        self.cohere_client = cohere.Client(self.api_key)

    def translate_text_prompt(self, prompt: TextPrompt) -> KurtRequest:
        return KurtRequest(text=prompt.text)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> KurtRequest:
        return KurtRequest(text=format_chat(prompt.text))
    
    def evaluate(self, request: KurtRequest) -> KurtResponse:
        # Return whether of not the word "what" appears in the response
        response = self.cohere_client.chat(message = request.text)
        return KurtResponse(text=response.text)
    
    def translate_response(self, request: KurtRequest, response: KurtResponse) -> SUTResponse:
        return SUTResponse(completions=[SUTCompletion(text=response.text)])
    
SUTS.register(KurtSUT, "kurt", InjectSecret(CohereAPIKey))

