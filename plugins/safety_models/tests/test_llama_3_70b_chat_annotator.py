import json

from modelgauge.suts.together_client import TogetherChatRequest, TogetherChatResponse
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.annotators.llama_3_70b_chat_annotator import (
    _ASSISTANT_ROLE,
    _USER_ROLE,
    Llama3_70bChatAnnotator,
    _flatten_conversation_to_str,
    LLAMA_3_70B_MODEL_PATH,
    _DEFAULT_MAX_CHAT_TOKENS,
)
from modelgauge.safety_model_prompts import (
    LABELER_SYS_PROMPT,
    MLC_TAXONOMY_0_5,
    LABELER_USER_PROMPT,
)
from modelgauge.suts.together_client import TogetherApiKey

from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion


def test_translate_request_text_prompt():
    annotator = Llama3_70bChatAnnotator("_", TogetherApiKey("fake_key"))

    prompt = PromptWithContext(
        prompt=TextPrompt(text="Hello"),
        source_id="some-id",
        context=None,
    )
    completion = SUTCompletion(text="Hi!")
    request = annotator.translate_request(prompt, completion)

    correct_request = TogetherChatRequest(
        model=LLAMA_3_70B_MODEL_PATH,
        messages=[
            TogetherChatRequest.Message(
                role="system",
                content=LABELER_SYS_PROMPT.format(taxonomy=MLC_TAXONOMY_0_5),
            ),
            TogetherChatRequest.Message(
                role=_USER_ROLE,
                content=LABELER_USER_PROMPT.format(
                    messages="User: Hello\n\nAssistant: Hi!",
                    role_to_measure=_ASSISTANT_ROLE,
                ),
            ),
        ],
        max_tokens=_DEFAULT_MAX_CHAT_TOKENS,
        n=1,
    )

    assert request.model_dump_json() == correct_request.model_dump_json()


def test_translate_request_chat_prompt():
    annotator = Llama3_70bChatAnnotator("_", TogetherApiKey("fake_key"))

    prompt = PromptWithContext(
        prompt=ChatPrompt(
            messages=[
                ChatMessage(role=ChatRole.user, text="Hello"),
                ChatMessage(role=ChatRole.sut, text="Hi!"),
            ]
        ),
        source_id="some-id",
        context=None,
    )
    completion = SUTCompletion(text="Hi!")
    request = annotator.translate_request(prompt, completion)

    correct_request = TogetherChatRequest(
        model=LLAMA_3_70B_MODEL_PATH,
        messages=[
            TogetherChatRequest.Message(
                role="system",
                content=LABELER_SYS_PROMPT.format(taxonomy=MLC_TAXONOMY_0_5),
            ),
            TogetherChatRequest.Message(
                role=_USER_ROLE,
                content=LABELER_USER_PROMPT.format(
                    messages="User: Hello\n\nAssistant: Hi!",
                    role_to_measure=_ASSISTANT_ROLE,
                ),
            ),
        ],
        max_tokens=_DEFAULT_MAX_CHAT_TOKENS,
        n=1,
    )

    assert request.model_dump_json() == correct_request.model_dump_json()


def test_translate_response_well_formatted():
    annotator = Llama3_70bChatAnnotator("_", TogetherApiKey("fake_key"))

    response_json_str = json.dumps(
        {"is_safe": True, "safety_categories": [], "reasoning": "nothing to see here"}
    )
    response = _make_together_base_response()
    response.choices = [
        TogetherChatResponse.Choice(
            message=TogetherChatResponse.Choice.Message(
                content=response_json_str, role=_USER_ROLE
            ),
        )
    ]

    result = annotator.translate_response(None, response)

    expected_result = SafetyModelResponse(
        is_safe=True,
        safety_categories=[],
        reasoning="nothing to see here",
        is_valid=True,
        total_tokens_used=response.usage.total_tokens,
    )

    assert result.model_dump_json() == expected_result.model_dump_json()


def test_translate_response_with_error():
    annotator = Llama3_70bChatAnnotator("_", TogetherApiKey("fake_key"))

    bogus_response = "not a json"
    response = _make_together_base_response()
    response.choices = [
        TogetherChatResponse.Choice(
            message=TogetherChatResponse.Choice.Message(
                content=bogus_response, role=_USER_ROLE
            ),
        )
    ]

    result = annotator.translate_response(None, response)

    expected_result = SafetyModelResponse(
        is_safe=False,
        safety_categories=[],
        is_valid=False,
    )

    assert result.model_dump_json() == expected_result.model_dump_json()


def test_flatten_conversation_to_str():
    chat = ChatPrompt(
        messages=[
            ChatMessage(role=ChatRole.user, text="Hello"),
            ChatMessage(role=ChatRole.sut, text="Hi!"),
        ]
    )

    result = _flatten_conversation_to_str(chat)

    assert result == "User: Hello\n\nAssistant: Hi!"


def _make_together_base_response():
    return TogetherChatResponse(
        id="some-id",
        choices=[],
        usage=TogetherChatResponse.Usage(
            prompt_tokens=11,
            completion_tokens=12,
            total_tokens=13,
        ),
        created=99,
        model="some-model",
        object="some-object",
    )
