from torch import tensor
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from typing import List
from unittest.mock import Mock, patch, MagicMock
import pytest
from modelgauge.prompt import SUTOptions, ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTCompletion, SUTResponse, TokenProbability, TopTokens
from modelgauge.suts.huggingface_client import (
    HuggingFaceCompletion,
    HuggingFaceRequest,
    HuggingFaceResponse,
    HuggingFaceSUT,
    HuggingFaceToken,
    WrappedPreTrainedTokenizer,
)

_DEFAULT_REQUEST_ARGS = {
    "temperature": 1.0,
    "top_p": 1,
    "top_k_per_token": 1,
}


def _make_client():
    return HuggingFaceSUT(
        uid="test-sut",
        pretrained_model_name_or_path="some-model",
        token=HuggingFaceToken("some-value"),
    )


class MockTokenizer:
    def __init__(self, output_id_sequences: List[List[int]], model_max_length=512):
        self.model_max_length = model_max_length
        outputs = []
        for ids in output_id_sequences:
            outputs.append(BatchEncoding({"input_ids": ids}))
        self.output_stack = list(reversed(outputs))
        self.text_inputs_receieved = []

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        self.text_inputs_receieved.append(args[0])
        return self.output_stack.pop()


def test_huggingface_translate_text_prompt_request(mocker):
    def mock_create_tokenizer(*args, **kwargs):
        return WrappedPreTrainedTokenizer(MockTokenizer([[1, 2, 3]]))

    mocker.patch(
        "modelgauge.suts.huggingface_client.create_tokenizer", new=mock_create_tokenizer
    )
    client = _make_client()
    prompt = TextPrompt(text="some text prompt")
    request = client.translate_text_prompt(prompt)
    assert request == HuggingFaceRequest(
        input_ids=[1, 2, 3],
        model="some-model",
        num_return_sequences=1,
        max_new_tokens=100,
        **_DEFAULT_REQUEST_ARGS
    )


def test_huggingface_translate_chat_prompt_request(mocker):
    mock_tokenizer = MockTokenizer([[1, 2, 3]])

    def mock_create_tokenizer(*args, **kwargs):
        return WrappedPreTrainedTokenizer(mock_tokenizer)

    mocker.patch(
        "modelgauge.suts.huggingface_client.create_tokenizer", new=mock_create_tokenizer
    )
    client = _make_client()
    prompt = ChatPrompt(
        messages=[
            ChatMessage(text="some-text", role=ChatRole.user),
            ChatMessage(text="more-text", role=ChatRole.sut),
        ]
    )
    request = client.translate_chat_prompt(prompt)
    assert mock_tokenizer.text_inputs_receieved == [format_chat(prompt)]


def test_huggingface_translate_request_reduce_max_tokens(mocker):
    """If total num. tokens (max_tokens + prompt length) exceeds model's max length, the max_new_tokens is adjusted."""

    def mock_create_tokenizer(*args, **kwargs):
        return WrappedPreTrainedTokenizer(
            MockTokenizer([[1, 2, 3, 4, 5]], model_max_length=55)
        )

    mocker.patch(
        "modelgauge.suts.huggingface_client.create_tokenizer", new=mock_create_tokenizer
    )
    client = _make_client()
    prompt = TextPrompt(text="prompt", options=SUTOptions(max_tokens=75))
    request = client.translate_text_prompt(prompt)
    assert request == HuggingFaceRequest(
        input_ids=[1, 2, 3, 4, 5],
        model="some-model",
        num_return_sequences=1,
        max_new_tokens=50,
        **_DEFAULT_REQUEST_ARGS
    )


def test_huggingface_translate_request_prompt_too_large_exception(
    mocker,
):
    """Exception is raised if the num. of prompt tokens exceeds the the model's max length."""

    def mock_create_tokenizer(*args, **kwargs):
        return WrappedPreTrainedTokenizer(
            MockTokenizer([[1, 2, 3]], model_max_length=3)
        )

    mocker.patch(
        "modelgauge.suts.huggingface_client.create_tokenizer", new=mock_create_tokenizer
    )
    client = _make_client()
    prompt = TextPrompt(text="prompt")
    with pytest.raises(AssertionError):
        client.translate_text_prompt(prompt)
