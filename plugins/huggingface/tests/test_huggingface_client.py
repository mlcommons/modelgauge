from torch import tensor, is_tensor, equal
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from typing import List
from unittest.mock import Mock, patch, MagicMock
import pytest
from modelgauge.caching import SqlDictCache
from modelgauge.prompt import SUTOptions, ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTCompletion, SUTResponse, TokenProbability, TopTokens
from modelgauge.suts.huggingface_client import (
    HuggingFaceCompletion,
    HuggingFaceRequest,
    HuggingFaceResponse,
    HuggingFaceSUT,
    HuggingFaceToken,
    StopAtSpecificTokenCriteria,
    WrappedPreTrainedTokenizer,
)

_DEFAULT_REQUEST_ARGS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k_per_token": 1,
    "num_return_sequences": 1,
    "model": "some-model",
}

_INPUT_LIST = [[1, 2, 3, 4, 5]]
_INPUT_TENSOR = tensor(_INPUT_LIST)
_MASK_LIST = [[1, 1, 1, 0, 0]]
_MASK_TENSOR = tensor(_MASK_LIST)
_STOP_SEQUENCES_IDS = [[4], [5, 6]]


def _make_client():
    return HuggingFaceSUT(
        uid="test-sut",
        pretrained_model_name_or_path="some-model",
        token=HuggingFaceToken("some-value"),
    )


class MockTokenizer:
    def __init__(self, output_id_sequences, model_max_length=512):
        self.model_max_length = model_max_length
        outputs = []
        for ids in output_id_sequences:
            outputs.append(BatchEncoding({"input_ids": ids}))
        self.output_stack = list(reversed(outputs))
        self.text_inputs_receieved = []

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        self.text_inputs_receieved.append(args[0])
        return self.output_stack.pop()


def test_huggingface_request_serialization():
    request = HuggingFaceRequest(
        input_ids=_INPUT_TENSOR, max_new_tokens=100, **_DEFAULT_REQUEST_ARGS
    )
    assert request.model_dump() == {
        "input_ids": _INPUT_LIST,
        "attention_mask": None,
        "max_new_tokens": 100,
        "stop_sequence_ids": None,
        **_DEFAULT_REQUEST_ARGS,
    }


def test_huggingface_request_optional_args_serialization():
    request = HuggingFaceRequest(
        input_ids=_INPUT_TENSOR,
        attention_mask=_MASK_TENSOR,
        max_new_tokens=100,
        stop_sequence_ids=_STOP_SEQUENCES_IDS,
        **_DEFAULT_REQUEST_ARGS
    )
    assert request.model_dump()["input_ids"] == _INPUT_LIST
    assert request.model_dump()["attention_mask"] == _MASK_LIST
    assert request.model_dump()["stop_sequence_ids"] == _STOP_SEQUENCES_IDS


@pytest.mark.parametrize("mask", [None, _MASK_TENSOR])
def test_huggingface_request_tensor_serialization_generation_context(mask):
    """Test that model_dump serializes tensors as tensors with generation context."""
    request = HuggingFaceRequest(
        input_ids=_INPUT_TENSOR,
        attention_mask=mask,
        max_new_tokens=100,
        **_DEFAULT_REQUEST_ARGS
    )
    serialized_request = request.model_dump(context={"keep_tensors": True})
    assert is_tensor(serialized_request["input_ids"])
    assert equal(serialized_request["input_ids"], _INPUT_TENSOR)
    if mask is None:
        assert serialized_request["attention_mask"] == None
    else:
        assert is_tensor(serialized_request["attention_mask"])
        assert equal(serialized_request["attention_mask"], mask)


def test_huggingface_request_serialization_cacheable(tmpdir):
    """Test that model_dump_json() does not raise an exception due to tensor serialization errors, for example"""
    request = HuggingFaceRequest(
        input_ids=_INPUT_TENSOR, max_new_tokens=100, **_DEFAULT_REQUEST_ARGS
    )
    with SqlDictCache(tmpdir, "sut_name") as cache:
        cache._hash_request(request)


def test_huggingface_translate_text_prompt_request(mocker):
    def mock_create_tokenizer(*args, **kwargs):
        return WrappedPreTrainedTokenizer(MockTokenizer([_INPUT_TENSOR]))

    mocker.patch(
        "modelgauge.suts.huggingface_client.create_tokenizer", new=mock_create_tokenizer
    )
    client = _make_client()
    prompt = TextPrompt(text="some text prompt")
    request = client.translate_text_prompt(prompt)
    assert request == HuggingFaceRequest(
        input_ids=_INPUT_TENSOR, max_new_tokens=100, **_DEFAULT_REQUEST_ARGS
    )


def test_huggingface_translate_chat_prompt_request(mocker):
    mock_tokenizer = MockTokenizer([_INPUT_TENSOR])

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
            MockTokenizer([_INPUT_TENSOR], model_max_length=55)
        )

    mocker.patch(
        "modelgauge.suts.huggingface_client.create_tokenizer", new=mock_create_tokenizer
    )
    client = _make_client()
    prompt = TextPrompt(text="prompt", options=SUTOptions(max_tokens=75))
    request = client.translate_text_prompt(prompt)
    assert request == HuggingFaceRequest(
        input_ids=_INPUT_TENSOR, max_new_tokens=50, **_DEFAULT_REQUEST_ARGS
    )


def test_huggingface_translate_request_prompt_too_large_exception(
    mocker,
):
    """Exception is raised if the num. of prompt tokens exceeds the the model's max length."""

    def mock_create_tokenizer(*args, **kwargs):
        return WrappedPreTrainedTokenizer(
            MockTokenizer(_INPUT_TENSOR, model_max_length=5)
        )

    mocker.patch(
        "modelgauge.suts.huggingface_client.create_tokenizer", new=mock_create_tokenizer
    )
    client = _make_client()
    prompt = TextPrompt(text="prompt")
    with pytest.raises(AssertionError):
        client.translate_text_prompt(prompt)


def test_huggingface_translate_request_stop_sequences(mocker):
    def mock_create_tokenizer(*args, **kwargs):
        return WrappedPreTrainedTokenizer(
            MockTokenizer([_INPUT_TENSOR, _STOP_SEQUENCES_IDS])
        )

    mocker.patch(
        "modelgauge.suts.huggingface_client.create_tokenizer", new=mock_create_tokenizer
    )
    client = _make_client()
    prompt = TextPrompt(
        text="prompt", options=SUTOptions(stop_sequences=["stop", "<eos>"])
    )
    request = client.translate_text_prompt(prompt)
    assert request == HuggingFaceRequest(
        input_ids=_INPUT_TENSOR,
        stop_sequence_ids=_STOP_SEQUENCES_IDS,
        max_new_tokens=100,
        **_DEFAULT_REQUEST_ARGS
    )


@pytest.mark.parametrize("mask", [None, _MASK_TENSOR])
def test_huggingface_generate_args(mask):
    """Tests that the expected arguments are passed to the .generate() call in the evaluate method."""
    client = _make_client()
    client._load_model = MagicMock()
    client._load_tokenizer = MagicMock()
    mask_arg = {}
    if mask is not None:
        mask_arg["attention_mask"] = mask
    request = HuggingFaceRequest(
        input_ids=_INPUT_TENSOR, max_new_tokens=100, **mask_arg, **_DEFAULT_REQUEST_ARGS
    )
    # Patching model attribute of the HuggingFaceSUT instance
    with patch.object(client, "model", autospec=True) as mock_model:
        client.evaluate(request)
        generate_call_args, generate_call_kwargs = mock_model.generate.call_args
        assert generate_call_kwargs == {
            "input_ids": _INPUT_TENSOR,
            "temperature": 1.0,
            "num_return_sequences": 1,
            "max_new_tokens": 100,
            "top_p": 1.0,
            "do_sample": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            **mask_arg,
        }
        assert is_tensor(generate_call_kwargs["input_ids"])
        if mask is not None:
            assert is_tensor(generate_call_kwargs["attention_mask"])


def test_huggingface_generate_stopping_criteria():
    client = _make_client()
    client._load_model = MagicMock()
    client._load_tokenizer = MagicMock()
    request = HuggingFaceRequest(
        input_ids=_INPUT_TENSOR,
        stop_sequence_ids=_STOP_SEQUENCES_IDS,
        max_new_tokens=100,
        **_DEFAULT_REQUEST_ARGS
    )
    # Patching model attribute of the HuggingFaceSUT instance
    with patch.object(client, "model", autospec=True) as mock_model:
        client.evaluate(request)
        generate_call_args, generate_call_kwargs = mock_model.generate.call_args

        assert isinstance(
            generate_call_kwargs["stopping_criteria"], StoppingCriteriaList
        )
        assert len(generate_call_kwargs["stopping_criteria"]) == len(
            _STOP_SEQUENCES_IDS
        )
        for i, token in enumerate(generate_call_kwargs["stopping_criteria"]):
            assert token.stop_sequence == _STOP_SEQUENCES_IDS[i]
