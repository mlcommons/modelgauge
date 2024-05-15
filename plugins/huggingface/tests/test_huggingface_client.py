from torch import Tensor, tensor, is_tensor, equal
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Union
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
    "model": "some-model",
    "top_k_per_token": 1,
    "generate_args": {
        "temperature": 1.0,
        "num_return_sequences": 1,
        "top_p": 1.0,
    },
}

_INPUT_LIST = [[1, 2, 3, 4, 5]]
# _INPUT_TENSOR = tensor(_INPUT_LIST)
_INPUT_TENSOR = tensor([[1, 2]])

_MASK_LIST = [[1, 1, 1, 0, 0]]
_MASK_TENSOR = tensor(_MASK_LIST)
_STOP_SEQUENCES_IDS = [[4], [5, 6]]

_VOCAB_MAP = {
    "<unk>": 0,
    "hello": 1,
    "world": 2,
    "<stop>": 3,
}


def _make_client():
    return HuggingFaceSUT(
        uid="test-sut",
        pretrained_model_name_or_path="some-model",
        token=HuggingFaceToken("some-value"),
    )


def _make_mocked_client(**kwargs):
    mock_model = MagicMock()
    client = _make_client()
    client.wrapped_tokenizer = WrappedPreTrainedTokenizer(MockTokenizer(**kwargs))
    client.model = mock_model
    return client


class MockTokenizer:
    def __init__(self, model_max_length=512, return_mask=False):
        self.model_max_length = model_max_length
        self.return_mask = return_mask
        self.vocab = _VOCAB_MAP
        self.id_to_token = {id: token for token, id in self.vocab.items()}

    def __call__(self, text: Union[str, List[str]], **kwargs) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]
        token_ids = []
        mask = []
        for sequence in text:
            sequence_ids = [self.vocab.get(token, 0) for token in sequence.split()]
            token_ids.append(sequence_ids)
            mask.append([1] * len(sequence_ids))

        if kwargs.get("return_tensors") == "pt":
            token_ids = tensor(token_ids)
            mask = tensor(mask)
        encoding_data = {"input_ids": token_ids}
        if self.return_mask:
            encoding_data["attention_mask"] = mask

        return BatchEncoding(encoding_data)

    def decode(self, token_ids: Union[int, List[int], "torch.Tensor"]) -> str:
        if is_tensor(token_ids):
            token_ids = token_ids.tolist()
        decoded_tokens = self.convert_ids_to_tokens(token_ids)
        if isinstance(decoded_tokens, list):
            return " ".join(decoded_tokens)
        return decoded_tokens

    def batch_decode(
        self, sequences: Union[List[int], List[List[int]], "torch.Tensor"]
    ) -> List[str]:
        return [self.decode(sequence) for sequence in sequences]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.id_to_token[ids]
        return [self.id_to_token[id] for id in ids]


def test_huggingface_translate_text_prompt_request():
    client = _make_client()
    prompt = TextPrompt(text="some text prompt")
    request = client.translate_text_prompt(prompt)
    assert request == HuggingFaceRequest(
        prompt="some text prompt",
        max_new_tokens=100,
        stop_sequences=[],
        **_DEFAULT_REQUEST_ARGS
    )


def test_huggingface_translate_chat_prompt_request():
    client = _make_client()
    prompt = ChatPrompt(
        messages=[
            ChatMessage(text="some-text", role=ChatRole.user),
            ChatMessage(text="more-text", role=ChatRole.sut),
        ]
    )
    request = client.translate_chat_prompt(prompt)
    assert request.prompt == format_chat(prompt)


def test_huggingface_translate_request_non_default_options():
    client = _make_client()
    options = SUTOptions(
        temperature=0.5,
        num_completions=2,
        top_p=0.4,
        max_tokens=15,
        top_k_per_token=3,
        stop_sequences=["stop"],
    )
    request = client._translate_request("some text prompt", options)
    assert request == HuggingFaceRequest(
        prompt="some text prompt",
        model="some-model",
        max_new_tokens=15,
        top_k_per_token=3,
        stop_sequences=["stop"],
        generate_args={"temperature": 0.5, "num_return_sequences": 2, "top_p": 0.4},
    )
    # Test that temperature of 0 is adjusted.
    request = client._translate_request("some text prompt", SUTOptions(temperature=0.0))
    assert request.generate_args.temperature == 1e-7


def test_huggingface_request_serialization_cacheable(tmpdir):
    request = HuggingFaceRequest(
        prompt="prompt", max_new_tokens=100, stop_sequences=[], **_DEFAULT_REQUEST_ARGS
    )
    with SqlDictCache(tmpdir, "sut_name") as cache:
        cache._hash_request(request)


def test_huggingface_generate_args():
    """Tests that the expected arguments are passed to the .generate() call in the evaluate method."""
    client = _make_mocked_client()
    request = HuggingFaceRequest(
        prompt="hello world",
        max_new_tokens=100,
        stop_sequences=[],
        **_DEFAULT_REQUEST_ARGS
    )
    client.evaluate(request)

    _, generate_call_kwargs = client.model.generate.call_args
    input_ids = generate_call_kwargs.pop("input_ids")
    assert is_tensor(input_ids)
    assert equal(input_ids, _INPUT_TENSOR)
    assert generate_call_kwargs == {
        **_DEFAULT_REQUEST_ARGS["generate_args"],
        "max_new_tokens": 100,
        "do_sample": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }


def test_huggingface_generate_args_mask():
    """Tests that the expected arguments are passed to the .generate() call in the evaluate method."""
    client = _make_mocked_client(return_mask=True)
    request = HuggingFaceRequest(
        prompt="hello world",
        max_new_tokens=100,
        stop_sequences=[],
        **_DEFAULT_REQUEST_ARGS
    )
    client.evaluate(request)

    _, generate_call_kwargs = client.model.generate.call_args
    mask = generate_call_kwargs.get("attention_mask")
    assert is_tensor(mask)
    assert equal(mask, tensor([[1, 1]]))


def test_huggingface_generate_stopping_criteria_arg():
    client = _make_mocked_client()
    request = HuggingFaceRequest(
        prompt="hello world",
        max_new_tokens=100,
        stop_sequences=["<stop>", "hello <eos>"],
        **_DEFAULT_REQUEST_ARGS
    )
    client.evaluate(request)

    _, generate_call_kwargs = client.model.generate.call_args
    stopping_criteria = generate_call_kwargs["stopping_criteria"]
    assert isinstance(stopping_criteria, StoppingCriteriaList)
    assert [token.stop_sequence for token in stopping_criteria] == [
        [_VOCAB_MAP["<stop>"]],
        [_VOCAB_MAP["hello"], _VOCAB_MAP["<unk>"]],
    ]


def test_huggingface_generate_args_with_reduced_max_tokens():
    """If total num. tokens (max_tokens + prompt length) exceeds model's max length, the max_new_tokens is adjusted."""
    client = _make_mocked_client(model_max_length=53)

    request = HuggingFaceRequest(
        prompt="some text prompt",
        max_new_tokens=75,
        stop_sequences=[],
        **_DEFAULT_REQUEST_ARGS
    )
    client.evaluate(request)
    _, generate_call_kwargs = client.model.generate.call_args
    assert generate_call_kwargs["max_new_tokens"] == 50


def test_huggingface_evaluate_prompt_too_large_exception():
    """Exception is raised if the num. of prompt tokens exceeds the the model's max length."""
    client = _make_mocked_client(model_max_length=3)

    request = HuggingFaceRequest(
        prompt="some text prompt",
        max_new_tokens=75,
        stop_sequences=[],
        **_DEFAULT_REQUEST_ARGS
    )
    with pytest.raises(
        AssertionError, match="Prompt has 3 tokens, which is >= max length 3"
    ):
        client.evaluate(request)
