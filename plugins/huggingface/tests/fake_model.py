from unittest.mock import MagicMock
from typing import List, Union

import torch
from transformers.tokenization_utils_base import BatchEncoding

from modelgauge.suts.huggingface_client import (
    HuggingFaceSUT,
    HuggingFaceToken,
    WrappedPreTrainedTokenizer,
)


def make_client():
    return HuggingFaceSUT(
        uid="test-sut",
        pretrained_model_name_or_path="some-model",
        token=HuggingFaceToken("some-value"),
    )


def make_mocked_client(vocab_map, **t_kwargs):
    mock_model = MagicMock()
    client = make_client()
    client.wrapped_tokenizer = WrappedPreTrainedTokenizer(
        MockTokenizer(vocab_map, **t_kwargs)
    )
    client.model = mock_model
    return client


class MockTokenizer:
    def __init__(self, vocab_map, model_max_length=512, return_mask=False):
        self.model_max_length = model_max_length
        self.returns_mask = return_mask
        self.vocab = vocab_map
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
            token_ids = torch.tensor(token_ids)
            mask = torch.tensor(mask)
        encoding_data = {"input_ids": token_ids}
        if self.returns_mask:
            encoding_data["attention_mask"] = mask

        return BatchEncoding(encoding_data)

    def decode(self, token_ids: Union[int, List[int], torch.Tensor]) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        decoded_tokens = self.convert_ids_to_tokens(token_ids)
        if isinstance(decoded_tokens, list):
            return " ".join(decoded_tokens)
        return decoded_tokens

    def batch_decode(
        self, sequences: Union[List[int], List[List[int]], torch.Tensor]
    ) -> List[str]:
        return [self.decode(sequence) for sequence in sequences]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.id_to_token[ids]
        return [self.id_to_token[id] for id in ids]
