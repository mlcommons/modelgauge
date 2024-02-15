# mypy: ignore-errors
import json
import os
from pydantic import BaseModel
from sqlitedict import SqliteDict  # type: ignore
from typing import Any, Dict, Generic, TypeVar, Optional, get_args

from newhelm.sut import SUT
from newhelm.typed_data import TypedData

RequestType = TypeVar("RequestType", bound=BaseModel)
ResponseType = TypeVar("ResponseType", bound=BaseModel)


class SUTResponseCache(Generic[RequestType, ResponseType]):
    """When caching is enabled, use previously cached SUT responses.

    When used, the local directory structure will look like this:
    data_dir/
        test_1/
            cached_responses/
                sut_1.sqlite
                sut_2.sqlite
        ...
      ...
    """

    def __init__(
        self,
        data_dir: str,
        sut: SUT,
    ):
        self.data_dir = data_dir
        self.fname = f"{sut.__class__.__name__,}.sqlite"
        sut_bases = get_args(sut.__orig_bases__[0])
        self._type_request: RequestType = sut_bases[0]
        self._type_response: ResponseType = sut_bases[1]
        self.cached_responses: SqliteDict = self._load_cached_responses()

    def get_cached_response(self, request: RequestType) -> Optional[ResponseType]:
        encoded_request = self._encode_request(request)
        encoded_response = self.cached_responses.get(encoded_request)
        if encoded_response:
            return encoded_response.to_instance(self._type_response)
        else:
            return None

    def update_cache(self, request: RequestType, response: ResponseType):
        encoded_request = self._encode_request(request)
        encoded_response = self._encode_response(response)
        self.cached_responses[encoded_request] = encoded_response
        self.cached_responses.commit()

    def close(self):
        if self.cached_responses is not None:
            self.cached_responses.close()

    def _load_cached_responses(self) -> SqliteDict:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        path = os.path.join(self.data_dir, self.fname)
        return SqliteDict(path)

    def _encode_response(self, response: ResponseType) -> TypedData:
        return TypedData.from_instance(response)

    def _decode_response(self, encoded_response: TypedData) -> ResponseType:
        return encoded_response.to_instance(self._type_response)

    def _encode_request(self, request: RequestType) -> str:
        return TypedData.from_instance(request).model_dump_json()

    def _decode_request(self, request_key: str) -> RequestType:
        return TypedData.model_validate_json(request_key).to_instance(
            self._type_request
        )
