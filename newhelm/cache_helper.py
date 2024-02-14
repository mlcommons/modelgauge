from abc import ABC, abstractmethod
import json
import os
from sqlitedict import SqliteDict
from typing import Any, Dict, Generic, Mapping, Optional, get_args

from newhelm.sut import RequestType, ResponseType
from newhelm.typed_data import TypedData


class CacheHelper(ABC, Generic[RequestType, ResponseType]):
    @abstractmethod
    def get_cached_response(self, request: RequestType) -> Optional[ResponseType]:
        """Returns the SUT's cached response to a request."""

    @abstractmethod
    def update_cache(self, request: RequestType, response: ResponseType):
        """Cache the SUT's response in-memory."""


# TODO: require objects to specify Generic types during instantiation
class SUTResponseCacheHelper(CacheHelper, Generic[RequestType, ResponseType]):
    """When caching is enabled, use previously cached SUT responses.

    When used, the local directory structure will look like this:
    data_dir/
        test_1/
            cached_responses/
                sut_1.jsonl
                sut_2.jsonl
        ...
      ...
    """

    def __init__(
        self,
        data_dir: str,
        sut_name: str,
    ):
        self.data_dir = data_dir
        self.fname = f"{sut_name}.sqlite"
        self.cached_responses: Optional[SqliteDict] = (
            None  # self._load_cached_responses()
        )

    def get_cached_response(self, request: RequestType) -> Optional[ResponseType]:
        # Load cache if hasn't already been loaded (Can't do it in constructor because decoding depends on __orig_class__)
        if not self.cached_responses:
            self.cached_responses = self._load_cached_responses()
        encoded_request = self._encode_request(request)
        encoded_response = self.cached_responses.get(encoded_request)
        if encoded_response:
            response_type = get_args(self.__orig_class__)[1]
            return encoded_response.to_instance(response_type)
        else:
            return None

    def update_cache(self, request: RequestType, response: ResponseType):
        encoded_request = self._encode_request(request)
        encoded_response = self._encode_response(response)
        self.cached_responses[encoded_request] = encoded_response
        self.cached_responses.commit()
        return

    def close(self):
        if self.cached_responses is not None:
            self.cached_responses.close()

    def _load_cached_responses(self):  # -> Mapping[str, ResponseType]:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        path = os.path.join(self.data_dir, self.fname)
        return SqliteDict(path)

    def _encode_response(self, response: ResponseType) -> str:
        """Encode responses"""
        return TypedData.from_instance(response)
        # encoded_response = response.model_dump()
        # return json.dumps(encoded_response)

    def _decode_response(self, encoded_response: str) -> ResponseType:
        """Decode responses"""
        response_type = get_args(self.__orig_class__)[1]
        return encoded_response.to_instance(response_type)
        # json_response = json.loads(encoded_response)
        # return ResponseType.model_validate(json_response)

    def _encode_request(self, request: RequestType) -> str:
        """Encode requests.Use pydantic to serialize request. Returns JSON-encoded string"""
        return TypedData.from_instance(request).model_dump_json()

    def _decode_request(self, request_key: str) -> RequestType:
        """Decode request keys"""
        request_type = get_args(self.__orig_class__)[0]
        return TypedData.model_validate_json(request_key).to_instance(request_type)
