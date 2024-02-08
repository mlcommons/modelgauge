from abc import ABC, abstractmethod
import json
import os
from sqlitedict import SqliteDict
from typing import Any, Dict, Generic, Mapping, Optional, get_args

from newhelm.sut import RequestType, ResponseType


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
        self.cached_responses: Optional[
            SqliteDict
        ] = None  # self._load_cached_responses()

    def get_cached_response(self, request: RequestType) -> Optional[ResponseType]:
        # Load cache if hasn't already been loaded (Can't do it in constructor because decoding depends on __orig_class__)
        if not self.cached_responses:
            self.cached_responses = self._load_cached_responses()
        return self.cached_responses.get(request)

    def update_cache(self, request: RequestType, response: ResponseType):
        self.cached_responses[request] = response
        self.cached_responses.commit()
        return

    def close(self):
        if self.cached_responses is not None:
            self.cached_responses.close()

    def _load_cached_responses(self):  # -> Mapping[str, ResponseType]:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        path = os.path.join(self.data_dir, self.fname)
        cache = SqliteDict(
            path,
            encode=self.encoder,
            decode=self.decoder,
            encode_key=self._serialize_request,
            decode_key=self._deserialize_request,
        )
        return cache

    def encoder(self, response: ResponseType) -> str:
        """Encode responses"""
        encoded_response = response.model_dump()
        return json.dumps(encoded_response)

    def decoder(self, encoded_response: str) -> ResponseType:
        """Decode responses"""
        response_type = get_args(self.__orig_class__)[1]
        json_response = json.loads(encoded_response)
        return response_type.model_validate(json_response)

    def _serialize_request(self, request: RequestType) -> str:
        """Encode requests.Use pydantic to serialize request. Returns JSON-encoded string"""
        return request.model_dump_json()

    def _deserialize_request(self, request_key: str) -> RequestType:
        """Decode request keys"""
        request_type = get_args(self.__orig_class__)[0]
        return request_type.model_validate_json(request_key)
