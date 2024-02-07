from abc import ABC, abstractmethod
import json
import os
from typing import Generic, Mapping, Optional, get_args

from newhelm.sut import RequestType, ResponseType


class CacheHelper(ABC, Generic[RequestType, ResponseType]):
    @abstractmethod
    def get_cached_response(self, request: RequestType) -> Optional[ResponseType]:
        """Returns the SUT's cached response to a request."""

    @abstractmethod
    def update_cache(self, request: RequestType, response: ResponseType):
        """Cache the SUT's response in-memory."""

    @abstractmethod
    def save_cache(self):
        """Write cache to disk"""


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
        self.fname = f"{sut_name}.jsonl"
        self.cached_responses: Optional[
            Mapping[str, ResponseType]
        ] = {}  # self._load_cached_responses()

    def get_cached_response(self, request: RequestType) -> Optional[ResponseType]:
        # Load cache if hasn't already been loaded (Can't do it in constructor because decoding depends on __orig_class__)
        if not self.cached_responses:
            self.cached_responses = self._load_cached_responses()
        key = self._serialize_request(request)
        if key in self.cached_responses:
            return self.cached_responses[key]
        else:
            return None

    def update_cache(self, request: RequestType, response: ResponseType):
        key = self._serialize_request(request)
        self.cached_responses[key] = response
        return

    def save_cache(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(dir)
        path = os.path.join(self.data_dir, self.fname)
        with open(path, "w") as f:
            f.write(json.dumps(self.cached_responses, default=self.encoder))

    def encoder(self, obj):
        response_type = get_args(self.__orig_class__)[1]
        if isinstance(obj, response_type):
            response = obj.model_dump()
            response["_type"] = response_type.__name__
            return response
        else:
            return json.JSONEncoder(obj)

    def decoder(self, obj):
        response_type = get_args(self.__orig_class__)[1]
        _type = obj.get("_type")
        if _type and _type == response_type.__name__:
            del obj["_type"]  # Delete the `_type` key as it isn't used in the models
            return response_type.model_validate(obj)
        else:
            return obj

    def _load_cached_responses(self) -> Mapping[str, ResponseType]:
        path = os.path.join(self.data_dir, self.fname)
        if os.path.exists(path):
            with open(path, "r") as data_file:
                data = json.load(data_file, object_hook=self.decoder)
            return data
        else:
            return {}

    def _serialize_request(self, request: RequestType) -> str:
        """Use pydantic to serialize request. Returns JSON-encoded string"""
        return request.model_dump_json()
