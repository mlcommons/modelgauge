from abc import ABC, abstractmethod
import hashlib
import os
from pydantic import BaseModel
from sqlitedict import SqliteDict  # type: ignore

from newhelm.typed_data import TypedData
from newhelm.general import normalize_filename


class BaseCache(ABC):
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, *exc_info):
        pass

    @abstractmethod
    def get_or_call(self, request, callable):
        pass

    @abstractmethod
    def get_cached_response(self, request):
        pass

    @abstractmethod
    def update_cache(self, request, response):
        pass


class CacheEntry(BaseModel):
    """Wrapper around the data we write to the cache."""

    payload: TypedData


class SqlDictCache(BaseCache):
    """Cache the response from a method using the request as the key.

    Will create a `file_identifier`.sqlite file in `data_dir` to persist
    the cache.
    """

    def __init__(self, data_dir, file_identifier):
        self.data_dir = data_dir
        self.fname = normalize_filename(f"{file_identifier}_cache.sqlite")
        self.cached_responses = self._load_cached_responses()

    def __enter__(self):
        self.cached_responses.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.cached_responses.close()

    def get_or_call(self, request, callable):
        """Return the cached value, otherwise cache calling `callable`"""
        response = self.get_cached_response(request)
        if response is not None:
            return response
        response = callable(request)
        self.update_cache(request, response)
        return response

    def get_cached_response(self, request):
        if not self._can_encode(request):
            return None
        encoded_request = self._hash_request(request)
        encoded_response = self.cached_responses.get(encoded_request)
        if encoded_response:
            return self._decode_response(encoded_response)
        else:
            return None

    def update_cache(self, request, response):
        if not self._can_encode(request) or not self._can_encode(response):
            return
        encoded_request = self._hash_request(request)
        encoded_response = self._encode_response(response)
        self.cached_responses[encoded_request] = encoded_response
        self.cached_responses.commit()

    def _load_cached_responses(self):
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, self.fname)
        return SqliteDict(path)

    def _can_encode(self, obj) -> bool:
        # Encoding currently requires Pydanic objects.
        return isinstance(obj, BaseModel)

    def _encode_response(self, response) -> CacheEntry:
        return CacheEntry(payload=TypedData.from_instance(response))

    def _decode_response(self, encoded_response: CacheEntry):
        return encoded_response.payload.to_instance()

    def _hash_request(self, request) -> str:
        return hashlib.sha256(
            TypedData.from_instance(request).model_dump_json().encode()
        ).hexdigest()


class NoCache(BaseCache):
    """Implements the caching interface, but never actually caches."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def get_or_call(self, request, callable):
        return callable(request)

    def get_cached_response(self, request):
        return None

    def update_cache(self, request, response):
        pass
