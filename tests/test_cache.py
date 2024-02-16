from pydantic import BaseModel
from typing import List

from newhelm.cache_helper import SUTResponseCache


class SimpleClass(BaseModel):
    value: str


class ParentClass(BaseModel):
    parent_value: str


class ChildClass1(ParentClass):
    child_value: str


class ChildClass2(ParentClass):
    pass


#  round-trip test (where it stores and retrieves a realistic object and makes sure what it gets back has the same content)
def test_simple_request_serialization(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
        simple_request1 = SimpleClass(value="simple request 1")
        assert cache.get_cached_response(simple_request1) == None

        response = SimpleClass(value="simple response")
        cache.update_cache(simple_request1, response)

        simple_request2 = SimpleClass(value="simple request 2")
        assert cache.get_cached_response(simple_request2) == None


def test_simple_round_trip(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
        request = SimpleClass(value="simple request")
        assert cache.get_cached_response(request) == None

        response = SimpleClass(value="simple response")
        cache.update_cache(request, response)
        returned_response = cache.get_cached_response(request)
        assert returned_response == response


def test_polymorphic_request(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
        parent_request = ParentClass(parent_value="parent")
        parent_response = SimpleClass(value="parent response")
        cache.update_cache(parent_request, parent_response)

        child1_request = ChildClass1(parent_value="parent 1", child_value="child 1")
        assert cache.get_cached_response(child1_request) == None
        child1_response = SimpleClass(value="child 1 response")
        cache.update_cache(child1_request, child1_response)

        child2_request = ChildClass2(parent_value="parent")
        assert cache.get_cached_response(child2_request) == None
        child2_response = SimpleClass(value="child 2 response")
        cache.update_cache(child2_request, child2_response)

        assert cache.get_cached_response(parent_request) == parent_response
        assert cache.get_cached_response(child1_request) == child1_response
        assert cache.get_cached_response(child1_request) != child2_response
        assert cache.get_cached_response(child2_request) == child2_response
        assert cache.get_cached_response(child2_request) != parent_response


def test_cache_update(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
        request = SimpleClass(value="val")
        cache.update_cache(request, SimpleClass(value="response 1"))
        new_response = SimpleClass(value="response 2")
        cache.update_cache(request, new_response)
        assert cache.get_cached_response(request) == new_response


def test_polymorphic_response(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
        parent_request = SimpleClass(value="parent request")
        parent_response = ParentClass(parent_value="parent")
        cache.update_cache(parent_request, parent_response)

        child1_request = SimpleClass(value="child 1 request")
        child1_response = ChildClass1(parent_value="parent", child_value="child")
        cache.update_cache(child1_request, child1_response)

        child2_request = SimpleClass(value="child 2 request")
        child2_response = ChildClass2(parent_value="parent")  # Same value as parent
        cache.update_cache(child2_request, child2_response)

        assert cache.get_cached_response(parent_request) == parent_response
        assert cache.get_cached_response(child1_request) == child1_response
        assert cache.get_cached_response(child1_request) != child2_response
        assert cache.get_cached_response(child2_request) == child2_response
        assert cache.get_cached_response(child2_request) != parent_response


# format stability test (where you check in a cache file for a stored object and have a test that reads that and make sure it deserializes properly).
