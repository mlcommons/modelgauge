from typing import Dict, List, Mapping

from pydantic import BaseModel
from newhelm.annotation import Annotation
from typing import Dict, List, Mapping
from newhelm.annotation import Annotation

from newhelm.benchmark import Score
from newhelm.placeholders import Result
from newhelm.record_init import InitializationRecord
from newhelm.single_turn_prompt_response import (
    PromptInteraction,
    TestItem,
)


class TestItemRecord(BaseModel):
    # TODO: This duplicates the list of prompts across test_item and interactions.
    # Maybe just copy the TestItem context.
    test_item: TestItem
    interactions: List[PromptInteraction]
    annotations: Dict[str, Annotation]
    measurements: Dict[str, float]

    __test__ = False


class TestRecord(BaseModel):
    """This is a rough sketch of the kind of data we'd want every Test to record."""

    test_name: str
    test_initialization: InitializationRecord
    dependency_versions: Mapping[str, str]
    sut_name: str
    sut_initialization: InitializationRecord
    # TODO We should either reintroduce "Turns" here, or expect
    # there to b different schemas for different TestImplementationClasses.
    test_item_records: List[TestItemRecord]
    results: List[Result]

    __test__ = False


class BenchmarkRecord(BaseModel):
    """This is a rough sketch of the kind of data we'd want every Benchmark to record."""

    benchmark_name: str
    sut_name: str
    test_records: List[TestRecord]
    score: Score
