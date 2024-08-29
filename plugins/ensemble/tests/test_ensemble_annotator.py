from unittest.mock import ANY, MagicMock

import pytest

from modelgauge.annotator import CompletionAnnotator, PromptWithContext, SUTCompletion
from modelgauge.annotators.ensemble_annotator import EnsembleAnnotator
from modelgauge.prompt import TextPrompt
from modelgauge.safety_model_response import (
    EnsembleSafetyModelResponse,
    SafetyModelResponse,
)


class FakeAnnotator(CompletionAnnotator[SafetyModelResponse]):
    def translate_request(self, prompt: PromptWithContext, completion: SUTCompletion):
        pass

    def annotate(self, annotation_request):
        pass

    def translate_response(self, request, response) -> SafetyModelResponse:
        return SafetyModelResponse(
            is_safe=True,
            safety_categories=[],
        )


def test_empty_ensemble():
    with pytest.raises(
        ValueError, match="EnsembleAnnotator requires at least 2 annotators"
    ):
        empty_ensemble = EnsembleAnnotator("_", annotators=[])


def test_too_few_ensemble():
    with pytest.raises(
        ValueError, match="EnsembleAnnotator requires at least 2 annotators"
    ):
        one_annotator_ensemble = EnsembleAnnotator("_", annotators=[FakeAnnotator("1")])


def test_ensemble_translate_request():
    annotator_1 = FakeAnnotator("1")
    annotator_2 = FakeAnnotator("2")

    annotator_1.translate_request = MagicMock(return_value="annotator 1 request")
    annotator_2.translate_request = MagicMock(return_value="annotator 2 request")

    ensemble = EnsembleAnnotator("_", [annotator_1, annotator_2])

    prompt = PromptWithContext(
        prompt=TextPrompt(text=""),
        source_id="_",
        context=None,
    )
    completion = SUTCompletion(text="")

    requests = ensemble.translate_request(prompt, completion)

    expected_requests = {"1": "annotator 1 request", "2": "annotator 2 request"}

    assert requests == expected_requests


def test_ensemble_annotate():
    annotator_1 = FakeAnnotator("1")
    annotator_2 = FakeAnnotator("2")

    annotator_1.annotate = MagicMock(return_value="annotator 1 annotation")
    annotator_2.annotate = MagicMock(return_value="annotator 2 annotation")

    ensemble = EnsembleAnnotator("_", [annotator_1, annotator_2])

    fake_requests = {
        "1": "request for annotator 1",
        "2": "request for annotator 2",
    }

    annotation = ensemble.annotate(fake_requests)

    expected_annotation = {"1": "annotator 1 annotation", "2": "annotator 2 annotation"}

    assert annotation == expected_annotation


def test_ensemble_join_response_safe():
    annotator_1 = FakeAnnotator("1")
    annotator_2 = FakeAnnotator("2")
    annotator_3 = FakeAnnotator("3")

    ensemble = EnsembleAnnotator("_", [annotator_1, annotator_2, annotator_3])

    fake_responses = {
        "1": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
        "2": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
        "3": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
    }

    annotator_1.translate_response = MagicMock(return_value=fake_responses["1"])
    annotator_2.translate_response = MagicMock(return_value=fake_responses["2"])
    annotator_3.translate_response = MagicMock(return_value=fake_responses["3"])

    fake_requests = {key: "fake request" for key in fake_responses.keys()}

    expected = EnsembleSafetyModelResponse(
        is_safe=True,
        safety_categories=[],
        is_valid=True,
        joined_responses=list(
            fake_responses.values()
        ),  # Not testing for joined response metadata behavior here. Will test in separate test
        is_unanimous=True,
    )

    response = ensemble.translate_response(fake_requests, fake_responses)

    assert response == expected


def test_ensemble_join_response_unsafe():
    annotator_1 = FakeAnnotator("1")
    annotator_2 = FakeAnnotator("2")
    annotator_3 = FakeAnnotator("3")

    ensemble = EnsembleAnnotator("_", [annotator_1, annotator_2, annotator_3])

    fake_responses = {
        "1": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
        "2": SafetyModelResponse(
            is_safe=False, safety_categories=["category1"], is_valid=True
        ),
        "3": SafetyModelResponse(
            is_safe=False, safety_categories=["category2"], is_valid=True
        ),
    }
    annotator_1.translate_response = MagicMock(return_value=fake_responses["1"])
    annotator_2.translate_response = MagicMock(return_value=fake_responses["2"])
    annotator_3.translate_response = MagicMock(return_value=fake_responses["3"])

    fake_requests = {key: "fake request" for key in fake_responses.keys()}

    expected = EnsembleSafetyModelResponse(
        is_safe=False,
        safety_categories=["category1", "category2"],
        is_valid=True,
        joined_responses=list(
            fake_responses.values()
        ),  # Not testing for joined response metadata behavior here. Will test in separate test
        is_unanimous=False,
    )

    response = ensemble.translate_response(fake_requests, fake_responses)

    # Sort lists before comparing
    response.safety_categories.sort()

    assert response == expected


def test_ensemble_join_response_tie():
    annotator_1 = FakeAnnotator("1")
    annotator_2 = FakeAnnotator("2")

    ensemble = EnsembleAnnotator("_", [annotator_1, annotator_2])

    fake_responses = {
        "1": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
        "2": SafetyModelResponse(
            is_safe=False, safety_categories=["category1"], is_valid=True
        ),
    }

    annotator_1.translate_response = MagicMock(return_value=fake_responses["1"])
    annotator_2.translate_response = MagicMock(return_value=fake_responses["2"])

    fake_requests = {key: "fake request" for key in fake_responses.keys()}

    expected = EnsembleSafetyModelResponse(
        is_safe=False,
        safety_categories=["category1"],
        is_valid=True,
        joined_responses=list(
            fake_responses.values()
        ),  # Not testing for joined response metadata behavior here. Will test in separate test
        is_unanimous=False,
    )

    response = ensemble.translate_response(fake_requests, fake_responses)

    # Sort lists before comparing
    response.safety_categories.sort()

    assert response == expected
