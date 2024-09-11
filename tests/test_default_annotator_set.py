import pytest
from unittest.mock import MagicMock, patch
from modelgauge.default_annotator_set import DefaultAnnotatorSet


@patch("modelgauge.default_annotator_set.load_secrets_from_config")
def test_constructor(load_secrets_from_config):
    annotators = DefaultAnnotatorSet()
    annotators.configure()
    assert len(annotators.annotators) == 1
    assert "llama_guard_2" in annotators.annotators


@patch("modelgauge.default_annotator_set.load_secrets_from_config")
def test_evaluate(load_secrets_from_config):
    annotators = DefaultAnnotatorSet()
    annotators.configure()
    item = MagicMock()
    assert type(annotators.evaluate(item).get("is_safe", None)) == float
