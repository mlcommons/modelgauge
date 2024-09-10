import os
import pytest
from unittest.mock import Mock, patch

from modelgauge.suts.together_client import TogetherApiKey


def test_can_load():
    try:
        # EnsembleAnnotator is required by the private annotators
        # If we can import it, then the EnsembleAnnotatorSet can be instantiated
        from modelgauge.annotators.ensemble_annotator import EnsembleAnnotator
        from modelgauge.private_ensemble_annotator_set import EnsembleAnnotatorSet

        annotators = EnsembleAnnotatorSet()
        assert annotators.annotators is not None
    except:
        # The EnsembleAnnotator can't be implemented, so the EnsembleAnnotatorSet can't either
        with pytest.raises(NotImplementedError):
            from modelgauge.private_ensemble_annotator_set import EnsembleAnnotatorSet


@patch("modelgauge.private_ensemble_annotator_set.WildguardAnnotator")
def test_annotators(WildguardAnnotator):
    from modelgauge.private_ensemble_annotator_set import (
        EnsembleAnnotatorSet,
        VllmApiKey,
        HuggingFaceKey,
    )

    os.environ["VLLM_ENDPOINT_URL"] = "fake"
    annotators = EnsembleAnnotatorSet()
    annotators.configure()
    assert len(annotators.configuration) == 4  # 3 secrets and one endpoint
    assert annotators.configuration["vllm_endpoint_url"] == "fake"
    assert isinstance(annotators.configuration["together_api_key"], TogetherApiKey)
    assert isinstance(annotators.configuration["huggingface_key"], HuggingFaceKey)
    assert isinstance(annotators.configuration["vllm_api_key"], VllmApiKey)
    assert len(annotators.annotators) == 4
