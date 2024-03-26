from functools import wraps
import inspect
from typing import Dict, Sequence, Type
from newhelm.base_test import BasePromptResponseTest, BaseTest
from newhelm.record_init import add_initialization_record
from newhelm.single_turn_prompt_response import TestItemAnnotations
from newhelm.sut_capabilities import ProducesPerTokenLogProbabilities, SUTCapability


# TODO: Remove defaulting.
def newhelm_test(requires_sut_capabilities: Sequence[Type[SUTCapability]] = []):
    """Decorator providing common behavior and hooks for all NewHELM Tests."""

    def inner(cls):
        assert issubclass(
            cls, BaseTest
        ), "Decorator can only be applied to classes that inherit from BaseTest."
        cls.requires_sut_capabilities = requires_sut_capabilities
        cls.__init__ = _wrap_init(cls.__init__)
        if issubclass(cls, BasePromptResponseTest):
            _override_measure_quality(cls)
        cls._newhelm_test = True
        return cls

    return inner


def _wrap_init(init):
    """Wrap the Test __init__ function to verify it behaves as expected."""

    if hasattr(init, "_newhelm_wrapped"):
        # Already wrapped, no need to do any work.
        return init

    _validate_init_signature(init)

    @wraps(init)
    def wrapped_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        add_initialization_record(self, *args, **kwargs)

    wrapped_init._newhelm_wrapped = True
    return wrapped_init


def _validate_init_signature(init):
    params = list(inspect.signature(init).parameters.values())
    assert params[1].name == "uid", "All Tests must have UID as the first parameter."


def _override_measure_quality(cls: Type[BasePromptResponseTest]) -> None:
    """Wrap the Test measure_quality function to verify it behaves as expected."""

    original = cls.measure_quality

    if hasattr(original, "_newhelm_wrapped"):
        # Already wrapped, no need to do any work.
        return

    @wraps(original)
    def require_logprobs(self, item: TestItemAnnotations) -> Dict[str, float]:
        for interaction in item.interactions:
            for completion in interaction.response.completions:
                assert (
                    completion.top_logprobs is not None
                ), f"{self.__class__.__name__} specifies it requires logprobs, but none were given."
        return original(self, item)

    @wraps(original)
    def remove_logprobs(self, item: TestItemAnnotations) -> Dict[str, float]:
        modified_interactions = []
        for interaction in item.interactions:
            modified_completions = []
            for completion in interaction.response.completions:
                modified_completions.append(
                    completion.model_copy(update={"top_logprobs": None})
                )
            modified_response = interaction.response.model_copy(
                update={"completions": modified_completions}
            )
            modified_interactions.append(
                interaction.model_copy(update={"response": modified_response})
            )
        modified = item.model_copy(update={"interactions": modified_interactions})
        return original(self, modified)

    if ProducesPerTokenLogProbabilities in cls.requires_sut_capabilities:
        cls.measure_quality = require_logprobs  # type: ignore [method-assign]
    else:
        cls.measure_quality = remove_logprobs  # type: ignore [method-assign]

    cls.measure_quality._newhelm_wrapped = True  # type: ignore [attr-defined]
