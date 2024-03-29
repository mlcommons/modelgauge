from functools import wraps
import inspect
from typing import Sequence, Type
from newhelm.not_implemented import is_not_implemented
from newhelm.record_init import add_initialization_record
from newhelm.sut import SUT, PromptResponseSUT
from newhelm.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt, SUTCapability


def newhelm_sut(capabilities: Sequence[Type[SUTCapability]]):
    """Decorator providing common behavior and hooks for all NewHELM SUTs."""

    def inner(cls):
        assert issubclass(
            cls, SUT
        ), "Decorator can only be applied to classes that inherit from SUT."
        cls.capabilities = capabilities
        cls.__init__ = _wrap_init(cls.__init__)
        if issubclass(cls, PromptResponseSUT):
            _assert_prompt_types(cls)
        cls._newhelm_sut = True
        return cls

    return inner


def assert_is_sut(obj):
    if not getattr(obj, "_newhelm_sut", False):
        raise AssertionError(
            f"{obj.__class__.__name__} should be decorated with @newhelm_sut."
        )


def _wrap_init(init):
    """Wrap the SUT __init__ function to verify it behaves as expected."""

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
    assert params[1].name == "uid", "All SUTs must have UID as the first parameter."


def _assert_prompt_types(cls: Type[PromptResponseSUT]):
    _assert_prompt_type(cls, AcceptsTextPrompt, cls.translate_text_prompt)
    _assert_prompt_type(cls, AcceptsChatPrompt, cls.translate_chat_prompt)


def _assert_prompt_type(cls, capability, method):
    accepts_text = capability in cls.capabilities
    implements_text = not is_not_implemented(method)
    if accepts_text and not implements_text:
        raise AssertionError(
            f"{cls.__name__} says it {capability.__name__}, but it does not implement {method.__name__}."
        )
    if not accepts_text and implements_text:
        raise AssertionError(
            f"{cls.__name__} implements {method.__name__}, but it does not say it {capability.__name__}."
        )
