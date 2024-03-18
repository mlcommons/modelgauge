from unittest.mock import ANY
import pytest
from newhelm.init_hooks import InitHooksMetaclass


class BaseWithHooks(metaclass=InitHooksMetaclass):
    def __init__(self, arg1):
        self.arg1 = arg1

    def _before_init(self, *args, **kwargs):
        pass

    def _after_init(self):
        pass


def test_basic(mocker):
    before_spy = mocker.spy(BaseWithHooks, "_before_init")
    after_spy = mocker.spy(BaseWithHooks, "_after_init")
    x = BaseWithHooks(1234)
    assert x.arg1 == 1234
    before_spy.assert_called_once_with(ANY, 1234)
    after_spy.assert_called_once_with(ANY)


class ChildNoHooks(BaseWithHooks):
    def __init__(self, arg1, arg2):
        super().__init__(arg1)
        self.arg2 = arg2


def test_child_no_hooks(mocker):
    before_spy = mocker.spy(BaseWithHooks, "_before_init")
    after_spy = mocker.spy(BaseWithHooks, "_after_init")
    x = ChildNoHooks(1234, 5)
    assert x.arg1 == 1234
    assert x.arg2 == 5
    before_spy.assert_called_once_with(ANY, 1234, 5)
    after_spy.assert_called_once_with(ANY)


class ChildOverridesHooks(BaseWithHooks):
    def __init__(self, arg2):
        # Note no arg1
        self.arg2 = arg2

    def _before_init(self, *args, **kwargs):
        self.called_before = True

    def _after_init(self):
        self.called_after = True


def test_child_overrides_hooks(mocker):
    base_before_spy = mocker.spy(BaseWithHooks, "_before_init")
    base_after_spy = mocker.spy(BaseWithHooks, "_after_init")
    before_spy = mocker.spy(ChildOverridesHooks, "_before_init")
    after_spy = mocker.spy(ChildOverridesHooks, "_after_init")
    x = ChildOverridesHooks(1234)
    assert x.arg2 == 1234
    assert not hasattr(x, "arg1")
    before_spy.assert_called_once_with(ANY, 1234)
    after_spy.assert_called_once_with(ANY)
    base_before_spy.assert_not_called()
    base_after_spy.assert_not_called()


class ChildCallsSuperHooks(BaseWithHooks):
    def __init__(self, arg1, arg2):
        super().__init__(arg1)
        self.arg2 = arg2

    def _before_init(self, *args, **kwargs):
        super()._before_init(*args, **kwargs)
        self.called_before = True

    def _after_init(self):
        super()._after_init()
        self.called_after = True


def test_child_calls_super_hooks(mocker):
    base_before_spy = mocker.spy(BaseWithHooks, "_before_init")
    base_after_spy = mocker.spy(BaseWithHooks, "_after_init")
    before_spy = mocker.spy(ChildCallsSuperHooks, "_before_init")
    after_spy = mocker.spy(ChildCallsSuperHooks, "_after_init")
    x = ChildCallsSuperHooks(1234, arg2=5)
    assert x.arg1 == 1234
    assert x.arg2 == 5
    assert x.called_before
    assert x.called_after
    before_spy.assert_called_once_with(ANY, 1234, arg2=5)
    after_spy.assert_called_once_with(ANY)
    base_before_spy.assert_called_once_with(ANY, 1234, arg2=5)
    base_after_spy.assert_called_once_with(ANY)


class ChildDoesProcessing(BaseWithHooks):
    def __init__(self, arg1, arg2, arg3):
        super().__init__(arg1)
        self.arg2 = arg2
        self.arg3 = arg3

    def _before_init(self, arg1, arg2, arg3):
        assert arg1 < arg2, "Before failure"

    def _after_init(self):
        assert self.arg2 < self.arg3, "After failure"


def test_child_does_processing_before():
    with pytest.raises(AssertionError) as err_info:
        ChildDoesProcessing(123, 2, 3)
    assert "Before failure" in str(err_info.value)


def test_child_does_processing_after():
    with pytest.raises(AssertionError) as err_info:
        ChildDoesProcessing(1, 222, 3)
    assert "After failure" in str(err_info.value)


def test_child_does_processing_constructs():
    x = ChildDoesProcessing(1, arg2=2, arg3=3)
    assert x.arg1 == 1
    assert x.arg2 == 2
    assert x.arg3 == 3
