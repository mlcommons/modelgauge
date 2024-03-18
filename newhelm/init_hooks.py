from functools import wraps


class InitHooksMetaclass(type):
    """Metaclass that will call _before_init and _after_init functions if present."""

    def __new__(cls, *args, **kwargs):
        result = super().__new__(cls, *args, **kwargs)
        result.__init__ = _wrap_init(result.__init__)
        return result


def _wrap_init(init):
    @wraps(init)
    def inner(self, *args, **kwargs):
        try:
            # Keep track of how many `init` calls we've
            # done to ensure before and after only happen once.
            self._init_nesting += 1
        except AttributeError:
            if hasattr(self, "_before_init"):
                self._before_init(*args, **kwargs)
            self._init_nesting = 1
        # Call the underlying __init__ function
        init(self, *args, **kwargs)
        
        self._init_nesting -= 1
        if self._init_nesting == 0:
            if hasattr(self, "_after_init"):
                self._after_init()
            del self._init_nesting

    return inner
