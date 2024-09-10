from abc import ABC, abstractmethod


class AnnotatorSet(ABC):
    @property
    def configuration(self):
        raise NotImplementedError

    @property
    def annotators(self):
        raise NotImplementedError

    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
