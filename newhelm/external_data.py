from abc import ABC, abstractmethod

from newhelm.general import shell


class ExternalData(ABC):
    @abstractmethod
    def download(self, location):
        pass


class WebData(ExternalData):
    def __init__(self, source_url):
        self.source_url = source_url

    def download(self, location):
        shell(["wget", self.source_url, "-O", location])
