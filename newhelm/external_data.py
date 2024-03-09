from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import shutil
import urllib.request
import gdown  # type: ignore

from newhelm.data_packing import DataDecompressor, DataUnpacker
from newhelm.general import UrlRetrieveProgressBar


@dataclass(frozen=True, kw_only=True)
class ExternalData(ABC):
    """Base class for defining a source of external data."""

    decompressor: Optional[DataDecompressor] = None
    unpacker: Optional[DataUnpacker] = None

    @abstractmethod
    def download(self, location):
        pass


@dataclass(frozen=True, kw_only=True)
class WebData(ExternalData):
    """External data that can be trivially downloaded using wget."""

    source_url: str

    def download(self, location):
        urllib.request.urlretrieve(
            self.source_url,
            location,
            reporthook=UrlRetrieveProgressBar(self.source_url),
        )


@dataclass(frozen=True, kw_only=True)
class GDriveData(ExternalData):
    """File downloaded using file's name and a google drive folder url."""

    folder_url: str
    filename: str

    def download(self, location):
        # Find file id needed to download the file.
        available_files = gdown.download_folder(
            url=self.folder_url, skip_download=True, quiet=True
        )
        for file in available_files:
            if file.path == self.filename:
                gdown.download(id=file.id, output=location)
                return
        raise RuntimeError(
            f"Cannot find file with name {self.filename} in google drive folder {self.folder_url}"
        )


@dataclass(frozen=True, kw_only=True)
class LocalData(ExternalData):
    """A file that is already on your local machine.

    WARNING: Only use this in cases where your data is not yet
    publicly available, but will be eventually.
    """

    path: str

    def download(self, location):
        shutil.copy(self.path, location)
