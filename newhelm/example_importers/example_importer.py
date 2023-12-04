from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData

from newhelm.instance import Instance


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    description: str


# TODO Figure out if we want to split the first two methods from the third, so the same "get_examples"
# logic can be used with multiple datasets. In the current form, all those datasets would share the
# same ABC which inherits from ExampleImporter and defines get_examples.
class ExampleImporter(ABC):
    @classmethod
    @abstractmethod
    def get_dependencies(cls) -> Mapping[str, ExternalData]:
        """Return a mapping of external dependency name to how it can be found downloaded."""
        pass

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> DatasetMetadata:
        """Return a description of the dataset."""
        pass

    @abstractmethod
    def get_examples(
        self, dependency_helper: DependencyHelper, subsets: List[str] = []
    ) -> List[Instance]:
        pass
