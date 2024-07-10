from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Mapping

from modelgauge.annotation import Annotation
from modelgauge.annotator import AnnotationType
from modelgauge.base_test import BaseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData
from modelgauge.interaction_annotation import AnnotationTestItem, MeasuredAnnotationItem
from modelgauge.typed_data import Typeable


class InteractionAnnotationTest(BaseTest, ABC, Generic[AnnotationType]):
    """Base class for tests that evaluate CompletionAnnotators."""

    @abstractmethod
    def get_dependencies(self) -> Mapping[str, ExternalData]:
        """Return a mapping of external dependency name to how it can be found downloaded."""
        pass

    @abstractmethod
    def make_test_items(
        self, dependency_helper: DependencyHelper
    ) -> List[AnnotationTestItem]:
        """Generate all data that will go to the annotator."""
        pass

    @abstractmethod
    def measure_quality(
        self, item: AnnotationTestItem, annotation: AnnotationType
    ) -> Dict[str, float]:
        """Measure how well the annotator did on this test item against the gold labels."""
        pass

    @abstractmethod
    def aggregate_measurements(self, items: List[MeasuredAnnotationItem]) -> Typeable:
        """Combine the measurements for each TestItem into a test specific Typeable."""
        pass
