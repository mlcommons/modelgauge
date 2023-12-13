from dataclasses import dataclass, field
from typing import List

from newhelm.sut import Interaction


@dataclass(frozen=True)
class Annotation:
    placeholder: str


@dataclass(frozen=True)
class AnnotatedInteraction:
    interaction: Interaction
    annotations: List[Annotation] = field(default_factory=list)
