from dataclasses import dataclass, field
from typing import List, Optional

from newhelm.schema.traceable_object import TraceableObject


@dataclass(frozen=True)
class PromptSUTSpecialization:
    sut_id: str
    num_in_context_examples_truncated: int = 0
    num_characters_truncated: int = 0


@dataclass(frozen=True, kw_only=True)
class Prompt(TraceableObject):
    """The exact data sent to the SUT."""

    text: str
    """The text sent to the model."""

    # TODO Fields for the Image/video/audio data can go here.

    template_id: str
    """The ID of the prompt template this prompt was derived from."""

    sut_specialization: Optional[PromptSUTSpecialization] = None
    """Details about how this prompt was changed beyond the template to fit this SUT."""
