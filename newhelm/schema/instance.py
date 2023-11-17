from dataclasses import dataclass, field
from typing import List, Optional

from newhelm.schema.traceable_object import TraceableObject


@dataclass(frozen=True)
class Reference:
    """
    A `Reference` specifies a possible output and how good/bad it is.  This
    could be used to represent multiple reference outputs which are all
    acceptable (e.g., in machine translation) or alternatives (e.g., in a
    multiple-choice exam).
    """

    text: str
    """The output"""

    # TODO How to represent what this reference implies about the SUT.


@dataclass(frozen=True)
class Instance(TraceableObject):
    """The basic unit of a test, storing the raw data."""

    text: str
    """The input"""

    # TODO Fields for the Image/video/audio data can go here.

    references: List[Reference]
    """References that helps us evaluate"""

    split: str
    """Split (e.g., train, valid, test)"""


@dataclass(frozen=True)
class InstancePromptBlock:
    """Block of text / other data derived from a single Instance."""

    text: str

    # TODO References to the Image/video/audio data can go here.

    instance_id: str
    """The Instance this was originally derived from."""


@dataclass(frozen=True, kw_only=True)
class InstancePromptTemplate(TraceableObject):
    """Contains all information required to make a specific prompt, without any SUT specific data"""

    eval_instance_block: InstancePromptBlock
    """Specify the evaluation text and where it came from."""

    reference_index: Optional[int] = None
    """Record the additional reference information used from the eval Instance, if any."""

    train_instances: List[InstancePromptBlock] = field(default_factory=list)
    """List the text and origin of in-context examples, if any."""

    # TODO Something about what adapter was used
    # TODO Something about what perturbations were made
