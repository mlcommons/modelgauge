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
class Item:
    """The basic unit of a test, storing the raw data."""

    text: str
    """The input"""

    # TODO Fields for the Image/video/audio data can go here.

    references: List[Reference]
    """References that helps us evaluate"""

    split: str
    """Split (e.g., train, valid, test)"""


@dataclass(frozen=True)
class ItemPromptBlock:
    """Block of text / other data derived from a single Item."""

    text: str

    # TODO References to the Image/video/audio data can go here.

    item_id: str
    """The Item this was originally derived from."""


@dataclass(frozen=True, kw_only=True)
class ItemPromptTemplate(TraceableObject):
    """Contains all information required to make a specific prompt, without any SUT specific data"""

    eval_item_block: ItemPromptBlock
    """Specify the evaluation text and where it came from."""

    reference_index: Optional[int] = None
    """Record the additional reference information used from the eval Item, if any."""

    train_items: List[ItemPromptBlock] = field(default_factory=list)
    """List the text and origin of in-context examples, if any."""

    # TODO Something about what adapter was used
    # TODO Something about what perturbations were made
