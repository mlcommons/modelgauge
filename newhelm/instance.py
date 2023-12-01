"""This file is a stripped down copy of HELM objects with the same name."""


from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Input:
    """
    The input of an `Instance`.
    """

    text: str = ""
    """The text of the input (e.g, passage to summarize, text for sentiment analysis, etc.)"""


@dataclass(frozen=True)
class PassageQuestionInput(Input):
    """
    Passage-question pair used for question answering Scenarios.
    """

    def __init__(
        self,
        passage: str,
        question: str,
        passage_prefix: str = "",
        question_prefix: str = "Question: ",
        separator: str = "\n",
    ):
        super().__init__(
            f"{passage_prefix}{passage}{separator}{question_prefix}{question}"
        )


@dataclass(frozen=True)
class Output:
    """
    The output of a `Reference`.
    """

    text: str = ""
    """The text of the output."""


@dataclass(frozen=True)
class Reference:
    """
    A `Reference` specifies a possible output and how good/bad it is.  This
    could be used to represent multiple reference outputs which are all
    acceptable (e.g., in machine translation) or alternatives (e.g., in a
    multiple-choice exam).
    """

    output: Output
    """The output"""

    tags: List[str]
    """Extra metadata (e.g., whether it's correct/factual/toxic)"""


@dataclass(frozen=True, eq=False)
class Instance:
    """
    An `Instance` represents one data point that we're evaluating on (e.g., one
    question in a QA task).
    Note: `eq=False` means that we hash by the identity.
    """

    input: Input
    """The input"""

    references: List[Reference]
    """References that helps us evaluate"""

    split: Optional[str] = None
    """Split (e.g., train, valid, test)"""
