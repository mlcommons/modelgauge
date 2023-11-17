from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class DatasetConfig:
    """Specifies which set of Items to use, and any filtering to perform."""

    dataset_name: str
    # TODO Fill in
    pass


@dataclass(frozen=True)
class PertubationConfig:
    """Specifies a way of modifying Items before making them into prompts."""

    # TODO Fill in
    pass


@dataclass(frozen=True)
class AdapterConfig:
    """Specifies how to structure the Items into prompts."""

    # TODO Fill in
    pass


# Config for where prompts come from


@dataclass(frozen=True)
class StaticPromptResponseConfig:
    """Specifies how to go from a list of static Items to a list of Prompts."""

    dataset_config: DatasetConfig
    perturbation_methods: List[PertubationConfig]
    adapter_method: AdapterConfig


@dataclass(frozen=True)
class HumanInTheLoopConfig:
    """Specifies what humans to recruit, what instructions to give them, so they can have multiturn interactions with the SUT."""

    # TODO Fill in
    pass


# Config for if we need external annotation of the responses


@dataclass(frozen=True)
class ModelAnnotationConfig:
    """Specifies how to collect data about a SUT Response from an auxilary model.

    For example, PerspectiveAPI.
    """

    # TODO Fill in
    pass


@dataclass(frozen=True)
class HumanAnnotationConfig:
    """Specifies how to collect data from humans reviewing SUT responses."""

    # TODO Fill in
    pass


# Config for how to convert a (prompt, response, annotation) tuple into a numerical Result


@dataclass(frozen=True)
class MetricConfig:
    """Specifies what functions to apply to produce a numerical Result."""

    # TODO Fill in
    pass


@dataclass(frozen=True, kw_only=True)
class TestConfig:
    """Specifies everything needed to perform a test end-to-end."""

    name: str

    # A test should define exactly one of the following
    prompt_response_config: Optional[StaticPromptResponseConfig] = None
    human_in_the_loop_config: Optional[HumanInTheLoopConfig] = None

    # A test can set zero or more of each of these.
    model_annotation_config: List[ModelAnnotationConfig] = field(default_factory=list)
    human_annotator_config: List[HumanAnnotationConfig] = field(default_factory=list)

    # A test should define at least one of the following.
    metric_config: List[MetricConfig]

    __test__ = False  # Pytest tries to run this as a fixture.
