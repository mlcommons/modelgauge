from typing import List
from newhelm.metric import MetricCalculator, Result
from newhelm.interaction_annotator import (
    HumanAnnotator,
    InteractionAnnotator,
    ModelBasedAnnotator,
)
from newhelm.schema.instance import InstancePromptTemplate

from newhelm.schema.test_config import StaticPromptResponseConfig, TestConfig
from newhelm.sut import SUT
from newhelm.sut_interaction import (
    HumanInTheLoopTest,
    SUTInteractionStrategy,
    StaticPromptResponseInteraction,
)


def end_to_end_run(
    interaction_maker: SUTInteractionStrategy,
    suts: List[SUT],
    annotators: List[InteractionAnnotator],
    metrics: List[MetricCalculator],
) -> List[Result]:
    # TODO `interact` and `annotate` should be asynchronous as either one could have a human
    # in the loop / make slow external calls.

    # In HELM's run_one method, this is everything up to and including `executor.execute`
    sut_interactions = interaction_maker.interact(suts)
    # TODO write down all interactions

    # In HELM's run_one, "annotations" are pushed into metric.evaluate, making it very
    # ugly to collect human feedback. Here we pull it out into its own step.
    annotations = {}
    for annotator in annotators:
        annotations[annotator.name] = annotator.annotate(sut_interactions)
    # TODO write down all annotations.

    # In HELM's run_one, this is `metric.evaluate`
    results = []
    for metric in metrics:
        results.extend(metric.calculate(sut_interactions, annotations))
    return results


def run_from_config(test_config: TestConfig, suts: List[SUT]):
    interaction_maker: SUTInteractionStrategy
    if test_config.prompt_response_config:
        # TODO have a way to load an existing test version as well.
        templates = make_new_test_version(test_config.prompt_response_config)
        interaction_maker = StaticPromptResponseInteraction(templates)
    elif test_config.human_in_the_loop_config:
        interaction_maker = HumanInTheLoopTest()
    else:
        raise ValueError("Must specify where prompts come from.")

    annotators: List[InteractionAnnotator]
    for model_annotator_config in test_config.model_annotation_config:
        annotators.append(ModelBasedAnnotator())
    for human_annotator_config in test_config.human_annotator_config:
        annotators.append(HumanAnnotator())

    metrics = [MetricCalculator() for metric in test_config.metric_config]

    return end_to_end_run(interaction_maker, suts, annotators, metrics)


def make_new_test_version(
    config: StaticPromptResponseConfig,
) -> List[InstancePromptTemplate]:
    # TODO Select Instances from storage
    # TODO Apply perturbations
    # TODO Apply Adapter
    pass
