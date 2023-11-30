from typing import List
from newhelm.interaction_annotator import InteractionAnnotation, InteractionAnnotator
from newhelm.metric import MetricCalculator
from newhelm.sut import SUT
from newhelm.sut_interaction import SUTInteraction, SUTInteractionStrategy


def run(
    interaction_maker: SUTInteractionStrategy,
    suts: List[SUT],
    annotators: List[InteractionAnnotator],
    metrics: List[MetricCalculator],
):
    interactions: List[SUTInteraction] = []
    for interaction_task in interaction_maker.get_tasks(suts):
        interactions.extend(interaction_task.execute())

    annotations: List[InteractionAnnotation] = []
    for annotator in annotators:
        for annotation_task in annotator.get_tasks(interactions):
            annotations.extend(annotation_task.execute())

    results = []
    for metric in metrics:
        results.extend(metric.calculate(interactions, annotations))
    return results
