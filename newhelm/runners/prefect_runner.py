from prefect import flow, task
from typing import List, cast
from newhelm.interaction_annotator import InteractionAnnotation, InteractionAnnotator
from newhelm.metric import MetricCalculator
from newhelm.sut import SUT
from newhelm.sut_interaction import SUTInteraction, SUTInteractionStrategy
from newhelm.task import Task


@task
def task_runner(task: Task):
    return task.execute()


@flow
def interact(
    interaction_maker: SUTInteractionStrategy, suts: List[SUT]
) -> List[SUTInteraction]:
    futures = []
    for task in interaction_maker.get_tasks(suts):
        futures.append(task_runner.submit(task))
    results: List[SUTInteraction] = []
    for future in futures:
        # Futures came from SUTInteractionStrategy.get_tasks, so
        # we know they return List[SUTInteraction].
        results.extend(cast(List[SUTInteraction], future.result()))
    return results


@flow
def annotate(
    interactions, annotators: List[InteractionAnnotator]
) -> List[InteractionAnnotation]:
    futures = []
    for annotator in annotators:
        for task in annotator.get_tasks(interactions):
            futures.append(task_runner.submit(task))
    results: List[InteractionAnnotation] = []
    for future in futures:
        # Futures came from InteractionAnnotator.get_tasks, so we know
        # they return List[InteractionAnnotation]
        results.extend(cast(List[InteractionAnnotation], future.result()))
    return results


@flow
def run(
    interaction_maker: SUTInteractionStrategy,
    suts: List[SUT],
    annotators: List[InteractionAnnotator],
    metrics: List[MetricCalculator],
):
    interactions = interact(interaction_maker, suts)
    annotations = annotate(interactions, annotators)

    results = []
    for metric in metrics:
        results.extend(metric.calculate(interactions, annotations))
    return results
