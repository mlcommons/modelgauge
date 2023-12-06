from newhelm.interaction_annotator import (
    SimpleInteractionAnnotator,
    PerspectiveAPIAnnotatorClient,
)
from newhelm.metric import MetricCalculator
from newhelm.prompt import InstancePromptTemplate
import newhelm.runners.serial_runner as serial_runner
from newhelm.sut import SUT
from newhelm.sut_interaction import StaticPromptResponseStrategy


def test_run():
    results = serial_runner.run(
        # One prompt
        interaction_maker=StaticPromptResponseStrategy([InstancePromptTemplate()]),
        # One SUT
        suts=[SUT()],
        # One annotator
        annotators=[SimpleInteractionAnnotator(PerspectiveAPIAnnotatorClient())],
        # One metric
        metrics=[MetricCalculator()],
    )
    # Currently just outputting 1 result per interaction and 1 per annotation
    assert len(results) == 2
