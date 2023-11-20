from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from newhelm.schema.instance import InstancePromptTemplate

from newhelm.schema.prompt import Prompt
from newhelm.sut import SUT, SUTResponse


@dataclass(frozen=True)
class SUTInteraction:
    """Placeholder wrapping up the full interaction."""

    # TODO make this multi-turn, or wrap in something that is.
    prompt: Prompt
    response: SUTResponse


class SUTInteractionStrategy(ABC):
    """Base class for how to directly interaction with a SUT."""

    @abstractmethod
    def interact(self, suts: List[SUT]) -> List[List[SUTInteraction]]:
        pass


class StaticPromptResponseInteraction(SUTInteractionStrategy):
    """Collect single turn SUT responses for tests that use a static prompt set."""

    def __init__(self, prompt_templates: List[InstancePromptTemplate]):
        # At construction the perturbations+adapter have already been applied.
        self.prompt_templates = prompt_templates

    def interact(self, suts: List[SUT]) -> List[List[SUTInteraction]]:
        results: List[List[SUTInteraction]] = []
        for sut in suts:
            single_sut_interactions = []
            for template in self.prompt_templates:
                prompt = sut.specialize(template)
                response = sut.evaluate(prompt)
                single_sut_interactions.append(SUTInteraction(prompt, response))
            results.append(single_sut_interactions)
        return results


class HumanInTheLoopTest(SUTInteractionStrategy):
    """Manage having humans do multi-turn interacts with the SUTs."""

    def interact(self, suts: List[SUT]):
        """Placeholder."""
