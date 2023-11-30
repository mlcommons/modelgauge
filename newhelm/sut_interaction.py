from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from newhelm.prompt import InstancePromptTemplate, Prompt
from newhelm.sut import SUT, SUTResponse
from newhelm.task import Task


@dataclass(frozen=True)
class SUTInteraction:
    """Placeholder wrapping up the full interaction."""

    # TODO make this multi-turn, or wrap in something that is.
    prompt: Prompt
    response: SUTResponse


class SUTInteractionTask(Task):
    @abstractmethod
    def execute(self) -> List[SUTInteraction]:
        pass


class SUTInteractionStrategy(ABC):
    """Base class for how to directly interaction with a SUT."""

    @abstractmethod
    def get_tasks(self, suts: List[SUT]) -> List[SUTInteractionTask]:
        pass


class StaticPromptResponseTask(SUTInteractionTask):
    def __init__(self, sut, template):
        self.sut = sut
        self.template = template

    def execute(self) -> List[SUTInteraction]:
        prompt = self.sut.specialize(self.template)
        response = self.sut.evaluate(prompt)
        return [SUTInteraction(prompt, response)]


class StaticPromptResponseStrategy(SUTInteractionStrategy):
    """Collect single turn SUT responses for tests that use a static prompt set."""

    def __init__(self, prompt_templates: List[InstancePromptTemplate]):
        # At construction the perturbations+adapter have already been applied.
        self.prompt_templates = prompt_templates

    def get_tasks(self, suts: List[SUT]) -> List[SUTInteractionTask]:
        tasks: List[SUTInteractionTask] = []
        for sut in suts:
            for template in self.prompt_templates:
                tasks.append(StaticPromptResponseTask(sut, template))
        return tasks
