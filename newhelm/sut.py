from dataclasses import dataclass

from newhelm.schema.instance import InstancePromptTemplate
from newhelm.schema.prompt import Prompt


class SUTResponse:
    """Placeholder for what comes back from a SUT."""


class SUT:
    """Placeholder"""

    def specialize(self, template: InstancePromptTemplate) -> Prompt:
        pass

    def evaluate(self, prompt: Prompt) -> SUTResponse:
        pass
