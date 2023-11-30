from newhelm.prompt import InstancePromptTemplate, Prompt


class SUTResponse:
    """Placeholder for what comes back from a SUT."""


class SUT:
    """Placeholder"""

    def specialize(self, template: InstancePromptTemplate) -> Prompt:
        return Prompt()

    def evaluate(self, prompt: Prompt) -> SUTResponse:
        return SUTResponse()
