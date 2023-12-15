from typing import List
from newhelm.annotation import AnnotatedInteraction
from newhelm.base_test import BasePromptResponseTest
from newhelm.load_plugins import load_plugins
from newhelm.sut import PromptResponseSUT
from newhelm.general import get_concrete_subclasses
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


if __name__ == "__main__":
    load_plugins()

    for test_key, test in TESTS.items():
        print("\n\nStarting a new test:", test_key)
        # This demo assumes all tests are BasePromptResponseTest.
        assert isinstance(test, BasePromptResponseTest)
        # Only have to make the prompt templates once, reusable across SUTs.
        prompt_templates = test.make_prompt_templates()
        for sut_key, sut in SUTS.items():
            print("Running sut:", sut_key)
            # This demo assumes all suts are PromptResponseSUT.
            assert isinstance(sut, PromptResponseSUT)
            interactions = []
            for template in prompt_templates:
                # Splitting specialize from evaluate allows us to track the prompts created.
                prompt = sut.specialize(template)
                interaction = sut.evaluate(prompt)
                print("Completed interaction:", interaction)
                interactions.append(interaction)
            # Here is where an annotator would go
            annotated = [
                AnnotatedInteraction(interaction) for interaction in interactions
            ]
            print("Results:", test.calculate_results(annotated))
