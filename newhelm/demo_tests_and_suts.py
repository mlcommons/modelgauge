from typing import List, Set, Type, TypeVar
from newhelm.annotation import AnnotatedInteraction
from newhelm.base_test import BasePromptResponseTest
from newhelm.load_plugins import load_plugins
from newhelm.sut import PromptResponseSUT

InT = TypeVar("InT")


def get_all_subclasses(cls: Type[InT]) -> Set[Type[InT]]:
    result = set()
    for subclass in cls.__subclasses__():
        result.add(subclass)
        result.update(get_all_subclasses(subclass))
    return result


if __name__ == "__main__":
    load_plugins()

    all_suts: List[PromptResponseSUT] = [
        cls() for cls in get_all_subclasses(PromptResponseSUT)  # type: ignore[type-abstract]
    ]

    for test_cls in get_all_subclasses(BasePromptResponseTest):  # type: ignore[type-abstract]
        print("\n\nStarting a new test:", test_cls)
        test = test_cls()
        prompt_templates = test.make_prompt_templates()
        for sut in all_suts:
            print("Running sut:", sut.__class__.__name__)
            interactions = []
            for template in prompt_templates:
                prompt = sut.specialize(template)
                interaction = sut.evaluate(prompt)
                print(interaction)
                interactions.append(interaction)
            # Here is where an annotator would go
            annotated = [
                AnnotatedInteraction(interaction) for interaction in interactions
            ]
            print("Results:", test.calculate_results(annotated))
