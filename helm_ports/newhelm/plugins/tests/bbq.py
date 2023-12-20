import os
from typing import List
from helm.benchmark.scenarios.scenario import create_scenario
from helm.benchmark.run_specs import get_bbq_spec
from helm.benchmark.data_preprocessor import DataPreprocessor
from helm.benchmark.adaptation.adapters.adapter_factory import AdapterFactory
from helm.benchmark.run_expander import ModelRunExpander
from newhelm.base_test import BasePromptResponseTest
from newhelm.placeholders import Measurement, Prompt, Result

from newhelm.single_turn_prompt_response import AnnotatedTestItem, MeasuredTestItem, PromptWithContext, TestItem

class FakeWindowService:
    def fits_within_context_window(self, *args, **kwargs):
        return True

class BBQ(BasePromptResponseTest):
    def __init__(self):
        self.run_spec = get_bbq_spec("all")
        # Get a valid model so that get_adapter works. We're going to ignore it anyway.
        self.run_spec = ModelRunExpander("text").expand(self.run_spec)[0]

    def make_test_items(self) -> List[TestItem]:
        scenario = create_scenario(self.run_spec.scenario_spec)

        scenario_cache_path = os.path.join("benchmark_output", "scenarios", scenario.name)
        instances = scenario.get_instances(scenario_cache_path)
        instances = DataPreprocessor(self.run_spec.data_augmenter_spec).preprocess(instances)


        adapter = AdapterFactory.get_adapter(self.run_spec.adapter_spec, None)
        # Monkey patch the window service
        adapter.window_service = FakeWindowService()
        scenario_state = adapter.adapt(instances, 1)

        # This ignores train trials, etc
        test_items = []
        for state in scenario_state.request_states:
            prompt_text = state.request.prompt
            test_items.append(TestItem(prompts=[PromptWithContext(Prompt(prompt_text), context=state)]))


        return test_items

    def measure_quality(self, item: AnnotatedTestItem) -> List[Measurement]:
        """Use the SUT responses with annotations to determine how well the SUT did on this TestItem."""
        # TODO
        return []

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        """Combine the measurements for each TestItem into a list of Results."""
        # TODO
        return []




