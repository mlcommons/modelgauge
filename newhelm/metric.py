from typing import List


class Result:
    """Placeholder for the numeric output of a Metric."""


class MetricCalculator:
    """Placeholder."""

    def calculate(self, sut_interactions, annotations) -> List[Result]:
        results = []
        for interaction in sut_interactions:
            # Add a dummy result so we can count interactions.
            results.append(Result())
        for annotation in annotations:
            # Add a dummy result so we can count annotations
            results.append(Result())
        return results
