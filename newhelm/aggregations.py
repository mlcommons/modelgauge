from typing import Dict, List
from newhelm.placeholders import Result
from newhelm.single_turn_prompt_response import MeasuredTestItem


def get_measurements(
    measurement_name: str, items: List[MeasuredTestItem]
) -> List[float]:
    """Extract a desired measurement for all TestItems."""
    # Raises a KeyError if that test item is missing that measurement.
    return [item.measurements[measurement_name] for item in items]


def mean_of_measurement(measurement_name: str, items: List[MeasuredTestItem]) -> float:
    """Calculate the mean across all TestItems for a desired measurement."""
    measurements = get_measurements(measurement_name, items)
    total = sum(measurements)
    return total / len(measurements)


def mean_of_results(results: Dict[str, List[Result]]) -> float:
    """Calculate the mean across all Results from all Tests."""
    flattened = sum(results.values(), [])
    if len(flattened) == 0:
        return 0
    total = sum(r.value for r in flattened)
    return total / len(flattened)
