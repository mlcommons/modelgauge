from pydantic import BaseModel


class SafetyModelMeasurement(BaseModel):
    valid_test: bool
    human_majority_is_safe: float
    sut_is_safe: float
