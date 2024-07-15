from pydantic import BaseModel


class SafetyModelMeasurement(BaseModel):
    valid_test: bool
    human_majority_is_unsafe: float
    sut_is_unsafe: float
