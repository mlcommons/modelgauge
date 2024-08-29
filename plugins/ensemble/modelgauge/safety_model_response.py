from typing import Any, List, Optional

from pydantic import BaseModel


class SafetyModelResponse(BaseModel):
    """Dataclass that all safety model responses should adhere to"""

    is_safe: bool
    safety_categories: List[str]
    is_valid: bool = True
    raw_response: Optional[str] = None
    reasoning: Optional[str] = None
    total_tokens_used: Optional[float] = None
    logprobs: Optional[Any] = None


class EnsembleSafetyModelResponse(SafetyModelResponse):
    """Dataclass for safety model responses computed by an ensemble

    Stores additional metadata on the original responses

    **joined_responses**: Store the individual SafetyModelResponse objects
    **is_unanimous**: Whether or not the is_safe votes were unanimous
    """

    joined_responses: List[SafetyModelResponse]
    is_unanimous: bool
