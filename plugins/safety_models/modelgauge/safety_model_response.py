from pydantic import BaseModel
from typing import List, Optional


class SafetyModelResponse(BaseModel):
    is_safe: bool
    safety_categories: List[str]
    is_valid: bool = True
    reasoning: Optional[str] = None
    total_tokens_used: Optional[float] = None
