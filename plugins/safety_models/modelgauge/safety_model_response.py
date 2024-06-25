from pydantic import BaseModel
from typing import List, Optional


class SafetyModelResponse(BaseModel):
    is_safe: bool
    safety_categories: List[str]
    reasoning: Optional[str] = None
