from pydantic import BaseModel
from typing import Dict, Optional


class SafetyEvalContext(BaseModel):
    uid: str
    prompt: str
    response: str
    hazard_category: str
    annotations: Dict[str, str]
    lg_1_is_safe: int
    lg_1_violation_categories: Optional[str] = None
