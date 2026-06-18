from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.medical_disclaimer import MEDICAL_DISCLAIMER


class PredictionRequest(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    temperature: Optional[float] = None
    pain_score: Optional[int] = Field(default=None, ge=0, le=10)
    duration: Optional[str] = None
    comorbidities: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    age: Optional[int] = None
    gender: Optional[str] = None
    is_pregnant: Optional[bool] = None
    evidence_codes: List[str] = Field(default_factory=list)
    top_k: int = 3


class PredictionResponse(BaseModel):
    matched: bool
    message: str
    possible_diseases: List[Dict[str, Any]] = Field(default_factory=list)
    triage_level: str
    triage_priority: int = 5
    triage_label: str = "Mức 5 - Tự theo dõi"
    action_recommendation: str = ""
    triage_reasons: List[str] = Field(default_factory=list)
    department: str
    medical_disclaimer: str = MEDICAL_DISCLAIMER
