from typing import List, Optional

from pydantic import BaseModel, Field

from app.core.medical_disclaimer import MEDICAL_DISCLAIMER


class LLMExtractRequest(BaseModel):
    message: str


class LLMExtractResponse(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    main_symptom: Optional[str] = None
    duration: Optional[str] = None
    temperature: Optional[float] = None
    pain_score: Optional[int] = Field(default=None, ge=0, le=10)  # [CHANGED] Chỉ giữ điểm đau dạng số 0-10.
    symptom_location: Optional[str] = None
    secondary_symptoms: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    comorbidities: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    is_pregnant: Optional[bool] = None
    ddxplus_evidences: List[str] = Field(default_factory=list)
    medical_disclaimer: str = MEDICAL_DISCLAIMER
