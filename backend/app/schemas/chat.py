from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.medical_disclaimer import MEDICAL_DISCLAIMER


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    device_id: Optional[str] = None  
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    missing_fields: List[str] = Field(default_factory=list)
    collected_data: Dict[str, Any] = Field(default_factory=dict)
    next_step: str = "continue"
    device_id: Optional[str] = None  #  Cho frontend/DB biết phiên này thuộc thiết bị nào.
    session_restarted: bool = False  #  True khi backend tự tách phiên mới sau phiên đã kết thúc.