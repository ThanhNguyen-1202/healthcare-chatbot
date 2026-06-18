

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from app.core.medical_disclaimer import MEDICAL_DISCLAIMER, with_medical_disclaimer


def _to_plain_dict(data: Any) -> Dict[str, Any]:
    """Convert Pydantic models or mappings into a plain dictionary."""
    if data is None:
        return {}
    if hasattr(data, "model_dump"):
        return dict(data.model_dump())
    if isinstance(data, Mapping):
        return dict(data)
    return {"value": data}


def api_response(
    data: Optional[Any] = None,
    message: str = "",
    success: bool = True,
    errors: Optional[List[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compatibility: bool = True,
) -> Dict[str, Any]:
    payload = with_medical_disclaimer(_to_plain_dict(data))
    response_metadata = dict(metadata or {})
    response_metadata.setdefault("medical_disclaimer", MEDICAL_DISCLAIMER)
    response_metadata.setdefault("safety_notice", MEDICAL_DISCLAIMER)

    response: Dict[str, Any] = {
        "success": success,
        "data": payload,
        "message": message,
        "errors": errors or [],
        "metadata": response_metadata,
    }

    if compatibility:
        for key, value in payload.items():
            response.setdefault(key, value)

    return response
