"""LLM extraction API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.core.api_response import api_response
from app.schemas.llm_extract import LLMExtractRequest
from app.services.llm_extractor_service import extract_intake_with_llm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/extract")
async def llm_extract_endpoint(request: LLMExtractRequest) -> dict:
    """Extract structured medical intake fields from free text."""
    result = extract_intake_with_llm(request.message)
    return api_response(data=result, message="LLM intake extraction completed")
