"""Chat API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.core.api_response import api_response
from app.core.config import settings
from app.core.rate_limit import limiter
from app.db.repositories.session_repo import SessionRepository
from app.schemas.chat import ChatRequest
from app.services.chat_service import process_chat
from app.services.intake_service import build_intake_from_messages, get_missing_fields

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/")
@limiter.limit(settings.chat_rate_limit)
async def chat_endpoint(request: Request, chat_request: ChatRequest) -> dict:
    """Process one chat turn with per-client rate limiting."""
    result = await process_chat(chat_request)
    return api_response(data=result, message="Chat response generated")


@router.get("/session/{session_id}")
async def get_chat_session(session_id: str) -> dict:
    """Return a stored chat session by ID."""
    session_repo = SessionRepository()
    session = await session_repo.get_session(session_id)

    if not session:
        return api_response(
            data={"message": "Session not found"},
            success=False,
            message="Session not found",
            errors=[{"code": "SESSION_NOT_FOUND", "session_id": session_id}],
        )

    if "_id" in session:
        session["_id"] = str(session["_id"])

    return api_response(data=session, message="Session loaded")


@router.get("/session/{session_id}/intake")
async def get_session_intake(session_id: str) -> dict:
    """Return intake data reconstructed from a stored session."""
    session_repo = SessionRepository()
    session = await session_repo.get_session(session_id)

    if not session:
        return api_response(
            data={"message": "Session not found"},
            success=False,
            message="Session not found",
            errors=[{"code": "SESSION_NOT_FOUND", "session_id": session_id}],
        )

    collected_data = session.get("intake_snapshot")

    if not collected_data:
        messages = session.get("messages", [])
        collected_data = build_intake_from_messages(messages)

    missing_fields = get_missing_fields(collected_data)

    return api_response(
        data={
            "session_id": session_id,
            "collected_data": collected_data,
            "missing_fields": missing_fields,
        },
        message="Session intake loaded",
    )

