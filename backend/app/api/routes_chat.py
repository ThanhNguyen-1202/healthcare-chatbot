"""Chat API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.core.api_response import api_response
from app.core.config import settings
from app.core.rate_limit import limiter
from app.db.repositories.session_repo import SessionRepository
from app.schemas.chat import ChatRequest
from app.services.chat_service import process_chat
from app.services.intake_service import build_intake_from_messages, get_missing_fields

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _owner_error_response(status_code: int, code: str, message: str, **extra) -> JSONResponse:
    """Return an API-shaped error with the intended HTTP status code."""
    return JSONResponse(
        status_code=status_code,
        content=api_response(
            data={"message": message, **extra},
            success=False,
            message=message,
            errors=[{"code": code, **extra}],
        ),
    )


def _is_session_owner(session: dict, device_id: str | None) -> bool:
    """A session can be read only by the device_id that created it."""
    supplied_device_id = str(device_id or "").strip()
    stored_device_id = str(session.get("device_id") or "").strip()
    return bool(supplied_device_id) and bool(stored_device_id) and supplied_device_id == stored_device_id


@router.post("/")
@limiter.limit(settings.chat_rate_limit)
async def chat_endpoint(request: Request, chat_request: ChatRequest) -> dict:
    """Process one chat turn with per-client rate limiting."""
    result = await process_chat(chat_request)
    return api_response(data=result, message="Chat response generated")


@router.get("/session/{session_id}")
async def get_chat_session(
    session_id: str,
    device_id: str | None = Query(default=None),
) -> dict:
    """Return a stored chat session only for the device that owns it."""
    session_repo = SessionRepository()
    session = await session_repo.get_session(session_id)

    if not session:
        return _owner_error_response(
            404,
            "SESSION_NOT_FOUND",
            "Session not found",
            session_id=session_id,
        )

    if not _is_session_owner(session, device_id):
        return _owner_error_response(
            403,
            "SESSION_FORBIDDEN",
            "Session does not belong to this device",
            session_id=session_id,
        )

    if "_id" in session:
        session["_id"] = str(session["_id"])

    return api_response(data=session, message="Session loaded")


@router.get("/session/{session_id}/intake")
async def get_session_intake(
    session_id: str,
    device_id: str | None = Query(default=None),
) -> dict:
    """Return intake data only for the device that owns the session."""
    session_repo = SessionRepository()
    session = await session_repo.get_session(session_id)

    if not session:
        return _owner_error_response(
            404,
            "SESSION_NOT_FOUND",
            "Session not found",
            session_id=session_id,
        )

    if not _is_session_owner(session, device_id):
        return _owner_error_response(
            403,
            "SESSION_FORBIDDEN",
            "Session does not belong to this device",
            session_id=session_id,
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

