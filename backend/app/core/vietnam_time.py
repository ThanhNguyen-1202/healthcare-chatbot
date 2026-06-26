"""Vietnam timezone helpers used across the backend.

The project stores and returns timestamps in Asia/Ho_Chi_Minh (+07:00)
so chat history, prediction history and logs match Vietnam local time.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

VIETNAM_TIMEZONE_NAME = "Asia/Ho_Chi_Minh"
VIETNAM_TZ = ZoneInfo(VIETNAM_TIMEZONE_NAME)


def now_vietnam() -> datetime:
    """Return the current timezone-aware datetime in Vietnam local time."""
    return datetime.now(VIETNAM_TZ)


def now_vietnam_iso() -> str:
    """Return the current Vietnam time as an ISO-8601 string."""
    return now_vietnam().isoformat()
