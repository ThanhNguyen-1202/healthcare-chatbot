from __future__ import annotations

import logging
import sys
from datetime import datetime

from app.core.vietnam_time import VIETNAM_TZ


class VietnamTimeFormatter(logging.Formatter):
    """Render log timestamps in Asia/Ho_Chi_Minh instead of server/UTC time."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, VIETNAM_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure process-wide console logging with Vietnam local timestamps."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        VietnamTimeFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=False,
    )
