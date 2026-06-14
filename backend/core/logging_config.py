import json
import logging
from datetime import datetime, timezone
from typing import Any

from backend.core.config import settings

AGENT_LOGGER_NAME = "mirrorcue"
LOGGER = logging.getLogger(AGENT_LOGGER_NAME)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "agent": getattr(record, "agent", "SYSTEM"),
            "user_id": getattr(record, "user_id", None),
            "analysis_id": getattr(record, "analysis_id", None),
            "event": getattr(record, "event", record.getMessage()),
            "duration_ms": getattr(record, "duration_ms", 0),
            "details": getattr(record, "details", {}),
        }
        if record.exc_info:
            payload["details"] = {
                **payload["details"],
                "exception": self.formatException(record.exc_info),
            }
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    if LOGGER.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    LOGGER.addHandler(handler)
    LOGGER.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    LOGGER.propagate = False


def log_event(
    *,
    level: int = logging.INFO,
    agent: str = "SYSTEM",
    user_id: str | None = None,
    analysis_id: str | None = None,
    event: str,
    duration_ms: int = 0,
    details: dict[str, Any] | None = None,
    exc_info: bool = False,
) -> None:
    configure_logging()
    LOGGER.log(
        level,
        event,
        extra={
            "agent": agent,
            "user_id": user_id,
            "analysis_id": analysis_id,
            "event": event,
            "duration_ms": duration_ms,
            "details": details or {},
        },
        exc_info=exc_info,
    )
