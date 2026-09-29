from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

_logger = logging.getLogger("islamic_research_ai")


def configure_logging() -> None:
    if _logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


def log_event(event: str, **fields: object) -> None:
    safe = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    _logger.info(json.dumps(safe, ensure_ascii=False, default=str, sort_keys=True))
