from __future__ import annotations

import logging


logger = logging.getLogger("resume_rag")


def trace_event(stage: str, message: str) -> None:
    logger.info("%s: %s", stage, message)
