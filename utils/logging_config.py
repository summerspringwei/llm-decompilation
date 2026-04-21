"""Centralized logging configuration for the llm-decompilation project.

Usage::

    from utils.logging_config import get_logger
    logger = get_logger(__name__)

Call ``setup_logging()`` once at programme entry (e.g. in ``__main__`` blocks)
to set the global level and optional log-file output.  Modules that only need
a logger should use ``get_logger()`` — it works even if ``setup_logging`` has
not been called yet (defaults to INFO level).
"""

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = (
    "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)

_setup_done = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """Configure the root logger once.

    Safe to call multiple times — subsequent calls are no-ops.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``).
        log_file: If provided, log output is *also* written to this file.
        fmt: Format string for log records.
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring basic config is in place."""
    if not _setup_done:
        setup_logging()
    return logging.getLogger(name)


# Backward-compatible alias — existing code imports ``from utils.mylogger import logger``.
logger = get_logger("llm_decompilation")
