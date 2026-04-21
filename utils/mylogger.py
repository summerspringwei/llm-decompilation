"""Backward-compatibility shim — prefer ``utils.logging_config`` for new code."""

from utils.logging_config import get_logger, logger, setup_logging  # noqa: F401

__all__ = ["get_logger", "logger", "setup_logging"]
