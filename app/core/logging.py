"""
Logging setup for Voice Pipeline API.

Two modes selected by ENVIRONMENT:

  development           Coloured, human-readable text on stdout.
                        Example:
                          09:41:23.456  INFO      models.whisper       Whisper 'base' loaded in 0.77s
                          09:41:24.012  INFO      app.main             POST /v1/transcribe → 200  [dur_ms=245.3  rid=a1b2c3d4]

  staging / production  Newline-delimited JSON on stdout (ingest with ELK, Datadog, GCP, etc.).
                        Example:
                          {"ts":"2026-02-25T09:41:24.012Z","level":"INFO","logger":"app.main",
                           "rid":"a1b2c3d4","msg":"POST /v1/transcribe → 200","dur_ms":245.3}

Extra structured context can be attached to any log call via extra=:
    logger.info("request done", extra={"status": 200, "dur_ms": 42.1})
In dev mode these appear as inline  key=value  pairs.
In JSON mode they are promoted to top-level keys.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from app.core.config import get_settings

# Per-request correlation ID, propagated through the async context.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

settings = get_settings()

# ── ANSI colours (dev only) ───────────────────────────────────────────────────

_R   = "\033[0m"          # reset
_GRY = "\033[38;5;244m"   # grey  – timestamps, brackets
_CYN = "\033[36m"         # cyan  – logger name
_GRN = "\033[32m"         # green – INFO
_YLW = "\033[33m"         # yellow – WARNING
_RED = "\033[31m"         # red   – ERROR
_BLD = "\033[1;31m"       # bold red – CRITICAL

_LEVEL_COLOUR = {
    "DEBUG":    _GRY,
    "INFO":     _GRN,
    "WARNING":  _YLW,
    "ERROR":    _RED,
    "CRITICAL": _BLD,
}

# Standard LogRecord attributes — never echoed as extra context.
_RECORD_ATTRS: frozenset[str] = frozenset({
    "name", "msg", "args", "created", "levelname", "levelno",
    "pathname", "filename", "module", "funcName", "lineno",
    "exc_info", "exc_text", "stack_info", "thread", "threadName",
    "processName", "process", "taskName", "message",
    "relativeCreated", "msecs", "asctime",
})


# ── Development formatter ─────────────────────────────────────────────────────


class DevFormatter(logging.Formatter):
    """
    Coloured, columnar output for local development.

    Layout (fixed-width columns):
      HH:MM:SS.mmm  LEVEL     logger.name               message  [k=v …]
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        hms = time.strftime("%H:%M:%S", time.localtime(record.created))
        ts = f"{_GRY}{hms}.{int(record.msecs):03d}{_R}"

        colour = _LEVEL_COLOUR.get(record.levelname, "")
        level_s = f"{colour}{record.levelname:<8}{_R}"

        # Last two logger-name segments  (e.g. app.models.indic → models.indic)
        parts = record.name.split(".")
        short = ".".join(parts[-2:]) if len(parts) >= 2 else record.name
        name_s = f"{_CYN}{short:<26}{_R}"

        msg = record.getMessage()

        # Inline extra fields
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RECORD_ATTRS and not k.startswith("_")
        }
        extra_s = ""
        if extras:
            pairs = "  ".join(f"{_GRY}{k}{_R}={v}" for k, v in extras.items())
            extra_s = f"  {_GRY}[{_R}{pairs}{_GRY}]{_R}"

        # Correlation ID from context (shown for every line, skipped when no request active)
        rid = request_id_var.get("-")
        rid_s = f"  {_GRY}rid={_R}{rid}" if rid != "-" else ""

        line = f"{ts}  {level_s}  {name_s}  {msg}{extra_s}{rid_s}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


# ── Production JSON formatter ─────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """
    Newline-delimited JSON for structured log ingestion.

    Always-present keys:
        ts      ISO-8601 UTC timestamp with milliseconds  (2026-02-25T09:41:24.012Z)
        level   DEBUG | INFO | WARNING | ERROR | CRITICAL
        logger  dotted Python logger name
        rid     per-request correlation ID
        msg     rendered log message

    Conditional keys:
        src     module:func:line – added for WARNING and above
        exc     formatted traceback – added when exc_info is present

    Any extra= keys supplied at the call site are promoted to top-level keys.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()

        hms = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        obj: dict[str, Any] = {
            "ts":     f"{hms}.{int(record.msecs):03d}Z",
            "level":  record.levelname,
            "logger": record.name,
            "rid":    request_id_var.get("-"),
            "msg":    record.message,
        }

        # Source location only for warnings / errors – keeps INFO lines lean
        if record.levelno >= logging.WARNING:
            obj["src"] = f"{record.module}:{record.funcName}:{record.lineno}"

        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)

        # Promote caller-supplied extra= fields to top-level JSON keys
        for k, v in record.__dict__.items():
            if k not in _RECORD_ATTRS and not k.startswith("_"):
                obj[k] = v

        return json.dumps(obj, ensure_ascii=False, default=str)


# ── Setup ─────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    """Configure the root logger once at application start."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        DevFormatter() if settings.ENVIRONMENT == "development" else JSONFormatter()
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence verbose third-party loggers
    for noisy in ("transformers", "torch", "torchaudio", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
