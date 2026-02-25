"""
Central model registry and lifecycle manager.

A single ModelManager instance is created at startup and injected via
FastAPI's dependency-injection system.  It owns both ASR models and
exposes an asyncio Semaphore to cap simultaneous GPU inferences.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.indic_model import IndicASRModel
from app.models.whisper_model import WhisperASRModel

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ModelStats:
    """Runtime statistics exposed via the /health endpoint."""

    indic_loaded: bool = False
    whisper_loaded: bool = False
    device: str = "unknown"
    torch_version: str = ""
    cuda_available: bool = False
    cuda_device_name: str | None = None
    startup_time_seconds: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """
    Manages both ASR models and exposes them as FastAPI dependencies.

    Thread-safety
    -------------
    Model.load() uses its own threading.Lock.
    The asyncio Semaphore guards concurrent inference calls on the event loop.
    """

    def __init__(self) -> None:
        self.indic = IndicASRModel()
        self.whisper = WhisperASRModel()
        self._semaphore: asyncio.Semaphore | None = None
        self._stats = ModelStats()
        self._startup_at: float | None = None

    # ── Semaphore (created inside the running event loop) ─────────────────────

    def _ensure_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        return self._semaphore

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._ensure_semaphore()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """
        Called from FastAPI lifespan context.
        Loads both models concurrently in a thread pool.

        A failure in one model is logged and recorded but never propagates –
        the app starts in degraded mode and serves whichever model did load.
        """
        self._startup_at = time.perf_counter()
        logger.info("ModelManager: starting up …")

        loop = asyncio.get_event_loop()

        # return_exceptions=True prevents one model failure from killing startup
        results = await asyncio.gather(
            loop.run_in_executor(None, self._load_safe, "indic", self.indic.load),
            loop.run_in_executor(None, self._load_safe, "whisper", self.whisper.load),
            return_exceptions=True,
        )

        # Log any unexpected errors from the gather itself (shouldn't happen,
        # but _load_safe already handles them internally)
        for r in results:
            if isinstance(r, BaseException):
                logger.error("Unexpected error during model gather: %s", r)

        self._stats.indic_loaded = self.indic.is_loaded
        self._stats.whisper_loaded = self.whisper.is_loaded
        self._stats.device = self.indic.device
        self._stats.torch_version = torch.__version__
        self._stats.cuda_available = torch.cuda.is_available()
        if torch.cuda.is_available():
            self._stats.cuda_device_name = torch.cuda.get_device_name(0)

        # Optional warm-up (only for models that actually loaded)
        if settings.WARMUP_ON_STARTUP:
            await asyncio.gather(
                loop.run_in_executor(None, self.indic.warmup),
                loop.run_in_executor(None, self.whisper.warmup),
                return_exceptions=True,
            )

        self._stats.startup_time_seconds = time.perf_counter() - self._startup_at

        if not self._stats.indic_loaded and not self._stats.whisper_loaded:
            logger.error(
                "ModelManager: BOTH models failed to load. "
                "Check HF_TOKEN and network connectivity."
            )
        else:
            logger.info(
                "ModelManager: ready in %.2fs (indic=%s, whisper=%s, device=%s).",
                self._stats.startup_time_seconds,
                self._stats.indic_loaded,
                self._stats.whisper_loaded,
                self._stats.device,
            )

    @staticmethod
    def _load_safe(name: str, load_fn) -> None:
        """Call load_fn(); catch and log any exception instead of re-raising."""
        try:
            load_fn()
        except Exception as exc:
            logger.error(
                "Failed to load '%s' model: %s – continuing without it.",
                name,
                exc,
            )

    async def shutdown(self) -> None:
        """Called from FastAPI lifespan context on shutdown."""
        logger.info("ModelManager: shutting down …")
        loop = asyncio.get_event_loop()
        await asyncio.gather(
            loop.run_in_executor(None, self.indic.unload),
            loop.run_in_executor(None, self.whisper.unload),
        )
        logger.info("ModelManager: shutdown complete.")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> ModelStats:
        return self._stats

    def record_request(self, *, failed: bool = False) -> None:
        self._stats.total_requests += 1
        if failed:
            self._stats.failed_requests += 1


# ── Application-level singleton ───────────────────────────────────────────────
# Instantiated once here; injected via get_model_manager().

_manager: ModelManager | None = None


def create_model_manager() -> ModelManager:
    global _manager
    _manager = ModelManager()
    return _manager


def get_model_manager() -> ModelManager:
    """FastAPI Depends() callable."""
    if _manager is None:
        raise RuntimeError("ModelManager has not been initialised.")
    return _manager
