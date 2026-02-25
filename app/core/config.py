"""
Application configuration via environment variables and defaults.
Uses pydantic-settings for typed, validated configuration.
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "Voice Pipeline API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Production-ready ASR API supporting 22 Indic languages "
        "(ai4bharat/indic-conformer-600m-multilingual) and English (Whisper)."
    )
    ENVIRONMENT: Literal["development", "staging", "production"] = "production"
    DEBUG: bool = False

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Keep 1; models are loaded per-process
    LOG_LEVEL: str = "INFO"

    # ── HuggingFace auth ──────────────────────────────────────────────────────
    # Required for gated models (e.g. ai4bharat/indic-conformer-600m-multilingual).
    # Set via HF_TOKEN env var or in .env file.
    HF_TOKEN: str | None = None

    # ── Model configuration ───────────────────────────────────────────────────
    INDIC_MODEL_ID: str = "ai4bharat/indic-conformer-600m-multilingual"
    WHISPER_MODEL_SIZE: str = "base"          # tiny | base | small | medium | large
    DEVICE: str = "auto"                      # auto | cpu | cuda | cuda:0 …
    TORCH_DTYPE: str = "auto"                 # auto | float16 | float32
    MODEL_CACHE_DIR: str = "./model_cache"

    # Warm-up on startup (runs a silent inference to JIT-compile kernels)
    WARMUP_ON_STARTUP: bool = True

    # ── Audio constraints ─────────────────────────────────────────────────────
    MAX_AUDIO_DURATION_SECONDS: float = 300.0   # 5 minutes
    MIN_AUDIO_DURATION_SECONDS: float = 0.1
    TARGET_SAMPLE_RATE: int = 16000
    MAX_UPLOAD_SIZE_MB: float = 50.0
    ALLOWED_AUDIO_FORMATS: list[str] = Field(
        default=["wav", "flac", "mp3", "ogg", "opus", "webm", "m4a", "aac"]
    )

    # ── Concurrency / rate limiting ───────────────────────────────────────────
    MAX_CONCURRENT_REQUESTS: int = 4   # semaphore cap for GPU inference
    REQUEST_TIMEOUT_SECONDS: float = 120.0

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = Field(default=["*"])
    CORS_ALLOW_CREDENTIALS: bool = False

    @field_validator("DEVICE", mode="before")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v != "auto":
            return v
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @field_validator("TORCH_DTYPE", mode="before")
    @classmethod
    def resolve_dtype(cls, v: str) -> str:
        return v  # resolved lazily in model loaders

    @property
    def max_upload_bytes(self) -> int:
        return int(self.MAX_UPLOAD_SIZE_MB * 1024 * 1024)

    @property
    def is_gpu(self) -> bool:
        return self.DEVICE.startswith("cuda")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
