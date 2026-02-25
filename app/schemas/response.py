"""
Pydantic response schemas for the Voice Pipeline API.
All timestamps are in ISO-8601 UTC.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# ─── Generic wrappers ─────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")
    details: dict[str, Any] | None = Field(None, description="Optional extra context.")


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail

    model_config = {"json_schema_extra": {
        "example": {
            "request_id": "a1b2c3d4e5f6",
            "error": {
                "code": "UNSUPPORTED_LANGUAGE",
                "message": "Language 'xx' is not supported for the Indic model.",
                "details": None,
            },
        }
    }}


# ─── Transcription responses ──────────────────────────────────────────────────


class WordTimestamp(BaseModel):
    word: str
    start: float = Field(..., description="Start time in seconds.")
    end: float = Field(..., description="End time in seconds.")
    probability: float | None = None


class Segment(BaseModel):
    id: int
    start: float = Field(..., description="Segment start time in seconds.")
    end: float = Field(..., description="Segment end time in seconds.")
    text: str
    words: list[WordTimestamp] | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class TranscriptionResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier.")
    text: str = Field(..., description="Full transcription text.")
    language: str = Field(..., description="Language code used for transcription.")
    language_name: str = Field(..., description="Human-readable language name.")
    model: str = Field(..., description="ASR model used: 'indic' or 'whisper'.")
    decode_mode: str | None = Field(None, description="Decoding mode (Indic only): 'ctc' or 'rnnt'.")
    duration_seconds: float = Field(..., description="Audio duration in seconds.")
    processing_time_ms: float = Field(..., description="Total inference time in milliseconds.")
    segments: list[Segment] | None = Field(None, description="Segment-level output (Whisper only).")
    word_timestamps: list[WordTimestamp] | None = Field(
        None, description="Word-level timestamps (Whisper, if requested)."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "request_id": "a1b2c3d4e5f6",
            "text": "नमस्ते, आप कैसे हैं?",
            "language": "hi",
            "language_name": "Hindi",
            "model": "indic",
            "decode_mode": "ctc",
            "duration_seconds": 2.5,
            "processing_time_ms": 312.4,
            "segments": None,
            "word_timestamps": None,
        }
    }}


# ─── Health responses ─────────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    loaded: bool
    device: str
    model_id: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    environment: str
    device: str
    cuda_available: bool
    cuda_device_name: str | None
    models: dict[str, ModelInfo]
    uptime_seconds: float
    total_requests: int
    failed_requests: int
    startup_time_seconds: float

    model_config = {"json_schema_extra": {
        "example": {
            "status": "healthy",
            "version": "1.0.0",
            "environment": "production",
            "device": "cuda",
            "cuda_available": True,
            "cuda_device_name": "NVIDIA A100",
            "models": {
                "indic": {"loaded": True, "device": "cuda", "model_id": "ai4bharat/indic-conformer-600m-multilingual"},
                "whisper": {"loaded": True, "device": "cuda", "model_id": "whisper-base"},
            },
            "uptime_seconds": 120.5,
            "total_requests": 42,
            "failed_requests": 0,
            "startup_time_seconds": 18.3,
        }
    }}


# ─── Languages responses ──────────────────────────────────────────────────────


class LanguageEntry(BaseModel):
    code: str
    name: str
    model: str = Field(..., description="Which ASR backend handles this language.")
    decode_modes: list[str] | None = Field(
        None, description="Supported decode modes (Indic only)."
    )


class LanguagesResponse(BaseModel):
    total: int
    languages: list[LanguageEntry]
