"""
Transcription service layer.

Bridges the HTTP endpoints to the ASR model wrappers, handling:
 - language routing (Indic vs Whisper)
 - audio pre-processing (via audio_utils)
 - async execution in a thread-pool (non-blocking event loop)
 - concurrency limiting via ModelManager semaphore
 - structured response building
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.indic_model import INDIC_LANGUAGES
from app.models.model_manager import ModelManager
from app.schemas.response import Segment, TranscriptionResponse, WordTimestamp
from app.utils.audio_utils import (
    AudioValidationError,
    preprocess_audio,
    validate_audio_bytes,
    wav_to_numpy,
)

logger = get_logger(__name__)
settings = get_settings()

# Language code → human-readable name for Whisper (common subset)
WHISPER_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "fa": "Persian",
}


def _language_name(code: str) -> str:
    if code in INDIC_LANGUAGES:
        return INDIC_LANGUAGES[code]
    return WHISPER_LANGUAGES.get(code, code.upper())


def _is_indic(language: str) -> bool:
    return language in INDIC_LANGUAGES


def _segments_from_whisper(raw_segments: list) -> list[Segment]:
    out = []
    for seg in raw_segments:
        words: list[WordTimestamp] | None = None
        raw_words = seg.get("words")
        if raw_words:
            words = [
                WordTimestamp(
                    word=w["word"],
                    start=w["start"],
                    end=w["end"],
                    probability=w.get("probability"),
                )
                for w in raw_words
            ]
        out.append(
            Segment(
                id=seg.get("id", 0),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                words=words,
                avg_logprob=seg.get("avg_logprob"),
                no_speech_prob=seg.get("no_speech_prob"),
                compression_ratio=seg.get("compression_ratio"),
            )
        )
    return out


# ─── Main service function ────────────────────────────────────────────────────


async def transcribe_audio(
    *,
    raw_audio: bytes,
    filename: str | None,
    language: str,
    decode_mode: Literal["ctc", "rnnt"] = "ctc",
    task: str = "transcribe",
    word_timestamps: bool = False,
    initial_prompt: str | None = None,
    beam_size: int = 5,
    request_id: str,
    manager: ModelManager,
) -> TranscriptionResponse:
    """
    Full pipeline: validate → pre-process → infer → build response.

    Raises
    ------
    AudioValidationError  – cheap, early rejection (no model needed)
    ValueError            – unsupported language / decode mode
    RuntimeError          – model not loaded
    """
    t_start = time.perf_counter()

    # 1. Lightweight byte-level validation (fast, no I/O)
    validate_audio_bytes(raw_audio, filename)

    loop = asyncio.get_event_loop()

    # 2. Pre-process audio in thread pool (CPU-bound resampling)
    wav, duration = await loop.run_in_executor(
        None, preprocess_audio, raw_audio, settings.TARGET_SAMPLE_RATE
    )

    # 3. Route to the correct model and run inference
    async with manager.semaphore:
        if _is_indic(language):
            # ── Indic Conformer ───────────────────────────────────────────────
            if not manager.indic.is_loaded:
                raise RuntimeError("Indic ASR model is not loaded.")

            text: str = await loop.run_in_executor(
                None,
                lambda: manager.indic.transcribe(wav, language, decode_mode),
            )

            processing_ms = (time.perf_counter() - t_start) * 1000
            return TranscriptionResponse(
                request_id=request_id,
                text=text,
                language=language,
                language_name=_language_name(language),
                model="indic-conformer-600m",
                decode_mode=decode_mode,
                duration_seconds=round(duration, 3),
                processing_time_ms=round(processing_ms, 1),
                segments=None,
                word_timestamps=None,
            )

        else:
            # ── OpenAI Whisper ────────────────────────────────────────────────
            if not manager.whisper.is_loaded:
                raise RuntimeError("Whisper ASR model is not loaded.")

            wav_np = wav_to_numpy(wav)

            result: dict = await loop.run_in_executor(
                None,
                lambda: manager.whisper.transcribe(
                    wav_np,
                    language=language,
                    task=task,
                    initial_prompt=initial_prompt,
                    word_timestamps=word_timestamps,
                    beam_size=beam_size,
                ),
            )

            segments = _segments_from_whisper(result.get("segments", []))

            # Collect word-level timestamps from segments if requested
            all_words: list[WordTimestamp] | None = None
            if word_timestamps:
                all_words = [w for seg in segments if seg.words for w in seg.words]

            processing_ms = (time.perf_counter() - t_start) * 1000
            detected_lang = result.get("language", language)
            return TranscriptionResponse(
                request_id=request_id,
                text=result.get("text", "").strip(),
                language=detected_lang,
                language_name=_language_name(detected_lang),
                model=f"whisper-{manager.whisper.model_size}",
                decode_mode=None,
                duration_seconds=round(duration, 3),
                processing_time_ms=round(processing_ms, 1),
                segments=segments if segments else None,
                word_timestamps=all_words,
            )
