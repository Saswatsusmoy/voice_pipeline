"""
POST /v1/transcribe          – transcribe an uploaded audio file
POST /v1/transcribe/url      – (future) transcribe from a remote URL
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.core.config import get_settings
from app.core.logging import get_logger, request_id_var
from app.models.indic_model import INDIC_LANGUAGES
from app.models.model_manager import ModelManager, get_model_manager
from app.schemas.response import ErrorDetail, ErrorResponse, TranscriptionResponse
from app.services.transcription import WHISPER_LANGUAGES, transcribe_audio
from app.utils.audio_utils import AudioValidationError, AudioProcessingError

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

# combined set of valid language codes
_ALL_LANGUAGES = set(INDIC_LANGUAGES.keys()) | set(WHISPER_LANGUAGES.keys())


# ─── Endpoint ────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio or parameters"},
        422: {"description": "Validation error"},
        429: {"description": "Too many concurrent requests"},
        503: {"model": ErrorResponse, "description": "Model not ready"},
    },
    summary="Transcribe audio",
    description="""
Upload an audio file and receive its transcription.

**Language routing**
- Pass an Indic language code (`hi`, `bn`, `ta`, …) → routed to
  **ai4bharat/indic-conformer-600m-multilingual**.
- Pass an ISO 639-1 code for any other language (e.g. `en`, `fr`, `de`) or
  `null` for auto-detection → routed to **OpenAI Whisper**.

**Decode mode** (Indic only)
- `ctc`  – faster, slightly lower accuracy
- `rnnt` – slower, higher accuracy

**Supported formats**: WAV · FLAC · MP3 · OGG · OPUS · WebM · M4A · AAC
**Max file size**: configured via `MAX_UPLOAD_SIZE_MB` (default 50 MB)
**Max duration**: configured via `MAX_AUDIO_DURATION_SECONDS` (default 300 s)
""",
)
async def transcribe(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="Audio file to transcribe."),
    ],
    language: Annotated[
        str | None,
        Form(
            description=(
                "BCP-47 / ISO 639-1 language code. "
                "Indic codes: hi, bn, ta, … | Other: en, fr, de, … | "
                "Omit for Whisper auto-detection."
            )
        ),
    ] = None,
    decode_mode: Annotated[
        Literal["ctc", "rnnt"],
        Form(description="Indic-only decoding strategy. Ignored for Whisper."),
    ] = "ctc",
    task: Annotated[
        Literal["transcribe", "translate"],
        Form(description="Whisper task. 'translate' outputs English. Ignored for Indic."),
    ] = "transcribe",
    word_timestamps: Annotated[
        bool,
        Form(description="Include word-level timestamps (Whisper only)."),
    ] = False,
    initial_prompt: Annotated[
        str | None,
        Form(description="Optional hint / context for Whisper decoding."),
    ] = None,
    beam_size: Annotated[
        int,
        Form(description="Whisper beam search width (1–10). Default 5.", ge=1, le=10),
    ] = 5,
    manager: ModelManager = Depends(get_model_manager),
) -> TranscriptionResponse:
    request_id: str = request_id_var.get("-")

    # ── Resolve language ──────────────────────────────────────────────────────
    if language is not None:
        language = language.strip().lower()
        if language not in _ALL_LANGUAGES:
            manager.record_request(failed=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorDetail(
                    code="UNSUPPORTED_LANGUAGE",
                    message=(
                        f"Language code '{language}' is not supported. "
                        f"See GET /v1/languages for the full list."
                    ),
                ).model_dump(),
            )

    # ── Read file bytes ───────────────────────────────────────────────────────
    try:
        raw = await file.read()
    except Exception as exc:
        manager.record_request(failed=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                code="FILE_READ_ERROR",
                message=f"Could not read uploaded file: {exc}",
            ).model_dump(),
        ) from exc
    finally:
        await file.close()

    if not raw:
        manager.record_request(failed=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                code="EMPTY_FILE", message="Uploaded file is empty."
            ).model_dump(),
        )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = await asyncio.wait_for(
            transcribe_audio(
                raw_audio=raw,
                filename=file.filename,
                language=language or "en",   # fallback for Whisper auto-detect
                decode_mode=decode_mode,
                task=task,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
                beam_size=beam_size,
                request_id=request_id,
                manager=manager,
            ),
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
    except AudioValidationError as exc:
        manager.record_request(failed=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                code="AUDIO_VALIDATION_ERROR", message=str(exc)
            ).model_dump(),
        ) from exc
    except AudioProcessingError as exc:
        manager.record_request(failed=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                code="AUDIO_PROCESSING_ERROR", message=str(exc)
            ).model_dump(),
        ) from exc
    except ValueError as exc:
        manager.record_request(failed=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                code="INVALID_PARAMETER", message=str(exc)
            ).model_dump(),
        ) from exc
    except RuntimeError as exc:
        manager.record_request(failed=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                code="MODEL_NOT_READY", message=str(exc)
            ).model_dump(),
        ) from exc
    except asyncio.TimeoutError:
        manager.record_request(failed=True)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=ErrorDetail(
                code="INFERENCE_TIMEOUT",
                message=(
                    f"Inference did not complete within "
                    f"{settings.REQUEST_TIMEOUT_SECONDS:.0f}s."
                ),
            ).model_dump(),
        )
    except Exception as exc:
        manager.record_request(failed=True)
        logger.exception("Unhandled error during transcription request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred.",
            ).model_dump(),
        ) from exc

    manager.record_request(failed=False)
    return result
