"""
GET /v1/languages           – list all supported languages
GET /v1/languages/{code}    – details for a single language code
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.indic_model import INDIC_LANGUAGES
from app.schemas.response import LanguageEntry, LanguagesResponse
from app.services.transcription import WHISPER_LANGUAGES

router = APIRouter(prefix="/languages", tags=["Languages"])

# Build a unified, sorted language table once at module load time
_LANGUAGE_TABLE: list[LanguageEntry] = sorted(
    [
        LanguageEntry(
            code=code,
            name=name,
            model="indic-conformer-600m",
            decode_modes=["ctc", "rnnt"],
        )
        for code, name in INDIC_LANGUAGES.items()
    ]
    + [
        LanguageEntry(
            code=code,
            name=name,
            model="whisper-base",
            decode_modes=None,
        )
        for code, name in WHISPER_LANGUAGES.items()
        if code not in INDIC_LANGUAGES   # avoid duplicates (e.g. 'ta')
    ],
    key=lambda e: e.name,
)

_LANGUAGE_MAP: dict[str, LanguageEntry] = {e.code: e for e in _LANGUAGE_TABLE}


@router.get(
    "",
    response_model=LanguagesResponse,
    summary="List supported languages",
    description=(
        "Returns all language codes understood by the API. "
        "Indic languages are handled by the Conformer model; "
        "all others are handled by Whisper."
    ),
)
async def list_languages() -> LanguagesResponse:
    return LanguagesResponse(total=len(_LANGUAGE_TABLE), languages=_LANGUAGE_TABLE)


@router.get(
    "/{code}",
    response_model=LanguageEntry,
    summary="Get language details",
    responses={
        404: {"description": "Language code not found"},
    },
)
async def get_language(code: str) -> LanguageEntry:
    code = code.strip().lower()
    entry = _LANGUAGE_MAP.get(code)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Language code '{code}' is not supported.",
        )
    return entry
