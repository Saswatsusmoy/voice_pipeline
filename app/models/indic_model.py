"""
Wrapper around ai4bharat/indic-conformer-600m-multilingual.

Supports both CTC and RNNT decoding strategies.
The underlying AutoModel is loaded once and kept in memory.
"""

from __future__ import annotations

import threading
import time
from typing import Literal

import torch

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ─── Supported language codes ─────────────────────────────────────────────────

INDIC_LANGUAGES: dict[str, str] = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

DecodingMode = Literal["ctc", "rnnt"]


class IndicASRModel:
    """
    Thread-safe singleton wrapper for the Indic Conformer model.

    Usage
    -----
    model = IndicASRModel()
    model.load()
    text = model.transcribe(wav_tensor, language="hi", decode_mode="ctc")
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._loaded = False
        self._device: str = settings.DEVICE
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download (if needed) and load the model into memory.  Idempotent."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:   # double-checked locking
                return
            logger.info(
                "Loading Indic Conformer model '%s' on device='%s' ...",
                settings.INDIC_MODEL_ID,
                self._device,
            )
            t0 = time.perf_counter()
            from transformers import AutoModel  # deferred import

            load_kwargs: dict = {
                "trust_remote_code": True,
                "cache_dir": settings.MODEL_CACHE_DIR,
            }
            if settings.HF_TOKEN:
                load_kwargs["token"] = settings.HF_TOKEN

            self._model = AutoModel.from_pretrained(
                settings.INDIC_MODEL_ID,
                **load_kwargs,
            )
            # The model internally handles device placement, but we move it
            # explicitly so GPU memory is allocated predictably.
            self._model = self._model.to(self._device)
            self._model.eval()
            elapsed = time.perf_counter() - t0
            logger.info(
                "Indic Conformer loaded in %.2fs (device=%s).",
                elapsed,
                self._device,
            )
            self._loaded = True

    def unload(self) -> None:
        """Release GPU/CPU memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                if self._device.startswith("cuda"):
                    torch.cuda.empty_cache()
                self._loaded = False
                logger.info("Indic Conformer model unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def transcribe(
        self,
        wav: torch.Tensor,
        language: str,
        decode_mode: DecodingMode = "ctc",
    ) -> str:
        """
        Transcribe a mono 16 kHz float32 waveform tensor.

        Parameters
        ----------
        wav         : torch.Tensor  shape (1, T)
        language    : ISO 639 code – must be in INDIC_LANGUAGES
        decode_mode : "ctc" (faster) or "rnnt" (more accurate)

        Returns
        -------
        Transcription string (stripped).
        """
        if not self._loaded:
            raise RuntimeError("IndicASRModel.load() must be called first.")
        if language not in INDIC_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Supported: {sorted(INDIC_LANGUAGES.keys())}"
            )

        wav = wav.to(self._device)
        t0 = time.perf_counter()

        with torch.inference_mode():
            result = self._model(wav, language, decode_mode)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Indic transcription: lang=%s mode=%s duration=%.2fs inf=%.1fms",
            language,
            decode_mode,
            wav.shape[-1] / 16_000,
            elapsed,
        )
        return result.strip() if isinstance(result, str) else str(result).strip()

    # ── Warm-up ────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Run a short silent inference to JIT-compile CUDA kernels."""
        if not self._loaded:
            return
        logger.info("Warming up Indic Conformer model …")
        dummy = torch.zeros(1, 16_000, device=self._device)   # 1 second of silence
        try:
            self.transcribe(dummy, language="hi", decode_mode="ctc")
            logger.info("Indic Conformer warm-up complete.")
        except Exception as exc:            # warm-up failure is non-fatal
            logger.warning("Indic Conformer warm-up failed: %s", exc)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @staticmethod
    def supported_languages() -> dict[str, str]:
        return dict(INDIC_LANGUAGES)
