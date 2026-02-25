"""
Wrapper around OpenAI Whisper for English (and multilingual) transcription.

Uses the `openai-whisper` package for loading and inference.
The model is loaded once and kept in memory.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import torch

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Whisper model size → approximate VRAM usage
# tiny  ~39 M params  (~150 MB)
# base  ~74 M params  (~290 MB)
# small ~244 M params (~970 MB)
# medium ~769 M params (~3.1 GB)
# large ~1.5 B params  (~5.8 GB)

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


class WhisperASRModel:
    """
    Thread-safe singleton wrapper for OpenAI Whisper.

    Usage
    -----
    model = WhisperASRModel()
    model.load()
    result = model.transcribe(wav_numpy, language="en")
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._loaded = False
        self._device: str = settings.DEVICE
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model_size: str = settings.WHISPER_MODEL_SIZE

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download (if needed) and load Whisper into memory. Idempotent."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            logger.info(
                "Loading Whisper '%s' on device='%s' …",
                self._model_size,
                self._device,
            )
            t0 = time.perf_counter()
            import whisper  # deferred import

            self._model = whisper.load_model(
                self._model_size,
                device=self._device,
                download_root=settings.MODEL_CACHE_DIR,
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                "Whisper '%s' loaded in %.2fs (device=%s).",
                self._model_size,
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
                logger.info("Whisper model unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def transcribe(
        self,
        wav: np.ndarray,
        language: str | None = "en",
        task: str = "transcribe",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        beam_size: int = 5,
        best_of: int = 5,
        condition_on_previous_text: bool = True,
    ) -> dict[str, Any]:
        """
        Transcribe a 1-D float32 numpy array at 16 kHz.

        Parameters
        ----------
        wav                      : numpy float32 array, shape (T,)
        language                 : BCP-47 language code or None (auto-detect)
        task                     : "transcribe" or "translate" (English output)
        initial_prompt           : Whisper context prompt
        word_timestamps          : Include per-word timing
        temperature              : Decoding temperature(s)
        beam_size                : Beam search width
        best_of                  : Number of candidates for sampling
        condition_on_previous_text: Feed prior segment text as context

        Returns
        -------
        Whisper result dict containing:
            text          : full transcript
            language      : detected/specified language
            segments      : list of segment dicts (id, start, end, text, …)
        """
        if not self._loaded:
            raise RuntimeError("WhisperASRModel.load() must be called first.")

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = self._model.transcribe(
                wav,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                condition_on_previous_text=condition_on_previous_text,
                fp16=self._device.startswith("cuda"),
            )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Whisper transcription: lang=%s task=%s duration=%.2fs inf=%.1fms",
            result.get("language"),
            task,
            len(wav) / 16_000,
            elapsed,
        )
        return result

    # ── Warm-up ────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Run a short silent inference to JIT-compile CUDA kernels."""
        if not self._loaded:
            return
        logger.info("Warming up Whisper model …")
        dummy = np.zeros(16_000, dtype=np.float32)  # 1 second of silence
        try:
            self.transcribe(dummy, language="en")
            logger.info("Whisper warm-up complete.")
        except Exception as exc:
            logger.warning("Whisper warm-up failed: %s", exc)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_size(self) -> str:
        return self._model_size
