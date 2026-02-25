"""
Audio loading, validation, and preprocessing utilities.
All heavy I/O is done in a thread-pool to avoid blocking the event loop.

Decoding strategy (torchaudio's backend system is bypassed entirely to
avoid the torchcodec dependency introduced in torchaudio ≥ 2.5):

  1. soundfile  – fast, pure-Python; handles WAV, FLAC, OGG-Vorbis, AIFF
  2. ffmpeg subprocess – handles everything else (MP3, WebM, M4A, AAC, …)
                         ffmpeg is already present because Whisper requires it.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio  # still used for Resample transform

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ─── Exceptions ──────────────────────────────────────────────────────────────


class AudioValidationError(ValueError):
    """Raised when an uploaded audio file fails validation."""


class AudioProcessingError(RuntimeError):
    """Raised when audio cannot be decoded or re-sampled."""


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _detect_format(raw: bytes) -> str | None:
    """Best-effort format detection from magic bytes."""
    magic: dict[bytes, str] = {
        b"RIFF": "wav",
        b"fLaC": "flac",
        b"OggS": "ogg",
        b"\xff\xfb": "mp3",
        b"\xff\xf3": "mp3",
        b"\xff\xf2": "mp3",
        b"ID3": "mp3",
        b"\x1a\x45\xdf\xa3": "webm",
    }
    for sig, fmt in magic.items():
        if raw[: len(sig)] == sig:
            return fmt
    return None


def _load_via_soundfile(raw: bytes) -> tuple[torch.Tensor, int]:
    """
    Decode with soundfile directly (no torchaudio backend routing).
    Returns (C, T) float32 tensor and sample rate.
    Raises sf.SoundFileError for unsupported formats (e.g. MP3).
    """
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    # soundfile shape: (T, C) → want (C, T)
    wav = torch.from_numpy(data.T.copy())
    logger.debug("soundfile decoded: sr=%d channels=%d samples=%d", sr, wav.shape[0], wav.shape[1])
    return wav, sr


def _load_via_ffmpeg(raw: bytes) -> tuple[torch.Tensor, int]:
    """
    Decode arbitrary audio via system ffmpeg → raw float32 mono PCM.
    Returns (1, T) float32 tensor at the file's native sample rate.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise AudioProcessingError(
            "ffmpeg binary not found. Install with: sudo apt-get install ffmpeg"
        )

    # Step 1: probe original sample rate (fast, no decoding)
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                "pipe:0",
            ],
            input=raw,
            capture_output=True,
            timeout=10,
        )
        sr = int(probe.stdout.strip()) if probe.returncode == 0 and probe.stdout.strip() else 16_000
    except Exception:
        sr = 16_000  # will be resampled downstream anyway

    # Step 2: decode → raw interleaved float32 PCM, mono
    try:
        result = subprocess.run(
            [
                ffmpeg_bin,
                "-i", "pipe:0",
                "-f", "f32le",          # raw float32 little-endian PCM
                "-acodec", "pcm_f32le",
                "-ar", str(sr),         # preserve original sample rate
                "-ac", "1",             # mix to mono
                "-loglevel", "error",
                "pipe:1",
            ],
            input=raw,
            capture_output=True,
            timeout=120,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip()
        raise AudioProcessingError(f"ffmpeg decode failed: {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError("ffmpeg timed out while decoding audio.") from exc

    pcm = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if pcm.size == 0:
        raise AudioProcessingError("ffmpeg produced empty output — file may be corrupt or silent.")

    wav = torch.from_numpy(pcm).unsqueeze(0)  # (1, T)
    logger.debug("ffmpeg decoded: sr=%d samples=%d", sr, pcm.size)
    return wav, sr


def _load_audio(raw: bytes) -> tuple[torch.Tensor, int]:
    """
    Load audio bytes to a (C, T) float32 tensor + sample rate.

    Tries soundfile first (fast, no subprocess), then falls back to ffmpeg.
    Never touches torchaudio's backend/codec selection.
    """
    try:
        return _load_via_soundfile(raw)
    except Exception as sf_exc:
        logger.debug("soundfile failed (%s), trying ffmpeg …", sf_exc)

    return _load_via_ffmpeg(raw)


# ─── Public API ───────────────────────────────────────────────────────────────


def validate_audio_bytes(raw: bytes, filename: str | None = None) -> None:
    """
    Raises AudioValidationError if the raw bytes fail size or format checks.
    Called *before* decoding to give fast, cheap rejection.
    """
    max_bytes = settings.max_upload_bytes
    if len(raw) > max_bytes:
        raise AudioValidationError(
            f"File size {len(raw) / 1_048_576:.1f} MB exceeds the "
            f"{settings.MAX_UPLOAD_SIZE_MB:.0f} MB limit."
        )
    if len(raw) < 4:
        raise AudioValidationError("File is too small to be valid audio.")

    if filename:
        ext = Path(filename).suffix.lstrip(".").lower()
        allowed = [f.lower() for f in settings.ALLOWED_AUDIO_FORMATS]
        if ext and ext not in allowed:
            raise AudioValidationError(
                f"Extension '.{ext}' is not allowed. "
                f"Accepted: {', '.join(allowed)}."
            )


def preprocess_audio(
    raw: bytes,
    target_sr: int = 16_000,
) -> tuple[torch.Tensor, float]:
    """
    Load and pre-process audio bytes into a mono float32 waveform tensor.

    Returns
    -------
    wav : torch.Tensor  shape (1, T)  – mono, 16 kHz, float32
    duration_seconds : float
    """
    t0 = time.perf_counter()

    wav, orig_sr = _load_audio(raw)

    # ── Convert stereo/multi-channel to mono ──────────────────────────────────
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # ── Resample if needed ────────────────────────────────────────────────────
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr
        )
        wav = resampler(wav)

    # ── Validate duration ─────────────────────────────────────────────────────
    num_samples = wav.shape[-1]
    duration = num_samples / target_sr

    if duration < settings.MIN_AUDIO_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration {duration:.2f}s is below the minimum "
            f"{settings.MIN_AUDIO_DURATION_SECONDS}s."
        )
    if duration > settings.MAX_AUDIO_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration {duration:.1f}s exceeds the maximum "
            f"{settings.MAX_AUDIO_DURATION_SECONDS:.0f}s."
        )

    # ── Normalise to [-1, 1] (peak normalisation) ─────────────────────────────
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak

    # ── Ensure float32 ────────────────────────────────────────────────────────
    wav = wav.to(torch.float32)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(
        "Audio pre-processed: orig_sr=%d → %d Hz, duration=%.2fs, time=%.1fms",
        orig_sr,
        target_sr,
        duration,
        elapsed,
    )
    return wav, duration


def wav_to_numpy(wav: torch.Tensor) -> np.ndarray:
    """Return a 1-D float32 numpy array (squeezed mono tensor)."""
    return wav.squeeze().numpy().astype(np.float32)

