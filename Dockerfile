# ─────────────────────────────────────────────────────────────────────────────
# Voice Pipeline API  ·  Production Docker image
# ─────────────────────────────────────────────────────────────────────────────
#
# Three-stage build:
#   builder       – compiles Python wheels
#   model-fetch   – pre-downloads Whisper model (no auth needed)
#                   Indic model is downloaded at first runtime start
#                   (requires HF_TOKEN env var or .env file)
#   runtime       – minimal, non-root, production image
#
# Build args:
#   WHISPER_MODEL_SIZE   (default: base)
#   PYTHON_VERSION       (default: 3.11)
#
# Build examples:
#   docker build -t voice-pipeline .
#   docker build --build-arg WHISPER_MODEL_SIZE=small -t voice-pipeline .
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Separate layer so code changes don't bust the pip cache
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Whisper model pre-fetch ─────────────────────────────────────────
# Whisper weights (~140 MB for 'base') are baked into the image so the
# container needs no internet access on first boot.
# Indic model is intentionally NOT pre-baked here because it requires an
# HF_TOKEN which must never be a build ARG (visible in docker history).
# The Indic model is downloaded at first startup into the model_cache volume.
FROM python:${PYTHON_VERSION}-slim AS model-fetch

ARG WHISPER_MODEL_SIZE=base

# Copy only the packages needed for the download script
COPY --from=builder /install /usr/local

WORKDIR /models

RUN python3 - <<'PYEOF'
import sys, os
size = os.environ.get("WHISPER_MODEL_SIZE", "base") or "base"
print(f"Pre-downloading Whisper '{size}' …")
try:
    import whisper
    whisper.load_model(size, download_root="/models")
    print("Whisper pre-download complete.")
except Exception as e:
    print(f"WARNING: Whisper pre-download failed: {e}", file=sys.stderr)
    print("Model will be downloaded on first container startup.", file=sys.stderr)
PYEOF


# ── Stage 3: production runtime ───────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="Voice Pipeline API" \
      org.opencontainers.image.description="Production ASR API – Indic Conformer + Whisper" \
      org.opencontainers.image.licenses="MIT"

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg   : audio decoding (MP3, WebM, AAC, …) used by the ffmpeg subprocess path
# libsndfile1 : required by soundfile Python library for WAV/FLAC decoding
# curl     : used by the HEALTHCHECK command
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ── Python packages (from builder) ────────────────────────────────────────────
COPY --from=builder /install /usr/local

# ── Pre-fetched Whisper weights (from model-fetch) ────────────────────────────
# Placed under /app/model_cache_seed so the volume mount at /app/model_cache
# can overlay it at runtime.  A custom entrypoint seeds the volume on first run.
COPY --from=model-fetch /models /app/model_cache_seed

# ── Application setup ─────────────────────────────────────────────────────────
WORKDIR /app

# Non-root user for security
RUN useradd --system --uid 1000 --create-home --shell /bin/bash appuser \
 && chown appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser app/            ./app/
COPY --chown=appuser:appuser gunicorn.conf.py .
COPY --chown=appuser:appuser run.py           .
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
 && mkdir -p /app/model_cache \
 && chown appuser:appuser /app/model_cache

# Persist downloaded models across container restarts
VOLUME ["/app/model_cache"]

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    # Suppress HuggingFace progress bars in logs
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    # Where models are stored inside the container
    MODEL_CACHE_DIR=/app/model_cache

EXPOSE 8000

# tini is a minimal init process that reaps zombies and forwards signals
# correctly to gunicorn (important for graceful shutdown)
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["gunicorn", "app.main:app", "-c", "gunicorn.conf.py"]

# Health check – uses the lightweight /v1/live endpoint
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD curl -sf http://localhost:8000/v1/live | grep -q '"alive":true' || exit 1
