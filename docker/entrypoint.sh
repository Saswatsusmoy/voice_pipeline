#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Container entrypoint
#
# Both models are downloaded at runtime on first startup:
#   • Whisper: downloaded from OpenAI (no auth required)
#   • Indic Conformer: downloaded from HuggingFace (requires HF_TOKEN)
#
# Models are cached in /app/model_cache volume and persist across restarts.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Hand off to CMD (gunicorn …) ─────────────────────────────────────────────
exec "$@"