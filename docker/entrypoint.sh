#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Container entrypoint
#
# Responsibilities:
#   1. Seed the model_cache volume with pre-baked Whisper weights on first run.
#      (The volume starts empty; model_cache_seed was baked in during docker build.)
#   2. Exec the CMD passed by docker (gunicorn …) so tini can manage it as PID 1.
#
# Volume layout:
#   /app/model_cache_seed   – baked-in Whisper weights (read-only, in image layer)
#   /app/model_cache        – volume mount, persisted across restarts
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SEED_DIR="/app/model_cache_seed"
CACHE_DIR="/app/model_cache"
STAMP="${CACHE_DIR}/.seeded"

# ── Seed Whisper weights on first start ──────────────────────────────────────
if [[ -d "${SEED_DIR}" && ! -f "${STAMP}" ]]; then
    echo "[entrypoint] Seeding model cache from image …"
    # Copy only if seed dir has content (build may have skipped download on failure)
    if [[ -n "$(ls -A "${SEED_DIR}" 2>/dev/null)" ]]; then
        cp -r "${SEED_DIR}/." "${CACHE_DIR}/"
        echo "[entrypoint] Model cache seeded."
    else
        echo "[entrypoint] Seed dir is empty — model will be downloaded on first use."
    fi
    touch "${STAMP}"
fi

# ── Hand off to CMD (gunicorn …) ─────────────────────────────────────────────
exec "$@"
