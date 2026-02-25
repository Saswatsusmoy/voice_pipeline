"""
Gunicorn configuration for production ASGI serving.

Why gunicorn over plain uvicorn?
  - Proper signal handling (SIGTERM → graceful drain, SIGKILL as last resort)
  - Worker crash recovery without losing the whole process
  - Worker-level max_requests rotation to prevent long-lived memory leaks
  - Integration with process supervisors (systemd, k8s, etc.)

Worker count is intentionally 1.  Each worker loads BOTH ASR models into RAM
(~2 GB on CPU, more on GPU).  Running N workers = N × model memory.
The async FastAPI event loop + asyncio Semaphore in ModelManager already
handles concurrent requests within a single process.
"""

import os

# ── Binding ───────────────────────────────────────────────────────────────────
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8000")
bind = f"{host}:{port}"

# ── Worker class ──────────────────────────────────────────────────────────────
worker_class = "uvicorn.workers.UvicornWorker"

# Keep at 1 — ML models are memory-heavy; see docstring above.
workers = int(os.getenv("WORKERS", "1"))

# ── Timeouts ──────────────────────────────────────────────────────────────────
# Must be greater than REQUEST_TIMEOUT_SECONDS in config.py (default 120 s).
# Long audio files can take 60-90 s on CPU.
timeout = int(os.getenv("WORKER_TIMEOUT", "180"))
graceful_timeout = 30   # time for in-flight requests to finish before SIGKILL
keepalive = 5           # seconds to keep idle connections alive

# ── Request limits ────────────────────────────────────────────────────────────
# Rotate the worker after this many requests to prevent gradual memory growth.
# jitter prevents all workers restarting simultaneously (matters if workers > 1).
max_requests = int(os.getenv("MAX_REQUESTS", "500"))
max_requests_jitter = 50

# ── Logging ───────────────────────────────────────────────────────────────────
# "-" = stdout/stderr; captured by Docker / k8s log drivers.
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()

# Suppress gunicorn's own access log – our FastAPI middleware already logs
# every request with request_id and timing.
access_log_format = ""   # empty → still uses errorlog but no duplicate lines

# ── Process identity ──────────────────────────────────────────────────────────
proc_name = "voice-pipeline"

# ── Preloading ────────────────────────────────────────────────────────────────
# preload_app=False is intentional for ML workloads:
#   - True  → model loaded before fork  → workers share memory pages (CoW)
#             BUT PyTorch CUDA contexts cannot be safely forked → crashes on GPU
#   - False → each worker loads its own model copy → safe, predictable
preload_app = False

# ── Security : limit request line + header sizes ─────────────────────────────
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# ── Forwarded headers (when behind nginx/ALB) ─────────────────────────────────
# Trust X-Forwarded-For from the nginx container only.
forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
