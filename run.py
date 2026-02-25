"""
Uvicorn entrypoint for local development.

Usage:
    python run.py
    python run.py --reload         # hot-reload (dev only)
"""

import argparse
import uvicorn
from app.core.config import get_settings

settings = get_settings()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Pipeline API")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--host", default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--workers", type=int, default=settings.WORKERS)
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_config=None,     # use our own logging setup
        access_log=False,    # we handle access logging in middleware
    )


if __name__ == "__main__":
    main()
