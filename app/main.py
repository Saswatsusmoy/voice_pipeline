"""
Voice Pipeline API – FastAPI application entry-point.

Startup sequence:
  1. Configure structured logging
  2. Initialise ModelManager singleton
  3. Load both ASR models concurrently (in thread pool)
  4. Optional warm-up inference
  5. Register API routers

Shutdown sequence:
  6. Unload models and free GPU memory
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import get_logger, new_request_id, request_id_var, setup_logging
from app.models.model_manager import create_model_manager, get_model_manager
from app.schemas.response import ErrorDetail, ErrorResponse

# ── Bootstrap logging before anything else ────────────────────────────────────
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ── Lifespan context (replaces on_event) ──────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage model loading and cleanup around the application lifecycle."""
    manager = create_model_manager()
    app.state.manager = manager   # accessible via request.app.state.manager

    logger.info(
        "Starting %s v%s in '%s' environment.",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )

    await manager.startup()
    yield
    await manager.shutdown()

    logger.info("%s shutdown complete.", settings.APP_NAME)


# ── Application factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # Disable default validation error handler – we supply our own
        # (see exception handler below)
        swagger_ui_parameters={"displayRequestDuration": True},
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID + access logging middleware ────────────────────────────────
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_var.set(rid)
        t0 = time.perf_counter()

        try:
            response: Response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            request_id_var.reset(token)

        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"

        logger.info(
            "%s %s → %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "method":  request.method,
                "path":    request.url.path,
                "status":  response.status_code,
                "dur_ms":  round(elapsed_ms, 1),
            },
        )
        return response

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = request_id_var.get("-")
        logger.exception(
            "Unhandled exception on %s %s rid=%s",
            request.method,
            request.url.path,
            rid,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                request_id=rid,
                error=ErrorDetail(
                    code="INTERNAL_SERVER_ERROR",
                    message="An unexpected internal error occurred.",
                ),
            ).model_dump(),
        )

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        rid = request_id_var.get("-")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                request_id=rid,
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Request validation failed.",
                    details={"errors": exc.errors()},
                ),
            ).model_dump(),
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(v1_router)

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/v1/health",
        }

    return app


# ── Application instance (used by uvicorn) ────────────────────────────────────
app = create_app()
