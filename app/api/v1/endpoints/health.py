"""
GET /v1/health  – liveness + readiness probe
GET /v1/ready   – Kubernetes readiness probe (returns 503 until models load)
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Response, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.model_manager import ModelManager, get_model_manager
from app.schemas.response import HealthResponse, ModelInfo

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["Operations"])

_APP_START = time.perf_counter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & readiness check",
    description=(
        "Returns detailed status about loaded models, GPU availability, "
        "and runtime statistics. Suitable for monitoring dashboards."
    ),
)
async def health(
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    stats = manager.get_stats()

    both_loaded = stats.indic_loaded and stats.whisper_loaded
    overall = "healthy" if both_loaded else ("degraded" if (stats.indic_loaded or stats.whisper_loaded) else "unhealthy")

    if overall == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif overall == "degraded":
        response.status_code = status.HTTP_200_OK  # still serves some requests

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        device=stats.device,
        cuda_available=stats.cuda_available,
        cuda_device_name=stats.cuda_device_name,
        models={
            "indic": ModelInfo(
                loaded=stats.indic_loaded,
                device=manager.indic.device,
                model_id=settings.INDIC_MODEL_ID,
            ),
            "whisper": ModelInfo(
                loaded=stats.whisper_loaded,
                device=manager.whisper.device,
                model_id=f"whisper-{settings.WHISPER_MODEL_SIZE}",
            ),
        },
        uptime_seconds=round(time.perf_counter() - _APP_START, 1),
        total_requests=stats.total_requests,
        failed_requests=stats.failed_requests,
        startup_time_seconds=round(stats.startup_time_seconds, 2),
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Kubernetes readiness probe",
    description="Returns 200 once both models are loaded, 503 otherwise.",
    response_model=None,
)
async def ready(
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
) -> dict:
    stats = manager.get_stats()
    if not (stats.indic_loaded and stats.whisper_loaded):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False, "message": "Models are still loading."}
    return {"ready": True}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Kubernetes liveness probe",
    description="Always returns 200 while the process is running.",
    response_model=None,
)
async def live() -> dict:
    return {"alive": True}
