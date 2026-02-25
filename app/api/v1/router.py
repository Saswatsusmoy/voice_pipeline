"""
API v1 router – aggregates all endpoint sub-routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.languages import router as languages_router
from app.api.v1.endpoints.transcribe import router as transcribe_router

router = APIRouter(prefix="/v1")

router.include_router(health_router)
router.include_router(languages_router)
router.include_router(transcribe_router)
