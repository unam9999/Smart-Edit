"""
VisionSort — Health Router
===========================
System health and status endpoint.
"""

from fastapi import APIRouter

from ..models.schemas import HealthResponse
from ..ml.config import Config

router = APIRouter(prefix="/api", tags=["System"])

# Injected from main.py
pipeline = None


def set_pipeline(p):
    """Called by main.py to inject the shared pipeline instance."""
    global pipeline
    pipeline = p


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns system status, model info, and available categories.",
)
async def health_check():
    """Check if the API is running and the model is loaded."""
    device = str(pipeline.device) if pipeline else "not initialized"

    return HealthResponse(
        status="ok" if pipeline else "initializing",
        device=device,
        categories=Config.CATEGORIES,
    )
