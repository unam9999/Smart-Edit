"""
VisionSort — FastAPI Application
==================================
Entry point for the backend API service.

Run with:
    uvicorn backend.app.main:app --reload

Or from the project root:
    python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ml.pipeline import VisionSortPipeline
from .routers import classify, health

# ── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visionsort")

# ── Shared pipeline instance ────────────────────────────────────
pipeline_instance: Optional[VisionSortPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the ML pipeline on startup, clean up on shutdown."""
    global pipeline_instance

    logger.info("=" * 60)
    logger.info("  VisionSort API — Starting up")
    logger.info("=" * 60)

    # Initialize the classification pipeline (loads model into memory)
    pipeline_instance = VisionSortPipeline()

    # Inject pipeline into routers
    classify.set_pipeline(pipeline_instance)
    health.set_pipeline(pipeline_instance)

    logger.info("=" * 60)
    logger.info("  VisionSort API — Ready to serve requests")
    logger.info("  Docs: http://localhost:8000/docs")
    logger.info("=" * 60)

    yield  # App is running

    # Cleanup
    logger.info("VisionSort API — Shutting down")
    pipeline_instance = None


# ── FastAPI App ─────────────────────────────────────────────────
app = FastAPI(
    title="VisionSort API",
    description=(
        "AI-powered image classification API.\n\n"
        "Automatically sorts images into 5 categories:\n"
        "- **Blurred** — out-of-focus or motion-blurred images\n"
        "- **People** — images containing human faces\n"
        "- **Animals** — images of animals\n"
        "- **Aesthetic** — visually appealing images (buildings, cars, scenery, food, etc.)\n"
        "- **Unlabelled** — images that don't make sense (partial crops, cut-outs, noise)\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ────────────────────────────────────────────
app.include_router(classify.router)
app.include_router(health.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — redirects to API docs."""
    return {
        "message": "VisionSort API",
        "docs": "/docs",
        "health": "/api/health",
    }
