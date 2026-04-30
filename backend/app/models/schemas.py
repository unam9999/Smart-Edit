"""
VisionSort -- API Schemas
=========================
Pydantic models for request/response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Result for a single image classification."""
    filename: str = Field(..., description="Original filename")
    category: str = Field(..., description="Predicted category: blurred | people | animals | aesthetic | uncategorized | unlabelled")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0–1.0)")
    label: str = Field(..., description="Human-readable label from the model")
    blur_score: float = Field(..., description="Laplacian variance (higher = sharper)")
    faces_detected: int = Field(..., ge=0, description="Number of faces detected")
    entropy: float = Field(..., description="Image entropy (higher = more information)")
    top3: list[dict] = Field(..., description="Top 3 ImageNet predictions")


class BatchClassificationResponse(BaseModel):
    """Response for batch classification (JSON only, no session)."""
    total: int = Field(..., description="Total number of images processed")
    results: list[ClassificationResult]
    summary: dict = Field(..., description="Count per category")


class BatchClassificationSessionResponse(BaseModel):
    """Response for batch classification with session support (used by /classify/batch)."""
    session_id: str = Field(..., description="Session ID — pass this to /api/process")
    total: int = Field(..., description="Total number of images processed")
    results: list[ClassificationResult]
    summary: dict = Field(..., description="Count per category")


class FolderPreset(BaseModel):
    """A preset choice for a single folder/category."""
    folder: str = Field(..., description="Category folder name (e.g. 'people', 'aesthetic')")
    preset: str = Field(..., description="Preset name: none | portraits | landscapes | aesthetic")


class ApplyPresetsRequest(BaseModel):
    """Request body for POST /api/process."""
    session_id: str = Field(..., description="Session ID returned by /api/classify/batch")
    presets: list[FolderPreset] = Field(..., description="Preset to apply per folder")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    model: str = "EfficientNet-B0 (ImageNet pretrained)"
    device: str = Field(..., description="Compute device (cpu/cuda/mps)")
    categories: list[str] = Field(..., description="Supported categories")
    version: str = "1.0.0"

