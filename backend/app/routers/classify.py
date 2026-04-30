"""
VisionSort -- Classification Router
====================================
Endpoints for image classification and filter processing.

POST /api/classify          — classify a single image
POST /api/classify/batch    — classify many images, save session, return JSON
POST /api/classify/batch/download — classify + immediately return sorted ZIP (no session)
POST /api/process           — apply per-folder presets to a session, return processed ZIP
"""

from __future__ import annotations

import io
import logging
import zipfile
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from ..ml import session_store
from ..ml.filters import PRESET_NAMES, apply_preset
from ..models.schemas import (
    ApplyPresetsRequest,
    BatchClassificationResponse,
    BatchClassificationSessionResponse,
    ClassificationResult,
)

logger = logging.getLogger("visionsort")
router = APIRouter(prefix="/api", tags=["Classification"])

# The pipeline instance is injected from main.py at startup
pipeline = None


def set_pipeline(p):
    """Called by main.py to inject the shared pipeline instance."""
    global pipeline
    pipeline = p


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_file_bytes(raw: bytes, filename: str, content_type: str | None) -> ClassificationResult:
    """Classify raw bytes and return a ClassificationResult."""
    if not content_type or not content_type.startswith("image/"):
        return ClassificationResult(
            filename=filename,
            category="unlabelled",
            confidence=0.0,
            label="invalid file type",
            blur_score=0.0,
            faces_detected=0,
            entropy=0.0,
            top3=[],
        )
    try:
        image = Image.open(BytesIO(raw))
        result = pipeline.classify(image)
        result["filename"] = filename
        return ClassificationResult(**result)
    except Exception as e:
        logger.error(f"Error classifying {filename}: {e}")
        return ClassificationResult(
            filename=filename,
            category="unlabelled",
            confidence=0.0,
            label=f"error: {e}",
            blur_score=0.0,
            faces_detected=0,
            entropy=0.0,
            top3=[],
        )


def _build_summary(results: list[ClassificationResult]) -> dict:
    summary: dict[str, int] = {}
    for r in results:
        summary[r.category] = summary.get(r.category, 0) + 1
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Session status check
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/session/{session_id}/status",
    summary="Check if a session is still alive",
    description="Returns whether the session exists and how many images it holds.",
)
async def session_status(session_id: str):
    """Check if a classification session is still alive."""
    session = session_store.get_session(session_id)
    if session is None:
        return {"alive": False, "image_count": 0}
    return {"alive": True, "image_count": len(session.images)}


# ─────────────────────────────────────────────────────────────────────────────
#  Single image
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/classify",
    response_model=ClassificationResult,
    summary="Classify a single image",
)
async def classify_image(file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)")):
    """Classify a single uploaded image."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    result = pipeline.classify(image)
    result["filename"] = file.filename or "unknown"
    return ClassificationResult(**result)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch classify — saves session so presets can be applied later
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/classify/batch",
    response_model=BatchClassificationSessionResponse,
    summary="Classify multiple images (returns session_id for preset pipeline)",
    description=(
        "Upload multiple images for batch classification. "
        "Returns a session_id that you must pass to POST /api/process "
        "to apply preset filters and download the sorted ZIP."
    ),
)
async def classify_batch(files: list[UploadFile] = File(..., description="Multiple image files")):
    """Classify a batch of uploaded images and store raw bytes in session."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")

    session_id = session_store.create_session()
    results: list[ClassificationResult] = []

    for file in files:
        raw = await file.read()
        filename = file.filename or "image"
        result = _classify_file_bytes(raw, filename, file.content_type)
        results.append(result)
        session_store.add_image(session_id, filename, raw, result.category)

    return BatchClassificationSessionResponse(
        session_id=session_id,
        total=len(results),
        results=results,
        summary=_build_summary(results),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Batch classify → direct ZIP download (no session, legacy endpoint)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/classify/batch/download",
    summary="Classify & immediately download sorted ZIP (no filter presets)",
    response_class=StreamingResponse,
)
async def classify_batch_download(
    files: list[UploadFile] = File(..., description="Multiple image files (max 100)"),
):
    """Classify images and return a ZIP with one sub-folder per category."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")

    zip_buffer = io.BytesIO()
    manifest_lines = ["filename,category,confidence,label,blur_score,faces_detected,entropy"]

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            raw = await file.read()
            filename = file.filename or "image"
            result = _classify_file_bytes(raw, filename, file.content_type)
            zf.writestr(f"{result.category}/{filename}", raw)
            manifest_lines.append(
                f"{filename},{result.category},{result.confidence},{result.label},"
                f"{result.blur_score},{result.faces_detected},{result.entropy}"
            )
        zf.writestr("manifest.csv", "\n".join(manifest_lines))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=VisionSort_output.zip"},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Apply presets → processed ZIP download
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/process",
    summary="Apply folder presets and download processed ZIP",
    description=(
        "Pass the session_id from /api/classify/batch plus a list of "
        "{folder, preset} pairs. Each image in each folder will have its "
        "chosen preset filter applied. Returns a ZIP with processed images "
        "sorted into category sub-folders."
    ),
    response_class=StreamingResponse,
)
async def apply_presets_and_download(body: ApplyPresetsRequest):
    """Apply per-folder preset filters to a session and return a ZIP."""
    # Validate preset names
    valid_presets = set(PRESET_NAMES)
    for fp in body.presets:
        if fp.preset.lower() not in valid_presets:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset {fp.preset!r}. Valid options: {sorted(valid_presets)}",
            )

    # Retrieve session
    session = session_store.get_session(body.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found or expired — no active process for this session ID. "
                "Please re-upload and re-classify your images to start a new session."
            ),
        )

    # Build preset lookup: folder → preset name
    folder_preset: dict[str, str] = {fp.folder.lower(): fp.preset.lower() for fp in body.presets}

    zip_buffer = io.BytesIO()
    manifest_lines = ["filename,category,preset_applied"]

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in session.images:
            preset_name = folder_preset.get(entry.category, "none")

            try:
                if preset_name == "none":
                    processed_bytes = entry.raw_bytes
                else:
                    pil_img = Image.open(BytesIO(entry.raw_bytes)).convert("RGB")
                    processed_img = apply_preset(pil_img, preset_name)

                    # Re-encode to JPEG (or PNG if original was PNG)
                    ext = entry.filename.rsplit(".", 1)[-1].lower()
                    fmt = "PNG" if ext == "png" else "JPEG"
                    buf = io.BytesIO()
                    save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
                    processed_img.save(buf, format=fmt, **save_kwargs)
                    processed_bytes = buf.getvalue()

            except Exception as e:
                logger.error(f"Error applying preset '{preset_name}' to {entry.filename}: {e}")
                processed_bytes = entry.raw_bytes   # fall back to original

            zf.writestr(f"{entry.category}/{entry.filename}", processed_bytes)
            manifest_lines.append(f"{entry.filename},{entry.category},{preset_name}")

        zf.writestr("manifest.csv", "\n".join(manifest_lines))

    # Session consumed — delete to free memory
    session_store.delete_session(body.session_id)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=VisionSort_processed.zip"},
    )
