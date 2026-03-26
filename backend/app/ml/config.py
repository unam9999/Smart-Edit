"""
VisionSort — Centralized Configuration
=======================================
Single source of truth for all hyperparameters, paths, and constants.
"""

import os
from pathlib import Path


class Config:
    """All configuration in one place."""

    # ── Project Paths ─────────────────────────────────────────────────
    # Resolved relative to the repository root (VisionSort/)
    PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/app/ml → VisionSort
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "backend" / "model"
    LOG_DIR = PROJECT_ROOT / "backend" / "logs"

    # ── Categories ────────────────────────────────────────────────────
    # Must match the folder names inside data/
    CATEGORIES = [
        "animal",
        "architecture",
        "document",
        "event",
        "food",
        "group",
        "landscape",
        "nature",
        "night",
        "portrait",
        "product",
        "selfie",
        "vehicle",
        "SUS",
    ]
    NUM_CLASSES = len(CATEGORIES)
    CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
    IDX_TO_CATEGORY = {idx: cat for idx, cat in enumerate(CATEGORIES)}

    # ── Image Preprocessing ───────────────────────────────────────────
    IMAGE_SIZE = 224  # EfficientNet-B0 native input size
    # ImageNet normalization constants (required for pretrained backbones)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # ── Training Hyperparameters ──────────────────────────────────────
    BACKBONE = "efficientnet_b0"  # Options: efficientnet_b0, efficientnet_b2, resnet50
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3      # Initial LR for Adam (head-only phase)
    FINETUNE_LR = 1e-4        # Lower LR for full backbone fine-tuning phase
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1     # Reduces overconfidence; improves calibration
    DROPOUT = 0.3             # Dropout in the classification head

    # ── Training Strategy ─────────────────────────────────────────────
    FREEZE_EPOCHS = 5         # Train only the head for this many epochs first
    EARLY_STOPPING_PATIENCE = 10
    GRADIENT_CLIP_MAX_NORM = 1.0
    USE_MIXED_PRECISION = True  # Auto-disabled on CPU

    # ── Data Split ────────────────────────────────────────────────────
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    SPLIT_SEED = 42           # Fixed seed for reproducible splits

    # ── Inference ─────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD = 0.70  # Below this → flagged as "uncertain"

    # ── Supported Image Extensions ────────────────────────────────────
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

    @classmethod
    def ensure_dirs(cls):
        """Create output directories if they don't exist."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def summary(cls) -> str:
        """Print a human-readable config summary."""
        lines = [
            "═" * 50,
            "  VisionSort — Configuration",
            "═" * 50,
            f"  Data directory    : {cls.DATA_DIR}",
            f"  Categories        : {cls.NUM_CLASSES}",
            f"  Backbone          : {cls.BACKBONE}",
            f"  Image size        : {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}",
            f"  Batch size        : {cls.BATCH_SIZE}",
            f"  Epochs            : {cls.EPOCHS}",
            f"  Learning rate     : {cls.LEARNING_RATE}",
            f"  Fine-tune LR      : {cls.FINETUNE_LR}",
            f"  Label smoothing   : {cls.LABEL_SMOOTHING}",
            f"  Dropout           : {cls.DROPOUT}",
            f"  Freeze epochs     : {cls.FREEZE_EPOCHS}",
            f"  Early stop patience: {cls.EARLY_STOPPING_PATIENCE}",
            f"  Mixed precision   : {cls.USE_MIXED_PRECISION}",
            f"  Confidence thresh : {cls.CONFIDENCE_THRESHOLD}",
            "═" * 50,
        ]
        return "\n".join(lines)
