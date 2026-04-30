"""
VisionSort — Configuration
===========================
All constants, thresholds, and category definitions.
"""

from pathlib import Path


class Config:
    # ── Categories ──────────────────────────────────────────────
    CATEGORIES = ["blurred", "people", "animals", "aesthetic", "uncategorized", "unlabelled"]

    # ── Detection Thresholds ────────────────────────────────────
    BLUR_THRESHOLD = 100.0        # Laplacian variance below this → blurry
    CONFIDENCE_THRESHOLD = 0.10   # Model confidence below this → unlabelled
    ENTROPY_THRESHOLD = 4.5       # Image entropy below this (combined with low confidence) → unlabelled
    FACE_SCALE_FACTOR = 1.1       # Haar cascade scale factor
    FACE_MIN_NEIGHBORS = 5        # Haar cascade min neighbors (higher = fewer false positives)
    FACE_MIN_SIZE = (30, 30)      # Minimum face size in pixels

    # ── Strict Aesthetic Threshold ──────────────────────────────
    # Model must be at least this confident AND the class must be in
    # AESTHETIC_IMAGENET_CLASSES for an image to qualify as "aesthetic".
    # Images that pass confidence but miss the whitelist → "uncategorized".
    AESTHETIC_CONFIDENCE_THRESHOLD = 0.40

    # ── Aesthetic ImageNet Class Whitelist ──────────────────────
    # Curated set of high-impact ImageNet-1K class indices.
    # Only images strongly recognized as these subjects earn the "aesthetic" label.
    AESTHETIC_IMAGENET_CLASSES: frozenset = frozenset({
        # ── Supercars / Sports Cars ──
        511,   # convertible
        573,   # go-kart
        628,   # limousine
        670,   # motor scooter
        671,   # mountain bike
        751,   # racer / race car
        814,   # speedboat
        817,   # sports car

        # ── Iconic Architecture / Structures ──
        483,   # castle
        484,   # catamaran (aesthetic watercraft)
        698,   # palace
        821,   # steel arch bridge
        839,   # suspension bridge
        873,   # triumphal arch
        888,   # viaduct

        # ── Natural Scenery / Landscapes ──
        970,   # alp
        972,   # cliff
        973,   # coral reef
        974,   # geyser
        975,   # lakeside
        976,   # promontory
        978,   # seashore
        979,   # valley
        980,   # volcano
    })

    # ── Model Settings ──────────────────────────────────────────
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # ── ImageNet Animal Class Range ─────────────────────────────
    # In the standard ImageNet-1K class list, indices 0–397 are animals
    # (fish, amphibians, reptiles, birds, mammals, insects, arachnids)
    ANIMAL_CLASS_START = 0
    ANIMAL_CLASS_END = 397  # inclusive

    # ── Paths ───────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parents[3]  # VisionSort/
    BACKEND_DIR = PROJECT_ROOT / "backend"
