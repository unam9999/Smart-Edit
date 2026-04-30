"""
VisionSort — Preset Filter Functions
======================================
Each function takes a PIL Image and returns a processed PIL Image.

Presets:
  portraits    — face smoothing, brightness lift, subtle warmth
  landscapes   — contrast boost, saturation enhancement
  aesthetic    — high clarity (unsharp mask), local contrast boost (CLAHE)
  none         — return image unchanged
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ── Public registry ───────────────────────────────────────────
PRESET_NAMES = ["none", "portraits", "landscapes", "aesthetic"]


def apply_preset(image: Image.Image, preset: str) -> Image.Image:
    """
    Dispatch to the correct filter function by preset name.
    Falls back to 'none' for unknown preset names.
    """
    preset = preset.lower().strip()
    dispatch = {
        "none":        _preset_none,
        "portraits":   _preset_portraits,
        "landscapes":  _preset_landscapes,
        "aesthetic":   _preset_aesthetic,
    }
    fn = dispatch.get(preset, _preset_none)
    return fn(image.convert("RGB"))


# ── Preset Implementations ────────────────────────────────────

def _preset_none(image: Image.Image) -> Image.Image:
    """Return the image completely unchanged."""
    return image


def _preset_portraits(image: Image.Image) -> Image.Image:
    """
    Portraits / People preset:
      1. Gentle skin-smoothing via a soft Gaussian-style blur blended with the original
      2. +15% brightness
      3. Slight warmth — boost red channel, reduce blue channel
    """
    # 1) Soft smoothing blend (subtle — preserves detail)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    # Blend: 65% original + 35% blurred  → gentle softening, not watercolour
    smoothed = Image.blend(image, blurred, alpha=0.35)

    # 2) Brightness lift
    smoothed = ImageEnhance.Brightness(smoothed).enhance(1.15)

    # 3) Warmth — manipulate R/G/B channels
    r, g, b = smoothed.split()
    r = r.point(lambda px: min(255, int(px * 1.06)))   # boost red
    b = b.point(lambda px: max(0,   int(px * 0.93)))   # cool down blue
    warm = Image.merge("RGB", (r, g, b))

    return warm


def _preset_landscapes(image: Image.Image) -> Image.Image:
    """
    Landscapes preset:
      1. +25% contrast
      2. +30% saturation (vibrance-like pop)
      3. Subtle sharpening
    """
    img = ImageEnhance.Contrast(image).enhance(1.25)
    img = ImageEnhance.Color(img).enhance(1.30)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img


def _preset_aesthetic(image: Image.Image) -> Image.Image:
    """
    Aesthetic / Automotive preset:
      1. Strong clarity via unsharp mask (fine edge pop)
      2. Local contrast enhancement via OpenCV CLAHE (dynamic range boost)
      3. Very subtle saturation lift
    """
    # 1) Unsharp mask for crisp edges
    img = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))

    # 2) CLAHE for local contrast (works in LAB colour space to avoid hue shift)
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)

    lab_merged = cv2.merge([l_ch, a_ch, b_ch])
    enhanced_np = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
    img = Image.fromarray(enhanced_np)

    # 3) Slight colour pop
    img = ImageEnhance.Color(img).enhance(1.15)

    return img
