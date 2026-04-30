"""
VisionSort — Pipeline Test Script
===================================
Quick test to verify the classification pipeline works with local images.

Usage:
    python -m notebooks.test_model
    (run from the VisionSort root directory)
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from backend.app.ml.pipeline import VisionSortPipeline


def main():
    print("\n" + "=" * 60)
    print("  VisionSort — Pipeline Test")
    print("=" * 60)

    # Initialize pipeline
    pipeline = VisionSortPipeline()

    # Find test images
    test_dir = project_root / "test_images"
    if not test_dir.exists():
        print(f"\n  ❌ Test directory not found: {test_dir}")
        print("  Create a 'test_images/' folder with some images to test.")
        return

    image_files = [
        f for f in test_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]

    if not image_files:
        print(f"\n  ❌ No images found in: {test_dir}")
        return

    print(f"\n  Found {len(image_files)} test image(s)\n")
    print("-" * 60)

    for img_path in image_files:
        img = Image.open(img_path)
        result = pipeline.classify(img)

        print(f"\n  >> {img_path.name}")
        print(f"     Category:    {result['category'].upper()}")
        print(f"     Confidence:  {result['confidence'] * 100:.1f}%")
        print(f"     Label:       {result['label']}")
        print(f"     Blur Score:  {result['blur_score']:.1f}")
        print(f"     Faces:       {result['faces_detected']}")
        print(f"     Entropy:     {result['entropy']:.2f}")
        print(f"     Top 3:")
        for i, pred in enumerate(result["top3"], 1):
            print(f"       {i}. {pred['label']} ({pred['confidence'] * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("  [OK] Test complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()