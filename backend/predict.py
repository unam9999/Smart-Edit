"""
VisionSort — Quick Prediction Script
======================================
Test the trained model on a single image or a folder of images.

Usage:
    python predict.py --image path/to/photo.jpg
    python predict.py --folder path/to/photos/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image

from app.ml.inference import Classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionSort — Predict image categories",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to a folder of images")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: best.pt)")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Confidence threshold for uncertainty (default: 0.70)")

    return parser.parse_args()


def predict_single(clf: Classifier, image_path: str, threshold: float):
    """Predict and print result for a single image."""
    path = Path(image_path)
    if not path.exists():
        print(f"  ❌ File not found: {path}")
        return

    img = Image.open(path)
    result = clf.predict_image(img, threshold=threshold)

    uncertain_flag = " ⚠️ UNCERTAIN" if result["is_uncertain"] else ""
    print(f"\n  📷 {path.name}")
    print(f"     Category   : {result['category']}{uncertain_flag}")
    print(f"     Confidence : {result['confidence'] * 100:.1f}%")
    print(f"     Top-3      :")
    for cat, conf in result["top3"]:
        print(f"       • {cat}: {conf * 100:.1f}%")


def predict_folder(clf: Classifier, folder_path: str, threshold: float):
    """Predict all images in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"  ❌ Not a directory: {folder}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    if not images:
        print(f"  ❌ No images found in {folder}")
        return

    print(f"\n  📁 Predicting {len(images)} images from: {folder}\n")
    print(f"  {'File':<30s} {'Category':<15s} {'Confidence':>10s}  {'Status'}")
    print(f"  {'─' * 70}")

    for img_path in sorted(images):
        img = Image.open(img_path)
        result = clf.predict_image(img, threshold=threshold)
        status = "⚠️ uncertain" if result["is_uncertain"] else "✅"
        print(
            f"  {img_path.name:<30s} {result['category']:<15s} "
            f"{result['confidence'] * 100:>9.1f}%  {status}"
        )


def main():
    args = parse_args()

    # Load model
    clf = Classifier(checkpoint_path=args.checkpoint)

    if args.image:
        predict_single(clf, args.image, args.threshold)
    elif args.folder:
        predict_folder(clf, args.folder, args.threshold)


if __name__ == "__main__":
    main()
