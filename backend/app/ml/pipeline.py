"""
VisionSort -- Classification Pipeline
======================================
The core engine that classifies images into 5 categories:
  1. Blurred    -- out-of-focus / motion-blurred images
  2. People     -- images containing human faces
  3. Animals    -- images of animals (mapped from ImageNet classes)
  4. Aesthetic  -- visually meaningful images (buildings, cars, scenery, food, etc.)
  5. Unlabelled -- images that don't make sense (cut-out faces, partial crops, noise)

Pipeline order:
  blur check -> face detection -> EfficientNet classification -> confidence gate -> category mapping
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .config import Config

logger = logging.getLogger("visionsort")


class VisionSortPipeline:
    """
    Production-ready image classification pipeline.

    Usage:
        pipeline = VisionSortPipeline()
        result = pipeline.classify(pil_image)
        # → {"category": "animals", "confidence": 0.91, "label": "golden retriever", "top3": [...]}
    """

    def __init__(self):
        logger.info("[INIT] Initializing VisionSort Pipeline...")

        # ── Device ──────────────────────────────────────────────
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"   Device: {self.device}")

        # ── EfficientNet-B0 (pretrained on ImageNet) ────────────
        logger.info("   Loading EfficientNet-B0...")
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.to(self.device)
        self.model.eval()

        # ── Image transforms (ImageNet normalization) ───────────
        self.transform = transforms.Compose([
            transforms.Resize(int(Config.IMAGE_SIZE * 1.15)),   # 257
            transforms.CenterCrop(Config.IMAGE_SIZE),            # 224
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])

        # ── OpenCV Face Detectors ───────────────────────────────
        logger.info("   Loading face detectors...")
        self.face_cascade_frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

        # ── ImageNet Labels ─────────────────────────────────────
        self.imagenet_labels = self._load_imagenet_labels()

        # ── Animal class index set ──────────────────────────────
        self.animal_indices = set(
            range(Config.ANIMAL_CLASS_START, Config.ANIMAL_CLASS_END + 1)
        )

        logger.info("   [OK] Pipeline ready!")

    # ─────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def classify(self, image: Image.Image) -> dict:
        """
        Classify a single PIL Image into one of 5 categories.

        Returns:
            {
                "category": str,          # blurred | people | animals | aesthetic | unlabelled
                "confidence": float,      # 0.0 – 1.0
                "label": str,             # human-readable label (ImageNet class name)
                "blur_score": float,      # Laplacian variance
                "faces_detected": int,    # number of faces found
                "entropy": float,         # image entropy
                "top3": [                 # top 3 ImageNet predictions
                    {"label": str, "confidence": float, "class_idx": int}, ...
                ]
            }
        """
        image = image.convert("RGB")
        img_array = np.array(image)

        # ── Step 1: Blur Detection ──────────────────────────────
        blur_score = self._compute_blur_score(img_array)
        is_blurry = blur_score < Config.BLUR_THRESHOLD

        # ── Step 2: Image Entropy ───────────────────────────────
        entropy = self._compute_entropy(img_array)

        # ── Step 3: Face Detection ──────────────────────────────
        faces_detected = self._detect_faces(img_array)

        # ── Step 4: EfficientNet Prediction ─────────────────────
        top3 = self._predict(image)
        top_confidence = top3[0]["confidence"]
        top_class_idx = top3[0]["class_idx"]
        top_label = top3[0]["label"]

        # ── Decision Logic ──────────────────────────────────────

        # 1) Blurry images
        if is_blurry:
            blur_confidence = max(0.0, min(1.0, 1.0 - (blur_score / Config.BLUR_THRESHOLD)))
            return self._result("blurred", blur_confidence,
                                "blurry image", blur_score, faces_detected, entropy, top3)

        # 2) People (faces detected)
        if faces_detected > 0:
            people_confidence = min(1.0, 0.7 + (faces_detected * 0.1))
            return self._result("people", people_confidence,
                                f"{faces_detected} face(s) detected", blur_score,
                                faces_detected, entropy, top3)

        # 3) Unlabelled — very low confidence + low entropy = nonsense / partial crop
        if top_confidence < Config.CONFIDENCE_THRESHOLD:
            return self._result("unlabelled", top_confidence,
                                top_label, blur_score, faces_detected, entropy, top3)

        if entropy < Config.ENTROPY_THRESHOLD and top_confidence < 0.25:
            return self._result("unlabelled", top_confidence,
                                top_label, blur_score, faces_detected, entropy, top3)

        # 4) Animals
        if top_class_idx in self.animal_indices:
            return self._result("animals", top_confidence,
                                top_label, blur_score, faces_detected, entropy, top3)

        # 5) Strict Aesthetic — must be a whitelisted class WITH high confidence.
        #    Whitelisted classes: supercars, superbikes, epic scenery, iconic architecture.
        if (top_class_idx in Config.AESTHETIC_IMAGENET_CLASSES
                and top_confidence >= Config.AESTHETIC_CONFIDENCE_THRESHOLD):
            return self._result("aesthetic", top_confidence,
                                top_label, blur_score, faces_detected, entropy, top3)

        # 6) Uncategorized — recognizable photo but lacks the "wow factor".
        return self._result("uncategorized", top_confidence,
                            top_label, blur_score, faces_detected, entropy, top3)

    def classify_batch(self, images: list[Image.Image]) -> list[dict]:
        """Classify multiple PIL Images. Returns a list of result dicts."""
        return [self.classify(img) for img in images]

    # ─────────────────────────────────────────────────────────────
    #  INTERNAL METHODS
    # ─────────────────────────────────────────────────────────────

    def _compute_blur_score(self, img_array: np.ndarray) -> float:
        """
        Compute Laplacian variance as a measure of image sharpness.
        Higher values = sharper image. Low values = blurry.
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _compute_entropy(self, img_array: np.ndarray) -> float:
        """
        Compute Shannon entropy of the grayscale histogram.
        Low entropy → uniform/featureless image → likely unlabelled.
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        # Remove zero bins to avoid log(0)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    def _detect_faces(self, img_array: np.ndarray) -> int:
        """
        Detect faces using OpenCV Haar cascades (frontal + profile).
        Returns the number of unique faces detected.
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Frontal faces
        frontal = self.face_cascade_frontal.detectMultiScale(
            gray,
            scaleFactor=Config.FACE_SCALE_FACTOR,
            minNeighbors=Config.FACE_MIN_NEIGHBORS,
            minSize=Config.FACE_MIN_SIZE,
        )

        # Profile faces (catches side views)
        profile = self.face_cascade_profile.detectMultiScale(
            gray,
            scaleFactor=Config.FACE_SCALE_FACTOR,
            minNeighbors=Config.FACE_MIN_NEIGHBORS,
            minSize=Config.FACE_MIN_SIZE,
        )

        frontal_count = len(frontal) if isinstance(frontal, np.ndarray) else 0
        profile_count = len(profile) if isinstance(profile, np.ndarray) else 0

        # Deduplicate overlapping detections
        return max(frontal_count, profile_count)

    @torch.no_grad()
    def _predict(self, image: Image.Image) -> list[dict]:
        """
        Run EfficientNet-B0 inference and return top 3 predictions.
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        probs = F.softmax(output, dim=1).squeeze(0)

        top3_probs, top3_idxs = probs.topk(3)

        results = []
        for prob, idx in zip(top3_probs, top3_idxs):
            class_idx = idx.item()
            results.append({
                "label": self.imagenet_labels.get(class_idx, f"class_{class_idx}"),
                "confidence": round(prob.item(), 4),
                "class_idx": class_idx,
            })

        return results

    def _load_imagenet_labels(self) -> dict[int, str]:
        """
        Load ImageNet class labels. Downloads from PyTorch Hub on first call,
        then caches locally.
        """
        import urllib.request

        cache_path = Config.BACKEND_DIR / "imagenet_classes.txt"

        if cache_path.exists():
            with open(cache_path, "r") as f:
                lines = f.readlines()
        else:
            logger.info("   Downloading ImageNet labels (one-time)...")
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    content = response.read().decode("utf-8")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    f.write(content)
                lines = content.strip().split("\n")
            except Exception as e:
                logger.warning(f"   [WARN] Could not download labels: {e}")
                return {i: f"class_{i}" for i in range(1000)}

        return {i: line.strip() for i, line in enumerate(lines)}

    @staticmethod
    def _result(
        category: str,
        confidence: float,
        label: str,
        blur_score: float,
        faces_detected: int,
        entropy: float,
        top3: list[dict],
    ) -> dict:
        """Build a standardized result dict."""
        return {
            "category": category,
            "confidence": round(float(confidence), 4),
            "label": label,
            "blur_score": round(float(blur_score), 2),
            "faces_detected": faces_detected,
            "entropy": round(float(entropy), 4),
            "top3": top3,
        }
