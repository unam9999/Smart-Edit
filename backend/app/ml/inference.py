"""
VisionSort — Inference Engine
==============================
Production-ready inference for the FastAPI service and CLI:
- Classifier class that loads the model once and keeps it in memory
- Single-image and batch prediction APIs
- Uncertainty flagging
- ONNX export utility
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .config import Config
from .model import build_model
from .utils import get_device, load_checkpoint


class Classifier:
    """
    Production inference wrapper.

    Usage:
        clf = Classifier()  # loads best.pt automatically
        result = clf.predict_image(pil_image)
        # result = {"category": "portrait", "confidence": 0.94, "is_uncertain": False}
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            checkpoint_path: path to the .pt checkpoint file.
                             Defaults to Config.MODEL_DIR / "best.pt"
            device: compute device. Auto-detected if not specified.
        """
        self.device = device or get_device()
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else Config.MODEL_DIR / "best.pt"

        # Load checkpoint and extract metadata
        print(f"\n  📦 Loading model from: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        backbone = checkpoint.get("backbone", Config.BACKBONE)
        num_classes = checkpoint.get("num_classes", Config.NUM_CLASSES)
        categories = checkpoint.get("categories", Config.CATEGORIES)

        # Rebuild model architecture
        self.model = build_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,  # We'll load our own weights
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.categories = categories
        self.num_classes = num_classes

        # Inference transforms (same as eval transforms)
        self.transform = transforms.Compose([
            transforms.Resize(int(Config.IMAGE_SIZE * 1.15)),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])

        val_acc = checkpoint.get("val_acc", "N/A")
        epoch = checkpoint.get("epoch", "N/A")
        print(f"  ✅ Model loaded (backbone={backbone}, epoch={epoch}, val_acc={val_acc}%)")

    @torch.no_grad()
    def predict_image(
        self,
        image: Image.Image,
        threshold: float = Config.CONFIDENCE_THRESHOLD,
    ) -> dict:
        """
        Classify a single PIL Image.

        Args:
            image: PIL Image (any size, any mode — will be converted to RGB)
            threshold: confidence below this → is_uncertain = True

        Returns:
            {
                "category": str,
                "category_idx": int,
                "confidence": float (0.0–1.0),
                "is_uncertain": bool,
                "top3": [(category, confidence), ...]
            }
        """
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        output = self.model(tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)

        # Top-3 predictions
        top3_probs, top3_idxs = probs.topk(3)
        top3 = [
            (self.categories[idx.item()], round(prob.item(), 4))
            for prob, idx in zip(top3_probs, top3_idxs)
        ]

        confidence = top3[0][1]
        category = top3[0][0]
        category_idx = top3_idxs[0].item()

        return {
            "category": category,
            "category_idx": category_idx,
            "confidence": confidence,
            "is_uncertain": confidence < threshold,
            "top3": top3,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[Image.Image],
        threshold: float = Config.CONFIDENCE_THRESHOLD,
    ) -> list[dict]:
        """
        Classify a batch of PIL Images.

        Args:
            images: list of PIL Images
            threshold: confidence threshold for uncertainty flagging

        Returns:
            List of prediction dicts (same format as predict_image)
        """
        if not images:
            return []

        # Prepare batch tensor
        tensors = []
        for img in images:
            img = img.convert("RGB")
            tensors.append(self.transform(img))

        batch = torch.stack(tensors).to(self.device)
        outputs = self.model(batch)
        probs = torch.softmax(outputs, dim=1)

        results = []
        for i in range(len(images)):
            img_probs = probs[i]
            top3_probs, top3_idxs = img_probs.topk(3)
            top3 = [
                (self.categories[idx.item()], round(prob.item(), 4))
                for prob, idx in zip(top3_probs, top3_idxs)
            ]

            confidence = top3[0][1]
            results.append({
                "category": top3[0][0],
                "category_idx": top3_idxs[0].item(),
                "confidence": confidence,
                "is_uncertain": confidence < threshold,
                "top3": top3,
            })

        return results


def export_onnx(
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Export the trained model to ONNX format for fast CPU inference.

    Args:
        checkpoint_path: path to .pt checkpoint (default: best.pt)
        output_path: ONNX output path (default: model/visionsort.onnx)

    Returns:
        Path to the exported ONNX file
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else Config.MODEL_DIR / "best.pt"
    output_path = Path(output_path) if output_path else Config.MODEL_DIR / "visionsort.onnx"

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = checkpoint.get("backbone", Config.BACKBONE)
    num_classes = checkpoint.get("num_classes", Config.NUM_CLASSES)

    model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=13,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"\n  📦 ONNX model exported to: {output_path}")
    print(f"     Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path
