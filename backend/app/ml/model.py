"""
VisionSort — Model Definition
==============================
Factory function to build transfer-learning models with a custom classification head.
Supports: EfficientNet-B0, EfficientNet-B2, ResNet-50.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from .config import Config


# ═══════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════

def build_model(
    backbone: str = Config.BACKBONE,
    num_classes: int = Config.NUM_CLASSES,
    dropout: float = Config.DROPOUT,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a classification model using transfer learning.

    The pretrained backbone is loaded, its classifier head is replaced with:
        Dropout → Linear(in_features, num_classes)

    By default, the backbone is frozen (only the new head trains).
    Call `unfreeze_backbone()` later for fine-tuning.

    Args:
        backbone: one of 'efficientnet_b0', 'efficientnet_b2', 'resnet50'
        num_classes: number of output categories
        dropout: dropout probability in the classification head
        pretrained: whether to use ImageNet pretrained weights

    Returns:
        The modified model
    """
    backbone = backbone.lower()

    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Unknown backbone: '{backbone}'. "
            f"Choose from: efficientnet_b0, efficientnet_b2, resnet50"
        )

    # Freeze backbone — only the new head will train initially
    freeze_backbone(model, backbone)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: {backbone}")
    print(f"   Total params     : {total_params:,}")
    print(f"   Trainable params : {trainable_params:,} (head only)")
    print(f"   Frozen params    : {total_params - trainable_params:,} (backbone)")

    return model


# ═══════════════════════════════════════════════════════════════════════
# Freeze / Unfreeze Utilities
# ═══════════════════════════════════════════════════════════════════════

def freeze_backbone(model: nn.Module, backbone: str) -> None:
    """Freeze all backbone parameters, leaving only the head trainable."""
    backbone = backbone.lower()

    if backbone.startswith("efficientnet"):
        # Freeze features (backbone) — keep classifier (head) trainable
        for param in model.features.parameters():
            param.requires_grad = False
    elif backbone == "resnet50":
        # Freeze everything except fc
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False


def unfreeze_backbone(
    model: nn.Module,
    backbone: str = Config.BACKBONE,
    unfreeze_from: Optional[int] = None,
) -> None:
    """
    Unfreeze backbone layers for fine-tuning.

    Args:
        model: the model to modify
        backbone: backbone name (for architecture-specific unfreezing)
        unfreeze_from: for EfficientNet, unfreeze blocks from this index onward.
                       None = unfreeze everything.
    """
    backbone = backbone.lower()

    if backbone.startswith("efficientnet"):
        if unfreeze_from is None:
            # Unfreeze ALL backbone layers
            for param in model.features.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only the last N blocks
            for idx, block in enumerate(model.features):
                if idx >= unfreeze_from:
                    for param in block.parameters():
                        param.requires_grad = True

    elif backbone == "resnet50":
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n🔓 Backbone unfrozen — Trainable: {trainable:,} / {total:,}")
