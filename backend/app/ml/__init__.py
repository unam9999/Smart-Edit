# VisionSort ML Package
"""
Complete ML pipeline for image classification:
- config: Centralized hyperparameters and paths
- dataset: Data loading with augmentation and train/val/test splits
- model: EfficientNet-B0 with custom classification head
- trainer: Full training loop with mixed precision, early stopping, TensorBoard
- evaluate: Test-set evaluation with confusion matrix and per-class metrics
- inference: Production inference engine with batch prediction support
- utils: Common utilities (seeding, device detection, checkpoints)
"""

from .config import Config
