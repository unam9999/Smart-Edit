"""
VisionSort — Utilities
======================
Common helpers: seeding, device detection, checkpoints, running averages.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  🍎 Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("  💻 Using CPU")
    return device


def save_checkpoint(
    state: dict,
    save_dir: Path,
    is_best: bool = False,
) -> None:
    """
    Save a training checkpoint.

    Args:
        state: dict with keys like 'epoch', 'model_state_dict', 'optimizer_state_dict', etc.
        save_dir: directory to save into
        is_best: if True, also copies the file as 'best.pt'
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    last_path = save_dir / "last.pt"
    torch.save(state, last_path)

    if is_best:
        best_path = save_dir / "best.pt"
        shutil.copy2(last_path, best_path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint into a model.

    Args:
        model: the PyTorch model
        checkpoint_path: path to the .pt file
        device: device to map the checkpoint to
        strict: whether to enforce that the keys match exactly

    Returns:
        The full checkpoint dict (contains epoch, optimizer state, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device if device else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    return checkpoint


class AverageMeter:
    """
    Keeps a running average of a value.
    Usage:
        meter = AverageMeter()
        meter.update(loss_value, batch_size)
        print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
