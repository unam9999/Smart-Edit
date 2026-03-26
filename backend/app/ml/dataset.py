"""
VisionSort — Dataset & Data Loading
====================================
- SmartSortDataset: custom PyTorch Dataset that reads images from category folders
- Train / Val / Test split with stratification
- Separate augmentation pipelines for training vs evaluation
- WeightedRandomSampler for class-imbalance handling
"""

import os
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import Config


# ═══════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════

def get_train_transforms() -> transforms.Compose:
    """
    Heavy augmentation for training:
    - Random resized crop (scale 0.8–1.0) to simulate zoom variation
    - Random horizontal flip
    - Color jitter (brightness, contrast, saturation, hue)
    - Random rotation ±15°
    - Normalize with ImageNet stats
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),  # Random cutout
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Clean transforms for validation/test — no randomness.
    Resize slightly larger then center-crop to IMAGE_SIZE.
    """
    return transforms.Compose([
        transforms.Resize(int(Config.IMAGE_SIZE * 1.15)),  # e.g. 256
        transforms.CenterCrop(Config.IMAGE_SIZE),           # 224
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class SmartSortDataset(Dataset):
    """
    Reads images from `root_dir/<category>/*.{jpg,png,...}`
    and assigns integer labels based on Config.CATEGORY_TO_IDX.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            samples: list of (image_path, label_index) tuples
            transform: torchvision transforms to apply
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Open and convert to RGB (handles grayscale, RGBA, etc.)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠️  Skipping corrupt image: {img_path} — {e}")
            # Return a black image of the correct size as fallback
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


# ═══════════════════════════════════════════════════════════════════════
# Data Discovery & Splitting
# ═══════════════════════════════════════════════════════════════════════

def discover_samples(data_dir: Union[str, Path]) -> list[tuple[str, int]]:
    """
    Walk through category folders and collect (path, label) pairs.

    Returns:
        List of (image_path, label_index) tuples
    """
    data_dir = Path(data_dir)
    samples = []

    for cat in Config.CATEGORIES:
        cat_dir = data_dir / cat
        if not cat_dir.exists():
            print(f"  ⚠️  Category folder missing: {cat_dir}")
            continue

        for file_path in cat_dir.iterdir():
            if file_path.suffix.lower() in Config.IMAGE_EXTENSIONS:
                samples.append((str(file_path), Config.CATEGORY_TO_IDX[cat]))

    return samples


def split_dataset(
    samples: list[tuple[str, int]],
    train_ratio: float = Config.TRAIN_RATIO,
    val_ratio: float = Config.VAL_RATIO,
    seed: int = Config.SPLIT_SEED,
) -> tuple[list, list, list]:
    """
    Stratified split into train / val / test sets.

    Args:
        samples: full list of (path, label) tuples
        train_ratio, val_ratio: proportions (test = 1 - train - val)
        seed: random seed for reproducibility

    Returns:
        (train_samples, val_samples, test_samples)
    """
    if len(samples) == 0:
        return [], [], []

    paths, labels = zip(*samples)
    paths = list(paths)
    labels = list(labels)

    # First split: train vs (val + test)
    val_test_ratio = 1.0 - train_ratio
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test (from the remaining portion)
    test_fraction_of_remainder = (1.0 - train_ratio - val_ratio) / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths, valtest_labels,
        test_size=test_fraction_of_remainder,
        stratify=valtest_labels,
        random_state=seed,
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))

    return train_samples, val_samples, test_samples


# ═══════════════════════════════════════════════════════════════════════
# Weighted Sampler for Class Imbalance
# ═══════════════════════════════════════════════════════════════════════

def make_weighted_sampler(samples: list[tuple[str, int]]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler so that under-represented categories
    are sampled more frequently during training.
    """
    labels = [label for _, label in samples]
    class_counts = Counter(labels)
    num_samples = len(labels)

    # Weight per class = total / count_for_that_class
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}

    # Weight per sample
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# DataLoader Factory
# ═══════════════════════════════════════════════════════════════════════

def get_dataloaders(
    data_dir: Union[str, Path] = Config.DATA_DIR,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = 0,  # Set >0 on Linux; Windows often needs 0
) -> dict[str, DataLoader]:
    """
    Main entry point: discovers images, splits them, creates DataLoaders.

    Returns:
        Dict with keys 'train', 'val', 'test', each holding a DataLoader.
        Also prints dataset statistics.
    """
    print("\n📂 Discovering images...")
    all_samples = discover_samples(data_dir)

    if len(all_samples) == 0:
        print("  ❌ No images found! Add images to data/<category>/ folders.")
        print(f"  Expected folders: {', '.join(Config.CATEGORIES)}")
        return {"train": None, "val": None, "test": None}

    print(f"  Found {len(all_samples)} images across {Config.NUM_CLASSES} categories\n")

    # Print class distribution
    print_class_distribution(all_samples)

    # Split
    train_samples, val_samples, test_samples = split_dataset(all_samples)
    print(f"\n📊 Split: Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}")

    # Datasets with appropriate transforms
    train_dataset = SmartSortDataset(train_samples, transform=get_train_transforms())
    val_dataset = SmartSortDataset(val_samples, transform=get_eval_transforms())
    test_dataset = SmartSortDataset(test_samples, transform=get_eval_transforms())

    # Weighted sampler for training
    sampler = make_weighted_sampler(train_samples)

    # DataLoaders
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,      # Weighted sampling replaces shuffle
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,       # Drop incomplete last batch for stable batch norm
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders


# ═══════════════════════════════════════════════════════════════════════
# Statistics / Helpers
# ═══════════════════════════════════════════════════════════════════════

def print_class_distribution(samples: list[tuple[str, int]]) -> None:
    """Print a bar-chart-style class distribution."""
    labels = [label for _, label in samples]
    counts = Counter(labels)
    max_count = max(counts.values()) if counts else 1

    print("  Class Distribution:")
    print("  " + "─" * 50)
    for idx in range(Config.NUM_CLASSES):
        cat = Config.IDX_TO_CATEGORY[idx]
        count = counts.get(idx, 0)
        bar_len = int(30 * count / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {cat:>15s} │ {bar} {count}")
    print("  " + "─" * 50)
