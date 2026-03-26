"""
VisionSort — Evaluation
========================
Test-set evaluation with:
- Per-class Precision, Recall, F1 scores
- Confusion matrix plot (saved as PNG)
- Overall accuracy and macro-avg F1
- JSON report export
- Weakest-category identification
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path = Config.MODEL_DIR,
) -> dict:
    """
    Run full evaluation on a dataset (typically the test set).

    Args:
        model: trained model in eval mode
        dataloader: test DataLoader
        device: compute device
        save_dir: directory to save confusion matrix PNG and JSON report

    Returns:
        dict with evaluation results
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_confidences = []

    use_amp = Config.USE_MIXED_PRECISION and device.type == "cuda"

    print(f"\n{'═' * 60}")
    print(f"  📊 Running Evaluation")
    print(f"{'═' * 60}\n")

    pbar = tqdm(dataloader, desc="  Evaluating", ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        confidences, predicted = probs.max(dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confidences.extend(confidences.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # ── Overall Metrics ──
    overall_acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    # ── Per-Class Report ──
    target_names = Config.CATEGORIES
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        zero_division=0,
    )

    # ── Print Results ──
    print(f"\n  Overall Accuracy : {overall_acc:.2f}%")
    print(f"  Macro F1 Score   : {macro_f1:.2f}%")
    print(f"  Macro Precision  : {macro_precision:.2f}%")
    print(f"  Macro Recall     : {macro_recall:.2f}%")
    print(f"\n  Per-Class Report:\n{report_text}")

    # ── Identify Weakest Categories ──
    per_class_f1 = []
    for cat in Config.CATEGORIES:
        if cat in report_dict:
            per_class_f1.append((cat, report_dict[cat]["f1-score"] * 100))
        else:
            per_class_f1.append((cat, 0.0))

    per_class_f1.sort(key=lambda x: x[1])
    print("  ⚠️  Weakest categories (consider adding more training data):")
    for cat, f1 in per_class_f1[:3]:
        print(f"      {cat}: F1 = {f1:.1f}%")

    # ── Confidence Analysis ──
    avg_confidence = np.mean(all_confidences) * 100
    uncertain_count = np.sum(all_confidences < Config.CONFIDENCE_THRESHOLD)
    uncertain_pct = uncertain_count / len(all_confidences) * 100
    print(f"\n  Average confidence: {avg_confidence:.1f}%")
    print(f"  Uncertain predictions (<{Config.CONFIDENCE_THRESHOLD * 100:.0f}%): {uncertain_count} ({uncertain_pct:.1f}%)")

    # ── Confusion Matrix Plot ──
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, target_names, save_dir / "confusion_matrix.png")

    # ── Save JSON Report ──
    results = {
        "overall_accuracy": round(overall_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "macro_precision": round(macro_precision, 2),
        "macro_recall": round(macro_recall, 2),
        "average_confidence": round(avg_confidence, 2),
        "uncertain_predictions_count": int(uncertain_count),
        "uncertain_predictions_pct": round(uncertain_pct, 2),
        "per_class": report_dict,
        "weakest_categories": [{"category": c, "f1": round(f, 2)} for c, f in per_class_f1[:3]],
        "confusion_matrix": cm.tolist(),
    }

    report_path = save_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  📄 Report saved  : {report_path}")
    print(f"  📊 Confusion matrix: {save_dir / 'confusion_matrix.png'}")
    print(f"{'═' * 60}\n")

    return results


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    """Generate and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("True", fontsize=13, fontweight="bold")
    ax.set_title("VisionSort — Confusion Matrix", fontsize=15, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
