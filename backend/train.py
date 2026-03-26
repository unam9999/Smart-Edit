"""
VisionSort — Training Script
=============================
CLI entrypoint to train the image classification model.

Usage:
    python train.py                          # Train with defaults
    python train.py --epochs 30 --lr 0.001   # Override hyperparameters
    python train.py --backbone resnet50      # Use ResNet-50 instead
    python train.py --resume                 # Resume from last checkpoint
    python train.py --evaluate-only          # Skip training, just evaluate

TensorBoard:
    tensorboard --logdir backend/logs
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import app.ml
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ml.config import Config
from app.ml.dataset import get_dataloaders
from app.ml.evaluate import evaluate
from app.ml.model import build_model
from app.ml.trainer import Trainer
from app.ml.utils import get_device, load_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionSort — Train the image classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--backbone", type=str, default=Config.BACKBONE,
        choices=["efficientnet_b0", "efficientnet_b2", "resnet50"],
        help=f"Backbone architecture (default: {Config.BACKBONE})",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS,
                        help=f"Number of training epochs (default: {Config.EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE,
                        help=f"Batch size (default: {Config.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE,
                        help=f"Learning rate (default: {Config.LEARNING_RATE})")
    parser.add_argument("--freeze-epochs", type=int, default=Config.FREEZE_EPOCHS,
                        help=f"Epochs to train head only (default: {Config.FREEZE_EPOCHS})")

    # Data
    parser.add_argument("--data-dir", type=str, default=str(Config.DATA_DIR),
                        help=f"Path to data directory (default: {Config.DATA_DIR})")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0 for Windows)")

    # Checkpointing
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    # Evaluation only
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training, just run evaluation on test set")

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Print Config ──
    print(Config.summary())

    # ── Seed ──
    set_seed(args.seed)

    # ── Device ──
    device = get_device()

    # ── Data ──
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if dataloaders["train"] is None:
        print("\n❌ Cannot proceed without training data.")
        print("   Add images to data/<category>/ folders and try again.")
        sys.exit(1)

    # ── Model ──
    model = build_model(
        backbone=args.backbone,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
        pretrained=True,
    )

    # ── Resume from checkpoint ──
    start_epoch = 0
    if args.resume:
        last_ckpt = Config.MODEL_DIR / "last.pt"
        if last_ckpt.exists():
            print(f"\n  🔄 Resuming from: {last_ckpt}")
            checkpoint = load_checkpoint(model, last_ckpt, device)
            start_epoch = checkpoint.get("epoch", 0)
            print(f"     Resuming from epoch {start_epoch}")
        else:
            print(f"\n  ⚠️  No checkpoint found at {last_ckpt}, starting fresh.")

    # ── Evaluate Only ──
    if args.evaluate_only:
        best_ckpt = Config.MODEL_DIR / "best.pt"
        if not best_ckpt.exists():
            print(f"\n❌ No best model found at {best_ckpt}. Train first!")
            sys.exit(1)
        load_checkpoint(model, best_ckpt, device)
        evaluate(model, dataloaders["test"], device)
        return

    # ── Train ──
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        device=device,
        backbone=args.backbone,
        epochs=args.epochs,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
    )
    trainer.run()

    # ── Evaluate on test set after training ──
    print("\n🔍 Running final evaluation on test set...")
    best_ckpt = Config.MODEL_DIR / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(model, best_ckpt, device)
        evaluate(model, dataloaders["test"], device)
    else:
        print("  ⚠️  No best checkpoint found, evaluating current model state.")
        evaluate(model, dataloaders["test"], device)


if __name__ == "__main__":
    main()
