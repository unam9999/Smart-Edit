"""
VisionSort — Trainer
====================
Production training loop with:
- Two-phase transfer learning (freeze backbone → unfreeze for fine-tuning)
- Mixed precision training (AMP)
- CosineAnnealing LR scheduler
- Early stopping on validation loss
- Gradient clipping
- TensorBoard logging
- Checkpoint saving (best + last)
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Config
from .model import unfreeze_backbone
from .utils import AverageMeter, save_checkpoint


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = Config.EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n  ⏹️  Early stopping triggered (no improvement for {self.patience} epochs)")
                return True  # Stop
            return False


class Trainer:
    """
    Handles the full training lifecycle.

    Usage:
        trainer = Trainer(model, dataloaders, device)
        trainer.run()
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: dict[str, DataLoader],
        device: torch.device,
        backbone: str = Config.BACKBONE,
        epochs: int = Config.EPOCHS,
        lr: float = Config.LEARNING_RATE,
        finetune_lr: float = Config.FINETUNE_LR,
        weight_decay: float = Config.WEIGHT_DECAY,
        freeze_epochs: int = Config.FREEZE_EPOCHS,
        label_smoothing: float = Config.LABEL_SMOOTHING,
        save_dir: Path = Config.MODEL_DIR,
        log_dir: Path = Config.LOG_DIR,
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.backbone = backbone
        self.epochs = epochs
        self.lr = lr
        self.finetune_lr = finetune_lr
        self.weight_decay = weight_decay
        self.freeze_epochs = freeze_epochs
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer — initially trains only unfrozen (head) parameters
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # LR Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        # Mixed precision
        self.use_amp = Config.USE_MIXED_PRECISION and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping()

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Best accuracy tracking
        self.best_val_acc = 0.0

    def run(self) -> dict:
        """
        Execute the full training pipeline.

        Returns:
            dict with training results summary
        """
        print(f"\n{'═' * 60}")
        print(f"  🚂 Starting Training")
        print(f"  Backbone       : {self.backbone}")
        print(f"  Device         : {self.device}")
        print(f"  Epochs         : {self.epochs}")
        print(f"  Freeze epochs  : {self.freeze_epochs}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Save directory : {self.save_dir}")
        print(f"  TensorBoard    : {self.log_dir}")
        print(f"{'═' * 60}\n")

        Config.ensure_dirs()
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # ── Phase transition: unfreeze backbone after freeze_epochs ──
            if epoch == self.freeze_epochs + 1:
                print(f"\n{'─' * 60}")
                print(f"  🔓 Phase 2: Unfreezing backbone for fine-tuning")
                print(f"{'─' * 60}")
                unfreeze_backbone(self.model, self.backbone)

                # Reset optimizer with lower LR for fine-tuning
                self.optimizer = Adam(
                    self.model.parameters(),
                    lr=self.finetune_lr,
                    weight_decay=self.weight_decay,
                )
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs - self.freeze_epochs,
                    eta_min=1e-6,
                )

            # ── Train one epoch ──
            train_loss, train_acc = self._train_epoch(epoch)

            # ── Validate ──
            val_loss, val_acc = self._validate_epoch(epoch)

            # ── Scheduler step ──
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # ── TensorBoard logging ──
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            # ── Print epoch summary ──
            phase = "HEAD" if epoch <= self.freeze_epochs else "FULL"
            print(
                f"  Epoch {epoch:3d}/{self.epochs} [{phase}] │ "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% │ "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% │ "
                f"LR: {current_lr:.2e}"
            )

            # ── Checkpoint ──
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"  ✅ New best val accuracy: {val_acc:.2f}%")

            save_checkpoint(
                state={
                    "epoch": epoch,
                    "backbone": self.backbone,
                    "num_classes": Config.NUM_CLASSES,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "categories": Config.CATEGORIES,
                },
                save_dir=self.save_dir,
                is_best=is_best,
            )

            # ── Early stopping check ──
            if self.early_stopping(val_loss):
                break

        # ── Training complete ──
        elapsed = time.time() - start_time
        self.writer.close()

        summary = {
            "total_epochs": epoch,
            "best_val_acc": self.best_val_acc,
            "elapsed_seconds": elapsed,
        }

        print(f"\n{'═' * 60}")
        print(f"  🏁 Training Complete")
        print(f"  Total epochs   : {epoch}")
        print(f"  Best val acc   : {self.best_val_acc:.2f}%")
        print(f"  Time elapsed   : {elapsed / 60:.1f} minutes")
        print(f"  Best model     : {self.save_dir / 'best.pt'}")
        print(f"{'═' * 60}\n")

        return summary

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy%)."""
        self.model.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        loader = self.dataloaders["train"]
        pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False, ncols=100)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=Config.GRADIENT_CLIP_MAX_NORM,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100.0 * correct / total:.1f}%")

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return loss_meter.avg, accuracy

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> tuple[float, float]:
        """Run one validation epoch. Returns (avg_loss, accuracy%)."""
        self.model.eval()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        loader = self.dataloaders["val"]
        pbar = tqdm(loader, desc=f"  Val   Epoch {epoch}", leave=False, ncols=100)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return loss_meter.avg, accuracy
