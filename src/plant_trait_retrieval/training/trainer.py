"""Trainer: encapsulates the training + validation loop for one fold."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from ..data.preprocessing import PowerTransformerWrapper
from .optim import build_optimizer, build_scheduler

console = Console()


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    bool_finite = torch.isfinite(y_true) & torch.isfinite(y_pred)
    y_true = y_true[bool_finite]
    y_pred = y_pred[bool_finite]

    if y_true.numel() == 0:
        return torch.tensor(float("nan"), device=y_pred.device)

    y_mean = torch.mean(y_true)
    total_var = torch.sum((y_true - y_mean) ** 2)
    if total_var == 0:
        return torch.tensor(0.0, device=y_pred.device)
    residual_var = torch.sum((y_true - y_pred) ** 2)
    return 1 - (residual_var / total_var)


def _finite_rmse_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float]:
    mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.numel() == 0:
        return float("nan"), float("nan")
    rmse = torch.sqrt(torch.mean((yp - yt) ** 2)).item()
    mae = torch.mean(torch.abs(yp - yt)).item()
    return float(rmse), float(mae)


class EarlyStopping:
    """Stop training when validation loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class CheckpointManager:
    """Keep only the top-k checkpoints ranked by val_loss."""

    def __init__(self, save_dir: Path, top_k: int = 3, fold: int = 0) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.fold = fold
        self._history: List[Tuple[float, Path]] = []  # (val_loss, path)

    def save(self, model: nn.Module, optimizer, epoch: int, val_loss: float) -> Path:
        path = self.save_dir / f"fold{self.fold}_epoch{epoch:04d}_loss{val_loss:.6f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
        self._history.append((val_loss, path))
        self._history.sort(key=lambda x: x[0])
        # Remove excess checkpoints
        while len(self._history) > self.top_k:
            _, old_path = self._history.pop()
            if old_path.exists():
                old_path.unlink()
        return path

    @property
    def best_path(self) -> Optional[Path]:
        return self._history[0][1] if self._history else None

    @property
    def best_loss(self) -> float:
        return self._history[0][0] if self._history else float("inf")


class Trainer:
    """Single-fold trainer.

    Parameters
    ----------
    model           : EfficientNet1d
    criterion       : loss function
    cfg             : Hydra config
    fold            : current fold index (for logging/checkpointing)
    checkpoint_dir  : where to save checkpoints
    device          : 'cuda' | 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        cfg: DictConfig,
        target_transformer: Optional[PowerTransformerWrapper] = None,
        fold: int = 0,
        checkpoint_dir: Path = Path("checkpoints"),
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.fold = fold
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.target_transformer = target_transformer

        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler, self.sched_on_metric = build_scheduler(self.optimizer, cfg)

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=cfg.training.mixed_precision and self.device.type == "cuda",
        )
        self.early_stopping = EarlyStopping(patience=cfg.training.patience)
        self.ckpt_manager = CheckpointManager(
            save_dir=checkpoint_dir / f"fold_{fold}",
            top_k=cfg.logging.save_top_k,
            fold=fold,
        )

        self.epochs = cfg.training.epochs
        self.grad_clip = cfg.training.gradient_clip
        self.log_interval = cfg.logging.log_interval
        self.use_wandb = cfg.logging.wandb.enabled
        self.use_sample_weights = bool(cfg.training.get("use_sample_weights", True))

    def _compute_loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_sample_weights and sample_weight is not None:
            try:
                return self.criterion(pred, y, sample_weight=sample_weight)
            except TypeError:
                return self.criterion(pred, y)
        return self.criterion(pred, y)

    def _transform_targets_for_loss(self, y: torch.Tensor) -> torch.Tensor:
        if self.target_transformer is None or not bool(self.cfg.data.get("power_transform", True)):
            return y
        return self.target_transformer.transform_tensor(y)

    # ------------------------------------------------------------------
    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns a dict with train/val histories.
        """
        history = {
            "train_loss": [],
            "train_r2": [],
            "val_loss": [],
            "val_r2": [],
            "train_skipped_batches": [],
            "val_skipped_batches": [],
        }

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Fold {self.fold}[/cyan] {{task.description}}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} epochs"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("training", total=self.epochs)

            for epoch in range(1, self.epochs + 1):
                train_loss, train_r2, train_skipped = self._train_epoch(train_loader, epoch)
                console.print(
                    f"[fold_{self.fold}] epoch {epoch}/{self.epochs} "
                    f"train_loss={train_loss:.4f} train_r2={train_r2:.4f}"
                )
                val_loss, val_metrics, val_skipped = self._val_epoch(val_loader)

                # Scheduler step
                if self.sched_on_metric:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                # Checkpoint
                self.ckpt_manager.save(self.model, self.optimizer, epoch, val_loss)

                history["train_loss"].append(train_loss)
                history["train_r2"].append(train_r2)
                history["val_loss"].append(val_loss)
                history["val_r2"].append(val_metrics["r2"])
                history["train_skipped_batches"].append(train_skipped)
                history["val_skipped_batches"].append(val_skipped)

                # W&B logging
                if self.use_wandb:
                    log_dict = {
                        f"fold{self.fold}/train_loss": train_loss,
                        f"fold{self.fold}/train_r2": train_r2,
                        f"fold{self.fold}/val_loss": val_loss,
                        f"fold{self.fold}/train_skipped_batches": train_skipped,
                        f"fold{self.fold}/val_skipped_batches": val_skipped,
                        f"fold{self.fold}/lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                    log_dict.update({
                        f"fold{self.fold}/{k}": v for k, v in val_metrics.items()
                    })
                    wandb.log(log_dict)

                progress.update(task, advance=1,
                                description=(
                                    f"train_loss={train_loss:.4f} train_r2={train_r2:.4f} "
                                    f"val_loss={val_loss:.4f} val_r2={val_metrics['r2']:.4f}"
                                ))
                console.print(
                    f"[fold_{self.fold}] epoch {epoch}/{self.epochs} "
                    f"train_loss={train_loss:.4f} train_r2={train_r2:.4f} "
                    f"val_loss={val_loss:.4f} val_r2={val_metrics['r2']:.4f} "
                    f"train_skipped={train_skipped} val_skipped={val_skipped}"
                )

                # Early stopping
                if self.early_stopping.step(val_loss):
                    console.print(
                        f"[yellow]Early stopping at epoch {epoch} "
                        f"(best val_loss={self.ckpt_manager.best_loss:.6f})[/yellow]"
                    )
                    break

        console.print(
            f"[green]Fold {self.fold} finished. "
            f"Best val_loss={self.ckpt_manager.best_loss:.6f} "
            f"@ {self.ckpt_manager.best_path}[/green]"
        )
        return history

    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader, epoch: int) -> Tuple[float, float, int]:
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)
        effective_batches = 0
        skipped_batches = 0
        all_preds, all_targets = [], []

        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                x, y, sample_weight = batch
                sample_weight = sample_weight.to(self.device, non_blocking=True)
            else:
                x, y = batch
                sample_weight = None
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if not torch.isfinite(x).all():
                skipped_batches += 1
                continue
            if not torch.isfinite(y).any():
                skipped_batches += 1
                continue

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=self.cfg.training.mixed_precision and self.device.type == "cuda",
            ):
                pred = self.model(x)
                y_trans = self._transform_targets_for_loss(y)
                loss = self._compute_loss(pred, y_trans, sample_weight=sample_weight)
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            effective_batches += 1
            all_preds.append(pred.detach().float())
            all_targets.append(y_trans.detach().float())

            if self.use_wandb and batch_idx % self.log_interval == 0:
                step = (epoch - 1) * n_batches + batch_idx
                batch_r2 = float(r_squared(y.detach().float(), pred.detach().float()).item())
                wandb.log(
                    {
                        f"fold{self.fold}/batch_train_loss": loss.item(),
                        f"fold{self.fold}/batch_train_r2": batch_r2,
                        "global_step": step,
                    }
                )

        if effective_batches == 0:
            return float("nan"), float("nan"), skipped_batches
        epoch_loss = total_loss / effective_batches
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        epoch_r2 = float(r_squared(targets, preds).item())
        return epoch_loss, epoch_r2, skipped_batches

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(
        self, loader: DataLoader
    ) -> Tuple[float, Dict[str, float], int]:
        self.model.eval()
        total_loss = 0.0
        effective_batches = 0
        skipped_batches = 0
        all_preds, all_targets = [], []

        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if not torch.isfinite(x).all():
                skipped_batches += 1
                continue
            if not torch.isfinite(y).any():
                skipped_batches += 1
                continue
            with torch.amp.autocast(
                "cuda",
                enabled=self.cfg.training.mixed_precision and self.device.type == "cuda",
            ):
                pred = self.model(x)
                y_trans = self._transform_targets_for_loss(y)
                loss = self._compute_loss(pred, y_trans, sample_weight=None)
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            total_loss += loss.item()
            effective_batches += 1
            all_preds.append(pred.cpu().float())
            all_targets.append(y_trans.cpu().float())

        if effective_batches == 0:
            metrics = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            return float("nan"), metrics, skipped_batches

        val_loss = total_loss / effective_batches
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        rmse, mae = _finite_rmse_mae(targets, preds)

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": float(r_squared(targets, preds).item()),
        }
        return val_loss, metrics, skipped_batches

    # ------------------------------------------------------------------
    def load_best_weights(self) -> None:
        """Load the best checkpoint into self.model (in-place)."""
        if self.ckpt_manager.best_path is None:
            raise RuntimeError("No checkpoint saved yet.")
        ckpt = torch.load(self.ckpt_manager.best_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        console.print(f"Loaded best weights from {self.ckpt_manager.best_path}")
