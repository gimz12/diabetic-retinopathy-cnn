"""Two-phase training loop: early stopping on validation kappa, cosine LR, checkpoints."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import (
    build_model,
    freeze_backbone,
    freeze_batchnorm,
    outputs_to_grades,
    param_counts,
    param_groups,
    unfreeze_top,
)

DEFAULT_CONFIG = {
    "arch": "efficientnet_b0",
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "dropout": 0.3,
    "label_smoothing": 0.1,
    "balance": "sampler",  # "sampler" or "weights"
    "regression": False,  # True = ordinal head (one output)
    "seed": 42,
    "phase1_epochs": 8,
    "phase1_lr": 1e-3,
    "phase2_epochs": 25,
    "phase2_lr_backbone": 1e-4,
    "phase2_lr_head": 1e-3,
    "unfreeze_stages": 3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 5,
}


def seed_everything(seed: int = 42) -> None:
    """Seed every RNG so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """MPS, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EarlyStopping:
    """Stop when validation kappa has not improved for `patience` epochs; keeps the best weights."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.best_state: dict | None = None
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Record this epoch; returns True when training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    freeze_bn: bool = False,
    desc: str = "",
) -> dict[str, float]:
    """One pass over a loader; trains if an optimizer is given. Returns loss, acc, kappa."""
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        freeze_batchnorm(model)  # model.train() re-enabled them

    total_loss, seen = 0.0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            if outputs.shape[1] == 1:
                # regression head: distance from the true grade
                loss = criterion(outputs[:, 0], labels.float())
            else:
                loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    # cap runaway gradients
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            all_preds.extend(outputs_to_grades(outputs).tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / max(seen, 1),
        "acc": float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "kappa": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
    }


def fit(
    loaders: dict[str, DataLoader],
    config: dict | None = None,
    run_dir: str | Path = "runs/latest",
    class_weight: torch.Tensor | None = None,
    model: nn.Module | None = None,
) -> tuple[nn.Module, dict]:
    """Train through both phases; writes config.json, history.json and best.pt to run_dir."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg["seed"])
    device = get_device()
    cfg["device"] = str(device)

    if model is None:
        model = build_model(cfg["arch"], dropout=cfg["dropout"], regression=cfg["regression"])
    model = model.to(device)

    if cfg["regression"]:
        # Huber loss: robust to a few mislabelled images
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.CrossEntropyLoss(
            weight=None if class_weight is None else class_weight.to(device),
            # label smoothing: less over-confidence on debatable labels
            label_smoothing=cfg["label_smoothing"],
        )

    history: list[dict] = []
    stopper = EarlyStopping(patience=cfg["patience"])
    started = time.time()

    # phase 1: head only
    freeze_backbone(model)
    cfg["phase1_params"] = param_counts(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["phase1_lr"],
        weight_decay=cfg["weight_decay"],
    )

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    for epoch in range(cfg["phase1_epochs"]):
        train_stats = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
            cfg["grad_clip"], desc=f"p1 e{epoch + 1} train",
        )
        val_stats = run_epoch(model, loaders["val"], criterion, device, desc=f"p1 e{epoch + 1} val")
        history.append(_record(1, epoch, train_stats, val_stats, optimizer))
        _report(history[-1])
        if stopper.step(val_stats["kappa"], model, len(history) - 1):
            print(f"early stop in phase 1 at epoch {epoch + 1}")
            break

    # phase 2: head plus the deepest backbone stages
    unfreeze_top(model, cfg["unfreeze_stages"])
    cfg["phase2_params"] = param_counts(model)
    optimizer = torch.optim.AdamW(
        param_groups(model, cfg["phase2_lr_backbone"], cfg["phase2_lr_head"]),
        weight_decay=cfg["weight_decay"],
    )
    # cosine schedule: LR eases to ~0 by the last epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["phase2_epochs"])

    stopper.bad_epochs = 0  # fresh patience for phase 2
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    for epoch in range(cfg["phase2_epochs"]):
        train_stats = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
            cfg["grad_clip"], freeze_bn=True, desc=f"p2 e{epoch + 1} train",
        )
        val_stats = run_epoch(model, loaders["val"], criterion, device, desc=f"p2 e{epoch + 1} val")
        history.append(_record(2, epoch, train_stats, val_stats, optimizer))
        _report(history[-1])
        scheduler.step()

        stop = stopper.step(val_stats["kappa"], model, len(history) - 1)
        if stopper.best_epoch == len(history) - 1:
            torch.save({"state_dict": model.state_dict(), "config": cfg}, run_dir / "best.pt")
        if stop:
            print(f"early stop in phase 2 at epoch {epoch + 1}")
            break

    stopper.restore(model)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, run_dir / "best.pt")

    summary = {
        "history": history,
        "best_epoch": stopper.best_epoch,
        "best_val_kappa": stopper.best_score,
        "minutes": round((time.time() - started) / 60, 1),
        "config": cfg,
    }
    (run_dir / "history.json").write_text(json.dumps(summary, indent=2))
    print(
        f"done in {summary['minutes']} min | best val kappa "
        f"{stopper.best_score:.4f} at epoch {stopper.best_epoch + 1}"
    )
    return model, summary


def _record(phase: int, epoch: int, train: dict, val: dict, optimizer) -> dict:
    return {
        "phase": phase,
        "epoch": epoch + 1,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": train["loss"],
        "train_acc": train["acc"],
        "val_loss": val["loss"],
        "val_acc": val["acc"],
        "val_kappa": val["kappa"],
    }


def _report(row: dict) -> None:
    print(
        f"phase {row['phase']} epoch {row['epoch']:2d} | "
        f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
        f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} kappa {row['val_kappa']:.4f}"
    )


if __name__ == "__main__":
    # smoke test on random noise: loop runs, artefacts saved, best epoch kept
    import tempfile

    from torch.utils.data import TensorDataset

    seed_everything(0)
    images = torch.randn(48, 3, 64, 64)
    labels = torch.randint(0, 5, (48,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)

    with tempfile.TemporaryDirectory() as tmp:
        model, summary = fit(
            {"train": loader, "val": loader},
            config={"phase1_epochs": 1, "phase2_epochs": 1, "num_workers": 0},
            run_dir=tmp,
            model=build_model("efficientnet_b0", pretrained=False),
        )
        run = Path(tmp)
        assert (run / "best.pt").exists(), "best.pt was not saved"
        assert (run / "history.json").exists(), "history.json was not saved"
        assert (run / "config.json").exists(), "config.json was not saved"
        assert len(summary["history"]) == 2, "expected one epoch per phase"
        assert {"train_loss", "val_kappa", "lr"} <= summary["history"][0].keys()
        assert summary["history"][0]["phase"] == 1 and summary["history"][1]["phase"] == 2

    # regression head uses the same loop
    with tempfile.TemporaryDirectory() as tmp:
        _, reg_summary = fit(
            {"train": loader, "val": loader},
            config={"phase1_epochs": 1, "phase2_epochs": 1, "num_workers": 0, "regression": True},
            run_dir=tmp,
            model=build_model("efficientnet_b0", pretrained=False, regression=True),
        )
        assert len(reg_summary["history"]) == 2 and reg_summary["config"]["regression"]

    # EarlyStopping keeps the peak score
    stopper = EarlyStopping(patience=2)
    tiny = nn.Linear(2, 2)
    assert not stopper.step(0.5, tiny, 0)
    assert not stopper.step(0.9, tiny, 1)
    assert not stopper.step(0.1, tiny, 2)
    assert stopper.step(0.1, tiny, 3), "should stop after patience is exhausted"
    assert stopper.best_epoch == 1 and stopper.best_score == 0.9

    print("train.py self-check passed")
