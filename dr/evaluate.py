"""Evaluation: metrics, kappa, confusion matrix, curves, TTA, threshold tuning and Grad-CAM."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from . import CLASSES
from .model import DEFAULT_THRESHOLDS, build_model, outputs_to_grades, target_layer
from .train import get_device


def load_checkpoint(path: str | Path, device: torch.device | None = None):
    """Rebuild the model from a best.pt saved by train.fit and load its weights."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(
        cfg["arch"], dropout=cfg["dropout"], pretrained=False,
        regression=cfg.get("regression", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg


def tta_views(images: torch.Tensor) -> list[torch.Tensor]:
    """Four views of a batch: as is, mirrored, upside down, both."""
    return [images, images.flip(3), images.flip(2), images.flip(2).flip(3)]


@torch.no_grad()
def raw_outputs(model, images: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Model output per image (class probabilities, or severity values for a regression head), averaged over TTA views if asked."""
    views = tta_views(images) if tta else [images]
    outs = []
    for view in views:
        out = model(view)
        outs.append(out if out.shape[1] == 1 else F.softmax(out, dim=1))
    return torch.stack(outs).mean(0)


def values_to_probs(values, n_classes: int = 5) -> np.ndarray:
    """Spread each severity value over its two nearest grades (1.3 -> 70 % grade 1, 30 % grade 2)."""
    v = np.clip(np.asarray(values, dtype=float).ravel(), 0, n_classes - 1)
    lo = np.floor(v).astype(int)
    hi = np.minimum(lo + 1, n_classes - 1)
    frac = v - lo
    probs = np.zeros((len(v), n_classes))
    rows = np.arange(len(v))
    probs[rows, lo] += 1 - frac
    probs[rows, hi] += frac
    return probs


@torch.no_grad()
def predict_raw(model, loader: DataLoader, device: torch.device | None = None, tta: bool = False):
    """True labels and raw outputs for a whole loader."""
    device = device or get_device()
    model = model.to(device).eval()
    y_true, outs = [], []
    for images, labels in loader:
        outs.append(raw_outputs(model, images.to(device), tta).cpu().numpy())
        y_true.extend(labels.tolist())
    return np.array(y_true), np.concatenate(outs)


def predict(model, loader: DataLoader, device: torch.device | None = None,
            tta: bool = False, thresholds=DEFAULT_THRESHOLDS):
    """Run the model over a loader; returns (true labels, predicted grades, probabilities)."""
    y_true, raw = predict_raw(model, loader, device, tta)
    y_pred = outputs_to_grades(torch.from_numpy(raw), thresholds).numpy()
    probs = values_to_probs(raw[:, 0]) if raw.shape[1] == 1 else raw
    return y_true, y_pred, probs


def optimise_thresholds(values, y_true, start=DEFAULT_THRESHOLDS,
                        step: float = 0.02, rounds: int = 3) -> list[float]:
    """Tune the regression cut points to maximise kappa; fit on validation only."""
    values = np.asarray(values, dtype=float).ravel()
    cuts = list(start)

    def kappa(c):
        # right=True matches torch.bucketize
        return cohen_kappa_score(y_true, np.digitize(values, c, right=True), weights="quadratic")

    for _ in range(rounds):
        for i in range(len(cuts)):
            lo = cuts[i - 1] + step if i > 0 else 0.0
            hi = cuts[i + 1] - step if i < len(cuts) - 1 else 4.0
            candidates = np.arange(lo, hi + 1e-9, step)
            if len(candidates):
                cuts[i] = float(max(candidates, key=lambda c: kappa(cuts[:i] + [c] + cuts[i + 1:])))
    return [round(c, 3) for c in cuts]


def expected_grade(probs: np.ndarray) -> np.ndarray:
    """Probability-weighted mean grade, one 0-4 severity value per classifier output."""
    return probs @ np.arange(probs.shape[1])


def threshold_for_sensitivity(is_sick: np.ndarray, score: np.ndarray, target: float = 0.9) -> float:
    """Highest alarm threshold that still flags at least `target` of the sick eyes."""
    sick_scores = np.sort(np.asarray(score)[np.asarray(is_sick) == 1])
    # epsilon absorbs float error in (1 - target) * n
    k = int(np.floor((1 - target) * len(sick_scores) + 1e-9))
    return float(sick_scores[k])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro F1 and quadratic kappa."""
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(
            classification_report(y_true, y_pred, output_dict=True, zero_division=0)["macro avg"][
                "f1-score"
            ]
        ),
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Precision, recall, F1 and support per grade."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=[f"{i} {name}" for i, name in enumerate(CLASSES)],
        output_dict=True,
        zero_division=0,
    )
    rows = {k: v for k, v in report.items() if isinstance(v, dict)}
    return pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]].round(4)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                   dr_score: np.ndarray | None = None) -> dict:
    """Healthy vs any-DR: accuracy, sensitivity, specificity, ROC-AUC and the confusion counts."""
    true_dr = (y_true > 0).astype(int)
    pred_dr = (y_pred > 0).astype(int)

    tp = int(((true_dr == 1) & (pred_dr == 1)).sum())
    tn = int(((true_dr == 0) & (pred_dr == 0)).sum())
    fp = int(((true_dr == 0) & (pred_dr == 1)).sum())
    fn = int(((true_dr == 1) & (pred_dr == 0)).sum())

    # any-DR score: 1 - P(grade 0), or the raw severity value for a regression head
    if dr_score is None:
        dr_score = 1.0 - probs[:, 0]

    return {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "roc_auc": float(roc_auc_score(true_dr, dr_score)) if len(set(true_dr)) > 1 else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def plot_curves(history: list[dict], save_to: str | Path | None = None):
    """Loss, accuracy and validation kappa per epoch, with the phase-2 boundary marked."""
    df = pd.DataFrame(history)
    df["step"] = range(1, len(df) + 1)
    boundary = df[df["phase"] == 2]["step"].min() if (df["phase"] == 2).any() else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [
        ("loss", ["train_loss", "val_loss"], "Loss"),
        ("acc", ["train_acc", "val_acc"], "Accuracy"),
        ("kappa", ["val_kappa"], "Validation quadratic kappa"),
    ]
    for ax, (_, cols, title) in zip(axes, panels):
        for col in cols:
            ax.plot(df["step"], df[col], marker="o", markersize=3, label=col.replace("_", " "))
        if boundary is not None:
            ax.axvline(boundary - 0.5, color="grey", linestyle="--", linewidth=1)
            ax.text(boundary - 0.4, ax.get_ylim()[1], " phase 2", va="top", fontsize=8, color="grey")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, save_to: str | Path | None = None):
    """5x5 confusion matrix as counts and row-normalised recall."""
    labels = list(range(len(CLASSES)))
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.nan_to_num(counts / counts.sum(axis=1, keepdims=True))

    names = [f"{i} {n}" for i, n in enumerate(CLASSES)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, data, fmt, title in (
        (axes[0], counts, "d", "Confusion matrix (counts)"),
        (axes[1], normed, ".2f", "Row-normalised (recall per grade)"),
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", cbar=False,
            xticklabels=names, yticklabels=names, ax=ax,
        )
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def gradcam(model, image_tensor: torch.Tensor, class_idx: int | None = None,
            device: torch.device | None = None) -> np.ndarray:
    """Grad-CAM heatmap (0-1, input size) for the predicted or given class."""
    device = device or get_device()
    model = model.to(device).eval()
    layer = target_layer(model)

    captured = {}
    handles = [
        layer.register_forward_hook(lambda _m, _i, out: captured.__setitem__("activations", out)),
        layer.register_full_backward_hook(
            lambda _m, _gi, gout: captured.__setitem__("gradients", gout[0])
        ),
    ]

    try:
        batch = image_tensor.unsqueeze(0).to(device)
        output = model(batch)
        if output.shape[1] == 1:
            # regression head: explain the severity value
            target = output[0, 0]
        else:
            idx = output.argmax(1).item() if class_idx is None else class_idx
            target = output[0, idx]

        model.zero_grad()
        target.backward()

        activations = captured["activations"][0]  # (channels, h, w)
        weights = captured["gradients"][0].mean(dim=(1, 2))  # mean gradient per channel
        cam = F.relu((weights[:, None, None] * activations).sum(0))
    finally:
        for handle in handles:
            handle.remove()

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[1]))
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Paint a heatmap over an RGB image."""
    if rgb.shape[:2] != cam.shape[:2]:
        cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb + alpha * heat)


def error_gallery(dataset, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                  n_each: int = 2) -> list[dict]:
    """Pick the most confident correct, adjacent-error and gross-error examples for the report."""
    gap = np.abs(y_true - y_pred)
    buckets = {
        "correct": np.flatnonzero(gap == 0),
        "adjacent error": np.flatnonzero(gap == 1),
        "gross error": np.flatnonzero(gap >= 2),
    }

    picks = []
    for kind, indices in buckets.items():
        # most confident first
        chosen = sorted(indices, key=lambda i: -probs[i].max())[:n_each]
        for i in chosen:
            picks.append(
                {
                    "index": int(i),
                    "kind": kind,
                    "true": int(y_true[i]),
                    "pred": int(y_pred[i]),
                    "confidence": float(probs[i].max()),
                    "id_code": dataset.ids[i] if hasattr(dataset, "ids") else str(i),
                }
            )
    return picks


def save_results(run_dir: str | Path, y_true, y_pred, probs, history=None) -> dict:
    """Write results.json and the figures for one run."""
    run_dir = Path(run_dir)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    results = {
        "overall": metrics(y_true, y_pred),
        "binary": binary_metrics(y_true, y_pred, probs),
        "per_class": per_class_table(y_true, y_pred).to_dict(),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))

    plot_confusion(y_true, y_pred, figures / "confusion_matrix.png")
    if history:
        plot_curves(history, figures / "training_curves.png")
    plt.close("all")
    return results


if __name__ == "__main__":
    # self-check with hand-computable synthetic predictions
    rng = np.random.default_rng(0)

    # perfect classifier
    y = rng.integers(0, 5, 200)
    perfect = metrics(y, y.copy())
    assert perfect["accuracy"] == 1.0 and perfect["quadratic_kappa"] == 1.0

    # "always healthy" classifier: high accuracy, zero kappa
    skewed = np.array([0] * 100 + [1] * 20 + [2] * 50 + [3] * 10 + [4] * 20)
    lazy = np.zeros_like(skewed)
    lazy_scores = metrics(skewed, lazy)
    assert lazy_scores["accuracy"] > 0.45, "sanity: the lazy model should look accurate"
    assert lazy_scores["quadratic_kappa"] == 0.0, "kappa must expose the lazy model"
    table = per_class_table(skewed, lazy)
    assert table.loc["4 Proliferative DR", "recall"] == 0.0

    # kappa punishes distant errors more
    truth = np.array([4, 4, 4, 4])
    near = cohen_kappa_score(np.append(truth, 0), np.append([3, 3, 3, 3], 0), weights="quadratic")
    far = cohen_kappa_score(np.append(truth, 0), np.append([0, 0, 0, 0], 0), weights="quadratic")
    assert near > far, "quadratic kappa should prefer near misses"

    # binary metrics by hand
    bt = np.array([0, 0, 1, 2, 3, 4])
    bp = np.array([0, 1, 1, 0, 3, 4])
    bprobs = np.zeros((6, 5))
    bprobs[np.arange(6), bp] = 1.0
    b = binary_metrics(bt, bp, bprobs)
    assert b["tp"] == 3 and b["fn"] == 1 and b["tn"] == 1 and b["fp"] == 1
    assert abs(b["sensitivity"] - 0.75) < 1e-9 and abs(b["specificity"] - 0.5) < 1e-9

    # Grad-CAM shape and range
    model = build_model("efficientnet_b0", pretrained=False)
    cam = gradcam(model, torch.randn(3, 224, 224))
    assert cam.shape == (224, 224), f"unexpected cam shape {cam.shape}"
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6

    overlay = overlay_cam(np.full((224, 224, 3), 120, dtype=np.uint8), cam)
    assert overlay.shape == (224, 224, 3) and overlay.dtype == np.uint8

    # TTA: fourth view is the 180-degree rotation
    x = torch.arange(4.0).view(1, 1, 2, 2)
    views = tta_views(x)
    assert len(views) == 4 and torch.equal(views[3], torch.rot90(x, 2, dims=(2, 3)))

    # regression head: soft probs, threshold tuning, Grad-CAM
    probs_reg = values_to_probs([0.0, 1.3, 4.0, 9.0])
    assert np.allclose(probs_reg.sum(1), 1.0)
    assert np.allclose(probs_reg[1, [1, 2]], [0.7, 0.3]) and probs_reg[3, 4] == 1.0

    # under-shooting model: tuned cuts beat the defaults
    grades = rng.integers(0, 5, 300)
    shifted = grades - 0.3 + rng.normal(0, 0.1, 300)
    tuned = optimise_thresholds(shifted, grades)
    default_k = cohen_kappa_score(grades, np.digitize(shifted, DEFAULT_THRESHOLDS, right=True), weights="quadratic")
    tuned_k = cohen_kappa_score(grades, np.digitize(shifted, tuned, right=True), weights="quadratic")
    assert tuned_k > default_k, f"threshold tuning did not help ({tuned_k:.3f} vs {default_k:.3f})"
    assert all(a < b for a, b in zip(tuned, tuned[1:])), "cuts must stay in order"

    reg_model = build_model("resnet50", pretrained=False, regression=True)
    reg_cam = gradcam(reg_model, torch.randn(3, 224, 224))
    assert reg_cam.shape == (224, 224)

    from torch.utils.data import TensorDataset
    tiny = DataLoader(TensorDataset(torch.randn(6, 3, 64, 64), torch.tensor([0, 1, 2, 3, 4, 0])), batch_size=3)
    for m in (model, reg_model):
        yt, yp, pr = predict(m, tiny, tta=True)
        assert yp.shape == (6,) and pr.shape == (6, 5) and np.allclose(pr.sum(1), 1.0, atol=1e-5)

    # expected grade and sensitivity-targeted threshold
    assert np.allclose(expected_grade(np.array([[0.1, 0.6, 0.3, 0, 0]])), [1.2])
    sick = np.array([1] * 10 + [0] * 10)
    scores = np.concatenate([np.linspace(0.1, 1.0, 10), np.linspace(0.0, 0.5, 10)])
    t = threshold_for_sensitivity(sick, scores, target=0.9)
    caught = (scores[sick == 1] >= t).mean()
    assert caught >= 0.9, f"threshold catches only {caught:.0%} of sick eyes"
    assert t > scores[sick == 1].min(), "threshold should be as strict as the target allows"

    print("evaluate.py self-check passed")
