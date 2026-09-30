"""Recalibrates the APTOS-only model for DDR using a few local labelled photos, no retraining."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dr import data as D
from dr import evaluate as E

RUNS = Path("runs")
OUT = RUNS / "local_calibration"
DDR_CSV = Path("data/splits_ddr/test.csv")  # written by evaluate_best.py
DDR_PROCESSED = Path("data/processed/ddr256")
SAMPLE_SIZES = (100, 250, 500, 1000)
SEEDS = range(5)
TARGET_SENSITIVITY = 0.90


def ddr_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Shipped model probabilities for every DDR image, cached after the first run."""
    cache = OUT / "ddr_predictions.npz"
    if cache.exists():
        data = np.load(cache)
        return data["y_true"], data["probs"]

    # the APTOS-only model
    model, cfg = E.load_checkpoint(RUNS / "best_aptos" / "best.pt")
    if cfg.get("regression"):
        raise SystemExit("this experiment expects the shipped classifier")
    ds = D.FundusDataset(DDR_CSV, DDR_PROCESSED, D.eval_tfms(cfg["img_size"]))
    print(f"predicting {len(ds)} DDR images with tta={cfg.get('tta', False)} (about 10 min)")
    y_true, probs = E.predict_raw(model, DataLoader(ds, batch_size=32, num_workers=4),
                                  tta=cfg.get("tta", False))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, y_true=y_true, probs=probs)
    return y_true, probs


def screening(is_sick: np.ndarray, flagged: np.ndarray) -> tuple[float, float]:
    """Sensitivity and specificity of a referral decision."""
    sensitivity = flagged[is_sick == 1].mean()
    specificity = 1 - flagged[is_sick == 0].mean()
    return float(sensitivity), float(specificity)


def one_split(y: np.ndarray, probs: np.ndarray, n_local: int, seed: int) -> dict:
    """Tune on a random local sample of n_local photos, score on the rest of DDR."""
    idx = np.arange(len(y))
    local, rest = train_test_split(idx, train_size=n_local, stratify=y, random_state=seed)

    severity = E.expected_grade(probs)
    cuts = E.optimise_thresholds(severity[local], y[local])

    alarm_score = 1 - probs[:, 0]  # probability of any disease
    alarm = E.threshold_for_sensitivity(y[local] > 0, alarm_score[local], TARGET_SENSITIVITY)

    shipped_grade = probs[rest].argmax(1)
    local_grade = np.digitize(severity[rest], cuts, right=True)
    is_sick = (y[rest] > 0).astype(int)

    shipped_sens, shipped_spec = screening(is_sick, (shipped_grade > 0).astype(int))
    local_sens, local_spec = screening(is_sick, (alarm_score[rest] >= alarm).astype(int))

    return {
        "local photos": n_local,
        "seed": seed,
        "kappa shipped": E.metrics(y[rest], shipped_grade)["quadratic_kappa"],
        "kappa recalibrated": E.metrics(y[rest], local_grade)["quadratic_kappa"],
        "sensitivity shipped": shipped_sens,
        "sensitivity recalibrated": local_sens,
        "specificity shipped": shipped_spec,
        "specificity recalibrated": local_spec,
        "cuts": cuts,
        "alarm threshold": alarm,
        "_rest": rest,
        "_local_grade": local_grade,
    }


def plot(summary: pd.DataFrame, save_to: Path) -> None:
    """Held-out kappa and screening sensitivity/specificity against local sample size."""
    n = summary.index.values
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.axhline(summary["kappa shipped"]["mean"].iloc[0], color="#C0705A", linestyle="--",
               label="as shipped (no local photos)")
    ax.errorbar(n, summary["kappa recalibrated"]["mean"], yerr=summary["kappa recalibrated"]["std"],
                marker="o", capsize=4, color="#4878A8", label="cut points tuned locally")
    ax.axhline(0.868, color="grey", linestyle=":", label="APTOS test (home hospital)")
    ax.set_xscale("log")
    ax.set_xticks(n, [str(v) for v in n])
    ax.set_xlabel("labelled local photos used")
    ax.set_ylabel("quadratic kappa on the rest of DDR")
    ax.set_title("Five-grade agreement with DDR doctors")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for metric, colour in (("sensitivity", "#4878A8"), ("specificity", "#5A9E6F")):
        ax.axhline(summary[f"{metric} shipped"]["mean"].iloc[0], color=colour, linestyle="--",
                   alpha=0.7, label=f"{metric}, as shipped")
        ax.errorbar(n, summary[f"{metric} recalibrated"]["mean"],
                    yerr=summary[f"{metric} recalibrated"]["std"],
                    marker="o", capsize=4, color=colour, label=f"{metric}, alarm set locally")
    ax.set_xscale("log")
    ax.set_xticks(n, [str(v) for v in n])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("labelled local photos used")
    ax.set_ylabel("share of held-out DDR eyes")
    ax.set_title(f"Screening, alarm tuned for {TARGET_SENSITIVITY:.0%} sensitivity")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    y, probs = ddr_predictions()
    print(f"{len(y)} DDR images, shipped kappa {E.metrics(y, probs.argmax(1))['quadratic_kappa']:.4f}")

    results = [one_split(y, probs, n, seed) for n in SAMPLE_SIZES for seed in SEEDS]
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    table.to_csv(OUT / "all_splits.csv", index=False)

    numeric = [c for c in table.columns if c.startswith(("kappa", "sensitivity", "specificity"))]
    summary = table.groupby("local photos")[numeric].agg(["mean", "std"])
    summary.round(4).to_csv(OUT / "summary.csv")

    print("\n=== held-out DDR, mean over 5 random local samples (std in brackets) ===")
    for n in SAMPLE_SIZES:
        row = summary.loc[n]
        print(f"\n{n} local photos:")
        for metric in ("kappa", "sensitivity", "specificity"):
            print(f"  {metric:<12} shipped {row[(f'{metric} shipped', 'mean')]:.3f}"
                  f"  ->  recalibrated {row[(f'{metric} recalibrated', 'mean')]:.3f}"
                  f" ({row[(f'{metric} recalibrated', 'std')]:.3f})")

    plot(summary, OUT / "local_calibration.png")

    # per-grade view: 500 local photos, first draw
    example = next(r for r in results if r["local photos"] == 500 and r["seed"] == 0)
    rest, local_grade = example["_rest"], example["_local_grade"]
    comparison = pd.DataFrame({
        "recall as shipped": E.per_class_table(y[rest], probs[rest].argmax(1))["recall"],
        "recall recalibrated": E.per_class_table(y[rest], local_grade)["recall"],
    }).head(5)
    print("\n=== per-grade recall, 500 local photos ===")
    print(comparison.to_string())
    comparison.to_csv(OUT / "per_grade_recall_500.csv")
    E.plot_confusion(y[rest], local_grade, OUT / "confusion_recalibrated_500.png")
    plt.close("all")

    (OUT / "example_500.json").write_text(json.dumps(
        {"cuts": example["cuts"], "alarm threshold": example["alarm threshold"]}, indent=2))
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
