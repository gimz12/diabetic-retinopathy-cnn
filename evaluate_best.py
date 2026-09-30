"""Round 1: pick the best run on validation, then score it once on the APTOS test split and on DDR."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dr import data as D
from dr import evaluate as E
from dr import preprocess as P
from dr.model import DEFAULT_THRESHOLDS, outputs_to_grades

RUNS, SPLITS, PROCESSED = Path("runs"), Path("data/splits"), Path("data/processed/256")
DDR, DDR_PROCESSED, DDR_SPLIT = Path("data/raw/ddr"), Path("data/processed/ddr256"), Path("data/splits_ddr")


def tuning_table() -> pd.DataFrame:
    """Every finished APTOS-only run ranked by best validation kappa; saved to runs/tuning_table.csv."""
    rows = []
    for history_file in sorted(RUNS.glob("*/history.json")):
        record = json.loads(history_file.read_text())
        cfg = record["config"]
        if "note" in cfg:
            # domain experiments have their own table
            continue
        rows.append({
            "run": history_file.parent.name,
            "architecture": cfg["arch"],
            "head": "regression" if cfg.get("regression") else "classification",
            "image size": cfg["img_size"],
            "dropout": cfg["dropout"],
            "balancing": cfg["balance"],
            "epochs": len(record["history"]),
            "minutes": record["minutes"],
            "best val kappa": round(record["best_val_kappa"], 4),
        })
    table = pd.DataFrame(rows).sort_values("best val kappa", ascending=False).reset_index(drop=True)
    table.to_csv(RUNS / "tuning_table.csv", index=False)
    return table


def score(y_true, raw, thresholds) -> float:
    y_pred = outputs_to_grades(torch.from_numpy(raw), thresholds).numpy()
    return E.metrics(y_true, y_pred)["quadratic_kappa"]


def choose_settings(model, cfg, val_loader) -> tuple[dict, pd.DataFrame]:
    """Try TTA on/off (and tuned cut points for a regression head) on validation; return the best setting and the table."""
    regression = cfg.get("regression", False)
    rows, candidates = [], []

    for tta in (False, True):
        y_true, raw = E.predict_raw(model, val_loader, tta=tta)
        options = [("default", list(DEFAULT_THRESHOLDS))]
        if regression:
            options.append(("tuned on val", E.optimise_thresholds(raw[:, 0], y_true)))
        for label, cuts in options:
            kappa = score(y_true, raw, cuts)
            setting = {"tta": tta, "thresholds": cuts}
            candidates.append((kappa, setting))
            rows.append({
                "test-time augmentation": "on" if tta else "off",
                "grade cut points": label if regression else "n/a (classifier)",
                "cuts": cuts if regression else "",
                "val kappa": round(kappa, 4),
            })

    best_kappa, best = max(candidates, key=lambda c: c[0])
    table = pd.DataFrame(rows)
    return best, table


def evaluate_ddr(model, cfg, settings):
    """Score the model on all graded DDR images, or return None if DDR is not prepared."""
    if not (DDR / "train.csv").exists():
        print("\nDDR not prepared, skipping external validation.")
        print("  download mariaherrerot/ddrdataset into data/raw/ddr, then run:")
        print("  .venv/bin/python prepare_data.py --root data/raw/ddr")
        return None

    sample = pd.read_csv(DDR / "train.csv")
    ext = next((DDR / "all_images").iterdir()).suffix
    print(f"\nDDR: preprocessing {len(sample)} images (cached after the first run)")
    P.cache_dataset(DDR / "all_images", DDR_PROCESSED, ids=sample["id_code"].astype(str).tolist(), ext=ext)

    DDR_SPLIT.mkdir(exist_ok=True)
    sample.to_csv(DDR_SPLIT / "test.csv", index=False)
    ds = D.FundusDataset(DDR_SPLIT / "test.csv", DDR_PROCESSED, D.eval_tfms(cfg["img_size"]))
    y_true, y_pred, probs = E.predict(model, DataLoader(ds, batch_size=32, num_workers=4),
                                      tta=settings["tta"], thresholds=settings["thresholds"])
    return y_true, y_pred, probs


def main() -> None:
    table = tuning_table()
    print("\n=== 1. tuning comparison (validation kappa during training) ===")
    print(table.to_string(index=False))

    best_run = RUNS / table.iloc[0]["run"]
    print(f"\n=== 2. winner: {best_run.name} ===")
    model, cfg = E.load_checkpoint(best_run / "best.pt")

    loaders = D.make_loaders(SPLITS, PROCESSED, img_size=cfg["img_size"], batch_size=32, num_workers=4)

    print("\n=== 3. prediction refinements, chosen on validation ===")
    settings, refinements = choose_settings(model, cfg, loaders["val"])
    print(refinements.to_string(index=False))
    print(f"chosen: tta={settings['tta']}, cuts={settings['thresholds']}")
    refinements.to_csv(best_run / "refinements_val.csv", index=False)

    print("\n=== 4. held-out test set, opened once ===")
    y_true, y_pred, probs = E.predict(model, loaders["test"], tta=settings["tta"],
                                      thresholds=settings["thresholds"])
    headline = E.metrics(y_true, y_pred)
    for key, value in headline.items():
        print(f"  {key:<18} {value:.4f}")

    per_class = E.per_class_table(y_true, y_pred)
    print("\n" + per_class.to_string())
    per_class.to_csv(best_run / "per_class_metrics.csv")

    # regression head: rank screening cases by raw severity
    dr_score = None
    if cfg.get("regression"):
        _, raw = E.predict_raw(model, loaders["test"], tta=settings["tta"])
        dr_score = raw[:, 0]
    binary = E.binary_metrics(y_true, y_pred, probs, dr_score=dr_score)
    print("\nhealthy vs any DR:")
    for key in ("accuracy", "sensitivity", "specificity", "roc_auc"):
        print(f"  {key:<12} {binary[key]:.4f}")

    history = json.loads((best_run / "history.json").read_text())["history"]
    results = E.save_results(best_run, y_true, y_pred, probs, history)
    results["settings"] = settings
    results["binary"] = binary
    (best_run / "results.json").write_text(json.dumps(results, indent=2, default=float))

    # Grad-CAM gallery: correct, near misses, gross errors
    test_ds = loaders["test"].dataset
    picks = E.error_gallery(test_ds, y_true, y_pred, probs, n_each=2)
    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.8))
    for col, pick in enumerate(picks):
        tensor, _ = test_ds[pick["index"]]
        image = P.read_rgb(PROCESSED / f"{pick['id_code']}.png")
        cam = E.gradcam(model, tensor, class_idx=pick["pred"])
        axes[0, col].imshow(image)
        axes[0, col].set_title(f"true {pick['true']} / pred {pick['pred']}\n{pick['kind']}", fontsize=9)
        axes[1, col].imshow(E.overlay_cam(image, cam))
        axes[1, col].set_title(f"{pick['confidence']:.0%} confident", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    plt.tight_layout()
    fig.savefig(best_run / "figures" / "gradcam_gallery.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print("\n=== 5. save APTOS-only winner ===")
    ckpt = torch.load(best_run / "best.pt", map_location="cpu", weights_only=False)
    ckpt["config"].update(settings)
    (RUNS / "best_aptos").mkdir(exist_ok=True)
    torch.save(ckpt, RUNS / "best_aptos" / "best.pt")
    print(f"  runs/best_aptos/best.pt <- {best_run.name} with tta={settings['tta']}, cuts={settings['thresholds']}")

    print("\n=== 6. external validation ===")
    ddr = evaluate_ddr(model, cfg, settings)
    if ddr is not None:
        d_true, d_pred, d_probs = ddr
        generalisation = pd.DataFrame({
            "APTOS test (same hospital network)": headline,
            "DDR, all 12,522 (147 Chinese hospitals)": E.metrics(d_true, d_pred),
        }).round(4)
        print(generalisation.to_string())
        generalisation.to_csv(best_run / "generalisation.csv")
        E.plot_confusion(d_true, d_pred, best_run / "figures" / "confusion_matrix_ddr.png")
        plt.close("all")
        print("\nDDR per class:")
        ddr_per_class = E.per_class_table(d_true, d_pred)
        print(ddr_per_class.to_string())
        ddr_per_class.to_csv(best_run / "per_class_metrics_ddr.csv")

        ddr_binary = E.binary_metrics(d_true, d_pred, d_probs)
        print("\nDDR healthy vs any DR:")
        for key in ("accuracy", "sensitivity", "specificity", "roc_auc"):
            print(f"  {key:<12} {ddr_binary[key]:.4f}   (APTOS test {binary[key]:.4f})")
        (best_run / "results_ddr.json").write_text(json.dumps(
            {"overall": E.metrics(d_true, d_pred), "binary": ddr_binary}, indent=2, default=float))


if __name__ == "__main__":
    main()
