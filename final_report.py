"""Final model evidence on all four datasets (curves, metrics, confusion matrices, Grad-CAM) into runs/final/."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

from dr import CLASSES
from dr import data as D
from dr import evaluate as E
from dr import preprocess as P
from pipeline_v2 import DATASETS, RUNS_DIR, SPLITS, image_dir

BEST = Path("runs/best")
OUT = Path("runs/final")
NAMES = {"aptos": "APTOS (India, home)", "ddr": "DDR (China)",
         "idrid": "IDRiD (India, other hospital, unseen)", "messidor": "Messidor-2 (France, unseen)"}


def source_run() -> str:
    """Name of the training run the shipped checkpoint came from."""
    return (BEST / "source.txt").read_text().strip()


def gradcam_sheet(model, dataset, img_dir, y, pred, probs, title, save_to) -> None:
    picks = E.error_gallery(dataset, y, pred, probs, n_each=2)
    if not picks:
        return
    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.8), squeeze=False)
    for col, pick in enumerate(picks):
        tensor, _ = dataset[pick["index"]]
        image = P.read_rgb(Path(img_dir) / f"{pick['id_code']}.png")
        cam = E.gradcam(model, tensor, class_idx=pick["pred"])
        axes[0, col].imshow(image)
        axes[0, col].set_title(f"true {pick['true']} / pred {pick['pred']}\n{pick['kind']}", fontsize=9)
        axes[1, col].imshow(E.overlay_cam(image, cam))
        axes[1, col].set_title(f"{pick['confidence']:.0%} confident", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(save_to, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(quick: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run = source_run()
    model, cfg = E.load_checkpoint(BEST / "best.pt")
    pipeline, size, tta = cfg.get("pipeline", "v1"), cfg["img_size"], cfg.get("tta", True)
    print(f"final model: {run} ({cfg['arch']}, pipeline {pipeline}, {size} px, TTA {tta})")

    history = json.loads((RUNS_DIR / run / "history.json").read_text())["history"]
    E.plot_curves(history, OUT / "training_curves.png")
    plt.close("all")

    rows = []
    for name in DATASETS:
        split = pd.read_csv(SPLITS / name / "test.csv")
        if quick:
            split = split.groupby("diagnosis", group_keys=False).head(8)
            split.to_csv(OUT / f"_quick_{name}.csv", index=False)
        csv = OUT / f"_quick_{name}.csv" if quick else SPLITS / name / "test.csv"
        ds = D.FundusDataset(csv, image_dir(pipeline, name), D.eval_tfms(size))
        y, pred, probs = E.predict(model, DataLoader(ds, batch_size=16, num_workers=0 if quick else 4),
                                   tta=tta, thresholds=cfg.get("thresholds", [0.5, 1.5, 2.5, 3.5]))
        m, b = E.metrics(y, pred), E.binary_metrics(y, pred, probs)
        rows.append({"hospital": NAMES[name], "photos": len(y), "accuracy": m["accuracy"],
                     "macro F1": m["macro_f1"], "quadratic kappa": m["quadratic_kappa"],
                     "sensitivity (any DR)": b["sensitivity"], "specificity (any DR)": b["specificity"],
                     "ROC-AUC (any DR)": b["roc_auc"]})
        E.per_class_table(y, pred).to_csv(OUT / f"per_class_{name}.csv")
        E.plot_confusion(y, pred, OUT / f"confusion_{name}.png")
        plt.close("all")
        gradcam_sheet(model, ds, image_dir(pipeline, name), y, pred, probs,
                      f"Grad-CAM, {NAMES[name]}", OUT / f"gradcam_{name}.png")
        print(f"  {NAMES[name]:<40} kappa {m['quadratic_kappa']:.3f}  acc {m['accuracy']:.3f}  "
              f"macro F1 {m['macro_f1']:.3f}", flush=True)

    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT / "metrics.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "run": run, "arch": cfg["arch"], "pipeline": pipeline, "img_size": size, "tta": tta,
        "note": cfg.get("note", ""), "generated": datetime.now().isoformat(timespec="minutes"),
        "quick": quick}, indent=2))
    for f in OUT.glob("_quick_*.csv"):
        f.unlink()
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
