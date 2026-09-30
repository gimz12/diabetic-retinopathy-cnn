"""Scores every model on an unseen dataset (IDRiD or Messidor-2); nothing is tuned here."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

from dr import data as D
from dr import evaluate as E
from dr import preprocess as P

RUNS = Path("runs")
MODELS = {
    "APTOS only": RUNS / "best_aptos" / "best.pt",
    "A: fine-tuned on 1,000 DDR": RUNS / "resnet50_256_ddr_finetune" / "best.pt",
    "B: APTOS + DDR": RUNS / "best" / "best.pt",
}
IMG_SIZE = 256


def overlap_with_training(raw_dir: Path) -> int:
    """Number of these photos byte-identical to an APTOS or DDR photo."""
    seen = set()
    for folder in (Path("data/raw/aptos/all_images"), Path("data/raw/ddr/all_images")):
        for p in folder.iterdir():
            seen.add(hashlib.md5(p.read_bytes()).hexdigest())
    return sum(hashlib.md5(p.read_bytes()).hexdigest() in seen for p in raw_dir.iterdir())


def main(name: str) -> None:
    raw = Path("data/raw") / name
    processed = Path("data/processed") / f"{name}256"
    out = RUNS / "external" / name
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(raw / "train.csv").astype({"id_code": str})
    print(f"{name}: {len(labels)} labelled photos, grades "
          f"{labels.diagnosis.value_counts().sort_index().tolist()}")

    dup = overlap_with_training(raw / "all_images")
    print(f"photos identical to an APTOS/DDR training photo: {dup}")

    ext = next((raw / "all_images").iterdir()).suffix
    print("preprocessing:", P.cache_dataset(raw / "all_images", processed, ids=labels.id_code.tolist(), ext=ext))

    loader = DataLoader(D.FundusDataset(raw / "train.csv", processed, D.eval_tfms(IMG_SIZE)),
                        batch_size=32, num_workers=4)

    rows, recalls = [], {}
    for label, ckpt in MODELS.items():
        model, _ = E.load_checkpoint(ckpt)
        y, pred, probs = E.predict(model, loader, tta=True)
        m, b = E.metrics(y, pred), E.binary_metrics(y, pred, probs)
        rows.append({"model": label, "dataset": name, "n": len(y), **{k: round(v, 4) for k, v in m.items()},
                     "sensitivity": round(b["sensitivity"], 4), "specificity": round(b["specificity"], 4),
                     "roc_auc": round(b["roc_auc"], 4)})
        recalls[label] = E.per_class_table(y, pred)["recall"].head(5)
        slug = label.split(":")[0].split(" ")[0].lower()
        E.plot_confusion(y, pred, out / f"confusion_{slug}.png")
        plt.close("all")
        print(f"  {label:<28} kappa {m['quadratic_kappa']:.3f}  acc {m['accuracy']:.3f}  "
              f"sens {b['sensitivity']:.3f}  spec {b['specificity']:.3f}  auc {b['roc_auc']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    recall = pd.DataFrame(recalls).round(3)
    recall.to_csv(out / "per_grade_recall.csv")
    print("\nper-grade recall:")
    print(recall.to_string())
    (out / "meta.json").write_text(json.dumps({"n": len(labels), "identical_to_training": dup}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
