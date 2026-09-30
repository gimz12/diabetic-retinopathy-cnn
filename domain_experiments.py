"""Domain shift fixes: (A) fine-tune on 1,000 DDR photos vs (B) train on APTOS + DDR together."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dr import data as D
from dr import evaluate as E
from dr import train as T

RUNS = Path("runs")
OUT = RUNS / "domain_experiments"

APTOS_SPLITS, APTOS_IMG = Path("data/splits"), Path("data/processed/256")
DDR_CSV, DDR_IMG = Path("data/raw/ddr/train.csv"), Path("data/processed/ddr256")
DDR_SPLITS = Path("data/splits_ddr70")  # frozen 70/15/15 of DDR
LOCAL_SPLITS = Path("data/splits_ddr_local")  # A: 1,000 of DDR train + DDR val/test
MULTI_SPLITS = Path("data/splits_multi")  # B: APTOS + DDR train/val
MULTI_IMG = Path("data/processed/multi256")  # symlinks into both caches

N_LOCAL = 1000
IMG_SIZE = 256


# prepare
def prepare() -> None:
    """Write the DDR, local and pooled splits and the combined image folder (idempotent)."""
    ddr = D.make_splits(DDR_CSV, DDR_SPLITS)
    print("DDR 70/15/15:")
    print(D.split_summary(ddr).to_string())

    # A: stratified 1,000 from DDR train
    LOCAL_SPLITS.mkdir(exist_ok=True)
    local, _ = train_test_split(ddr["train"], train_size=N_LOCAL,
                                stratify=ddr["train"]["diagnosis"], random_state=42)
    local.to_csv(LOCAL_SPLITS / "train.csv", index=False)
    ddr["val"].to_csv(LOCAL_SPLITS / "val.csv", index=False)
    ddr["test"].to_csv(LOCAL_SPLITS / "test.csv", index=False)
    print(f"\nlocal fine-tune sample: {len(local)} images, grades "
          f"{local['diagnosis'].value_counts().sort_index().tolist()}")

    # B: pooled train/val; tests stay per domain
    aptos = {name: pd.read_csv(APTOS_SPLITS / f"{name}.csv") for name in ("train", "val", "test")}
    MULTI_SPLITS.mkdir(exist_ok=True)
    for name in ("train", "val"):
        pooled = pd.concat([aptos[name].assign(domain="aptos"), ddr[name].assign(domain="ddr")])
        pooled.to_csv(MULTI_SPLITS / f"{name}.csv", index=False)
        print(f"multi {name}: {len(pooled)} images ({len(aptos[name])} APTOS + {len(ddr[name])} DDR)")
    # make_loaders needs a test.csv; DDR test is scored separately
    aptos["test"].to_csv(MULTI_SPLITS / "test.csv", index=False)

    MULTI_IMG.mkdir(parents=True, exist_ok=True)
    made = 0
    for src_dir in (APTOS_IMG, DDR_IMG):
        for src in src_dir.glob("*.png"):
            dst = MULTI_IMG / src.name
            if not dst.exists():
                dst.symlink_to(src.resolve())
                made += 1
    print(f"combined image folder: {len(list(MULTI_IMG.iterdir()))} files ({made} new links)")


# A
def finetune() -> None:
    """A: continue training the shipped model on 1,000 DDR photos."""
    run_dir = RUNS / "resnet50_256_ddr_finetune"
    if (run_dir / "history.json").exists():
        print("fine-tune already done"); return

    model, cfg = E.load_checkpoint(RUNS / "resnet50_256" / "best.pt")
    loaders = D.make_loaders(LOCAL_SPLITS, DDR_IMG, img_size=IMG_SIZE, batch_size=32, num_workers=4)

    # gentler than a fresh run: short warm-up, then a third of the usual learning rate
    T.fit(loaders, run_dir=run_dir, model=model, config={
        **{k: cfg[k] for k in ("arch", "img_size", "dropout", "balance")},
        "phase1_epochs": 2, "phase1_lr": 3e-4,
        "phase2_epochs": 15, "phase2_lr_backbone": 3e-5, "phase2_lr_head": 3e-4,
        "patience": 4,
        "note": f"fine-tuned from resnet50_256 on {N_LOCAL} DDR images",
    })


# B
def multi() -> None:
    """B: train a fresh ResNet50 on APTOS and DDR together."""
    run_dir = RUNS / "resnet50_256_multi"
    if (run_dir / "history.json").exists():
        print("multi-source run already done"); return

    loaders = D.make_loaders(MULTI_SPLITS, MULTI_IMG, img_size=IMG_SIZE, batch_size=32, num_workers=4)
    for name, loader in loaders.items():
        print(f"  {name}: {len(loader.dataset)} images")

    # 4.4x more images per epoch, so fewer epochs
    T.fit(loaders, run_dir=run_dir, config={
        "arch": "resnet50", "img_size": IMG_SIZE,
        "phase1_epochs": 4, "phase2_epochs": 15, "patience": 4,
        "note": "trained on APTOS train + DDR train (70% of DDR)",
    })


# evaluate
def domain_loader(split_csv: Path, img_dir: Path) -> DataLoader:
    return DataLoader(D.FundusDataset(split_csv, img_dir, D.eval_tfms(IMG_SIZE)),
                      batch_size=32, num_workers=4)


def evaluate() -> None:
    """Score the shipped, fine-tuned and multi-source models on both test sets."""
    OUT.mkdir(exist_ok=True)
    # *_clean.csv drop test photos that have a twin in train/val
    tests = {
        "APTOS test": domain_loader(APTOS_SPLITS / "test_clean.csv", APTOS_IMG),
        "DDR test": domain_loader(DDR_SPLITS / "test_clean.csv", DDR_IMG),
    }
    models = {
        "shipped (APTOS only)": RUNS / "resnet50_256" / "best.pt",
        "A: fine-tuned on 1,000 DDR": RUNS / "resnet50_256_ddr_finetune" / "best.pt",
        "B: trained on APTOS + DDR": RUNS / "resnet50_256_multi" / "best.pt",
    }

    rows, per_class = [], {}
    for label, ckpt in models.items():
        if not ckpt.exists():
            print(f"skipping {label}: no checkpoint yet"); continue
        model, _ = E.load_checkpoint(ckpt)
        for test_name, loader in tests.items():
            y, pred, probs = E.predict(model, loader, tta=True)
            m = E.metrics(y, pred)
            b = E.binary_metrics(y, pred, probs)
            rows.append({"model": label, "test set": test_name,
                         "kappa": m["quadratic_kappa"], "accuracy": m["accuracy"],
                         "macro F1": m["macro_f1"], "sensitivity": b["sensitivity"],
                         "specificity": b["specificity"]})
            per_class[(label, test_name)] = E.per_class_table(y, pred)["recall"].head(5)
            slug = f"{label.split(':')[0].split(' ')[0].lower()}_{test_name.split()[0].lower()}"
            E.plot_confusion(y, pred, OUT / f"confusion_{slug}.png")
            plt.close("all")
            print(f"{label:<28} {test_name:<11} kappa {m['quadratic_kappa']:.3f}  "
                  f"acc {m['accuracy']:.3f}  sens {b['sensitivity']:.3f}  spec {b['specificity']:.3f}")

    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT / "summary.csv", index=False)
    recall = pd.DataFrame(per_class).round(3)
    recall.to_csv(OUT / "per_grade_recall.csv")

    print("\n=== summary ===")
    print(table.pivot(index="model", columns="test set", values="kappa").round(3).to_string())
    print("\n=== per-grade recall ===")
    print(recall.to_string())


# ship
def ship() -> None:
    """Make the multi-source model the one the demo loads (runs/best/best.pt), TTA on."""
    ckpt = torch.load(RUNS / "resnet50_256_multi" / "best.pt", map_location="cpu", weights_only=False)
    ckpt["config"].update({"tta": True, "thresholds": [0.5, 1.5, 2.5, 3.5]})
    (RUNS / "best").mkdir(exist_ok=True)
    torch.save(ckpt, RUNS / "best" / "best.pt")
    print("runs/best/best.pt <- resnet50_256_multi (tta on)")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    {"prepare": prepare, "finetune": finetune, "multi": multi,
     "evaluate": evaluate, "ship": ship}[step]()
