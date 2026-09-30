"""Round 1 APTOS training runs (kept as a file because DataLoader workers re-import it)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dr import data as D
from dr import train as T

SPLITS = Path("data/splits")
PROCESSED = Path("data/processed/256")
RUNS = Path("runs")

# (config overrides, balancing strategy) per experiment
EXPERIMENTS: dict[str, tuple[dict, str]] = {
    "effnetb0_sampler": ({"arch": "efficientnet_b0"}, "sampler"),
    "resnet50": ({"arch": "resnet50"}, "sampler"),
    "effnetb0_256": ({"arch": "efficientnet_b0", "img_size": 256}, "sampler"),
    "effnetb0_drop5": ({"arch": "efficientnet_b0", "dropout": 0.5}, "sampler"),
    "effnetb0_weights": ({"arch": "efficientnet_b0", "balance": "weights"}, "weights"),
    # round two, after ResNet50 won round one
    "resnet50_256": ({"arch": "resnet50", "img_size": 256}, "sampler"),
    "resnet50_ordinal": ({"arch": "resnet50", "regression": True}, "sampler"),
}


def run(name: str, skip_existing: bool = True) -> None:
    """Train one experiment into runs/<name>/."""
    config, balance = EXPERIMENTS[name]
    run_dir = RUNS / name

    if skip_existing and (run_dir / "history.json").exists():
        print(f"skipping {name}, already complete")
        return

    print(f"\n{'=' * 60}\n{name}: {config}, balance={balance}\n{'=' * 60}", flush=True)

    loaders = D.make_loaders(
        SPLITS, PROCESSED,
        img_size=config.get("img_size", 224),
        batch_size=32,
        num_workers=4,
        balance=balance,
    )
    for split, loader in loaders.items():
        print(f"  {split}: {len(loader.dataset)} images, {len(loader)} batches", flush=True)

    # class weights from the training split for the weighted-loss variant
    weight = None
    if balance == "weights":
        labels = pd.read_csv(SPLITS / "train.csv")["diagnosis"].tolist()
        weight = D.class_weights(labels)
        print(f"  class weights: {weight.numpy().round(2).tolist()}", flush=True)

    T.fit(loaders, config=config, run_dir=run_dir, class_weight=weight)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exp", default="effnetb0_sampler", choices=list(EXPERIMENTS) + ["all"])
    parser.add_argument("--force", action="store_true", help="re-run even if results exist")
    parser.add_argument("--list", action="store_true", help="list the experiments and exit")
    args = parser.parse_args()

    if args.list:
        for name, (config, balance) in EXPERIMENTS.items():
            done = "done" if (RUNS / name / "history.json").exists() else "pending"
            print(f"  {name:<20} {balance:<8} {config}  [{done}]")
        return

    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    for name in names:
        run(name, skip_existing=not args.force)


if __name__ == "__main__":
    main()
