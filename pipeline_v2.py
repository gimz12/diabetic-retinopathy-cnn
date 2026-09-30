"""Pipeline v2: clean, split, cache, train and evaluate every experiment on the cleaned data."""

from __future__ import annotations

import json
import sys
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
from dr import train as T

DATASETS = ("aptos", "ddr", "idrid", "messidor")
RAW = {n: Path("data/raw") / n for n in DATASETS}
OLD_CACHE = {"aptos": "data/processed/256", "ddr": "data/processed/ddr256",
             "idrid": "data/processed/idrid256", "messidor": "data/processed/messidor256"}
CLEAN = Path("data/clean")
SPLITS = Path("data/splits_v2")
CACHE = Path("data/processed/v2_512")
CACHE_V3 = Path("data/processed/v3_512")  # v2 + common outline
CACHE_V4 = Path("data/processed/v4_512")  # v3 + Ben Graham + CLAHE
RUNS_DIR = Path("runs/v2")
GROUPS = Path("runs/dedup/groups_aptos_ddr_idrid_messidor.json")

# each run changes one thing relative to the one above it
RUNS = {
    "r0_oldchain_256": dict(pipeline="v1", img_size=256, batch=32, multi=False,
                            note="v1 enhancement chain, 256 px, cleaned splits (baseline)"),
    "r1_v2_256":       dict(pipeline="v2", img_size=256, batch=32, multi=False,
                            note="v2: no enhancement + strong augmentation, 256 px"),
    # v2 photos with v1 augmentation: isolates preprocessing from augmentation
    "r1b_v2prep_v1aug_256": dict(pipeline="v2", aug="v1", img_size=256, batch=32, multi=False,
                            note="v2 preprocessing (no enhancement) with v1 augmentation, 256 px"),
    "r2_v2_448":       dict(pipeline="v2", img_size=448, batch=16, multi=False,
                            note="v2 at 448 px"),
    "r3_v2_448_multi": dict(pipeline="v2", img_size=448, batch=16, multi=True,
                            note="v2 at 448 px, APTOS + DDR, domain x class balanced sampler"),
    # v3 = v2 + common outline; each ConvNeXt run matches the ResNet50 run above it
    "s2_resnet_v3_448":       dict(arch="resnet50", pipeline="v3", aug="v2", img_size=448, batch=16, multi=False,
                                   note="ResNet50, v3 (common FOV band), APTOS only"),
    "c2_convnext_v3_448":     dict(arch="convnext_tiny", pipeline="v3", aug="v2", img_size=448, batch=16, multi=False,
                                   note="ConvNeXt-Tiny, otherwise identical to s2_resnet_v3_448"),
    "s3_resnet_v3_448_multi": dict(arch="resnet50", pipeline="v3", aug="v2", img_size=448, batch=16, multi=True,
                                   note="ResNet50, v3, APTOS + DDR, domain x class balanced"),
    "c3_convnext_v3_448_multi": dict(arch="convnext_tiny", pipeline="v3", aug="v2", img_size=448, batch=16, multi=True,
                                   note="ConvNeXt-Tiny, otherwise identical to s3_resnet_v3_448_multi"),
    # v4 = v3 + contrast and edge enhancement; otherwise identical to s2 / s3
    "e2_resnet_v4_448":       dict(arch="resnet50", pipeline="v4", aug="v2", img_size=448, batch=16, multi=False,
                                   note="ResNet50, v4 (v3 + Ben Graham + CLAHE), APTOS only"),
    "e3_resnet_v4_448_multi": dict(arch="resnet50", pipeline="v4", aug="v2", img_size=448, batch=16, multi=True,
                                   note="ResNet50, v4 (v3 + Ben Graham + CLAHE), APTOS + DDR, domain x class balanced"),
    # one severity output (ordinal regression) instead of 5 scores; otherwise identical to s3
    "o3_resnet_v3_448_multi_single": dict(arch="resnet50", pipeline="v3", aug="v2", img_size=448, batch=16, multi=True,
                                   regression=True,
                                   note="ResNet50, v3, APTOS + DDR, ONE severity output (regression head)"),
}


# clean
def clean() -> None:
    """Keep one photo per duplicate group; drop groups whose copies disagree on the grade."""
    groups = json.loads(GROUPS.read_text())
    group_of = {member: gi for gi, members in enumerate(groups) for member in members}
    CLEAN.mkdir(exist_ok=True)
    summary = []

    for name in DATASETS:
        df = pd.read_csv(RAW[name] / "train.csv").astype({"id_code": str})
        df["group"] = [group_of.get(f"{name}:{i}", f"solo:{name}:{i}") for i in df.id_code]

        grades_per_group = df.groupby("group")["diagnosis"].nunique()
        conflicting = set(grades_per_group[grades_per_group > 1].index)
        dropped_conflict = df[df.group.isin(conflicting)]
        kept = df[~df.group.isin(conflicting)].drop_duplicates("group", keep="first")

        kept[["id_code", "diagnosis"]].to_csv(CLEAN / f"{name}.csv", index=False)
        summary.append({"dataset": name, "raw": len(df),
                        "in duplicate groups": int(df.group.map(lambda g: not str(g).startswith("solo")).sum()),
                        "dropped (conflicting grades)": len(dropped_conflict),
                        "dropped (extra copies)": len(df) - len(dropped_conflict) - len(kept),
                        "clean": len(kept)})
    table = pd.DataFrame(summary)
    table.to_csv(CLEAN / "summary.csv", index=False)
    print(table.to_string(index=False))


# split
def split() -> None:
    for name in ("aptos", "ddr"):
        s = D.make_splits(CLEAN / f"{name}.csv", SPLITS / name)
        print(f"\n{name}:\n{D.split_summary(s).to_string()}")
    for name in ("idrid", "messidor"):
        out = SPLITS / name
        out.mkdir(parents=True, exist_ok=True)
        pd.read_csv(CLEAN / f"{name}.csv").to_csv(out / "test.csv", index=False)
        print(f"\n{name}: whole set is the test set")

    # pooled APTOS + DDR with a domain column; tests stay per dataset
    multi = SPLITS / "multi"
    multi.mkdir(exist_ok=True)
    for part in ("train", "val"):
        pooled = pd.concat([pd.read_csv(SPLITS / n / f"{part}.csv").assign(domain=n) for n in ("aptos", "ddr")])
        pooled.to_csv(multi / f"{part}.csv", index=False)
        print(f"multi {part}: {len(pooled)} ({pooled.domain.value_counts().to_dict()})")
    pd.read_csv(SPLITS / "aptos" / "test.csv").to_csv(multi / "test.csv", index=False)


# cache
def cache(version: str = "v2") -> None:
    """Preprocess every photo of every dataset once, one folder per pipeline version."""
    out, fn = {"v2": (CACHE, P.preprocess_v2), "v3": (CACHE_V3, P.preprocess_v3),
               "v4": (CACHE_V4, P.preprocess_v4)}[version]
    out.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        ids = pd.read_csv(CLEAN / f"{name}.csv").astype({"id_code": str}).id_code.tolist()
        result = P.cache_dataset(RAW[name] / "all_images", out, ids=ids, size=512, fn=fn)
        print(f"{name} ({version}): {result}", flush=True)


def image_dir(pipeline: str, dataset: str) -> Path | str:
    """Folder of cached photos for a pipeline version."""
    return {"v1": OLD_CACHE[dataset], "v2": CACHE, "v3": CACHE_V3, "v4": CACHE_V4}[pipeline]


# train
def train(run: str) -> None:
    spec = RUNS[run]
    run_dir = RUNS_DIR / run
    if (run_dir / "history.json").exists():
        print(f"{run} already done"); return

    img_dir = image_dir(spec["pipeline"], "aptos")  # v1 uses the old 256 px cache
    splits_dir = SPLITS / ("multi" if spec["multi"] else "aptos")

    loaders = D.make_loaders(splits_dir, img_dir, img_size=spec["img_size"], batch_size=spec["batch"],
                             num_workers=4, balance="domain_class" if spec["multi"] else "sampler",
                             pipeline=spec.get("aug", spec["pipeline"]))
    for k, l in loaders.items():
        print(f"  {k}: {len(l.dataset)} images, batch {spec['batch']}", flush=True)

    config = {"arch": spec.get("arch", "resnet50"), "img_size": spec["img_size"], "batch_size": spec["batch"],
              "pipeline": spec["pipeline"], "aug": spec.get("aug", spec["pipeline"]), "note": spec["note"],
              "balance": "domain_class" if spec["multi"] else "sampler",
              "regression": spec.get("regression", False)}
    if spec["multi"]:
        config.update(phase1_epochs=4, phase2_epochs=15, patience=4)
    T.fit(loaders, config=config, run_dir=run_dir)


# evaluate
def evaluate(only: list[str] | None = None) -> None:
    """Score finished runs on every hospital's test set with TTA; `only` restricts and merges into the existing summary."""
    RUNS_DIR.mkdir(exist_ok=True)
    rows, recalls = [], {}
    for run, spec in RUNS.items():
        if only and run not in only:
            continue
        ckpt = RUNS_DIR / run / "best.pt"
        if not ckpt.exists():
            print(f"skipping {run}: not trained yet"); continue
        model, cfg = E.load_checkpoint(ckpt)
        cuts = E.DEFAULT_THRESHOLDS
        if cfg.get("regression"):
            # regression head: tune cut points on validation only
            val = DataLoader(D.FundusDataset(SPLITS / "multi" / "val.csv", image_dir(spec["pipeline"], "aptos"),
                                             D.eval_tfms(cfg["img_size"])), batch_size=16, num_workers=4)
            yv, raw = E.predict_raw(model, val, tta=True)
            cuts = E.optimise_thresholds(raw[:, 0], yv)
            (RUNS_DIR / run / "cuts.json").write_text(json.dumps(cuts))
            print(f"{run}: tuned cuts {[round(c, 2) for c in cuts]}", flush=True)
        for name in DATASETS:
            loader = DataLoader(D.FundusDataset(SPLITS / name / "test.csv", image_dir(spec["pipeline"], name),
                                                D.eval_tfms(cfg["img_size"])),
                                batch_size=16, num_workers=4)
            y, pred, probs = E.predict(model, loader, tta=True, thresholds=cuts)
            m, b = E.metrics(y, pred), E.binary_metrics(y, pred, probs)
            seen = "seen" if (name == "aptos" or (name == "ddr" and spec["multi"])) else "UNSEEN"
            rows.append({"run": run, "test set": name, "seen in training": seen, "n": len(y),
                         "kappa": round(m["quadratic_kappa"], 4), "accuracy": round(m["accuracy"], 4),
                         "macro_f1": round(m["macro_f1"], 4), "sensitivity": round(b["sensitivity"], 4),
                         "specificity": round(b["specificity"], 4), "auc": round(b["roc_auc"], 4)})
            recalls[(run, name)] = E.per_class_table(y, pred)["recall"].head(5)
            E.plot_confusion(y, pred, RUNS_DIR / run / f"confusion_{name}.png")
            plt.close("all")
            print(f"{run:<18} {name:<9} {seen:<7} kappa {m['quadratic_kappa']:.3f}  sens {b['sensitivity']:.3f}  spec {b['specificity']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    recall = pd.DataFrame(recalls).round(3)
    if only and (RUNS_DIR / "summary.csv").exists():
        old = pd.read_csv(RUNS_DIR / "summary.csv")
        table = pd.concat([old[~old.run.isin(only)], table], ignore_index=True)
        old_r = pd.read_csv(RUNS_DIR / "per_grade_recall.csv", header=[0, 1], index_col=0)
        keep = [c for c in old_r.columns if c[0] not in only]
        recall = pd.concat([old_r[keep], recall], axis=1)
    table.to_csv(RUNS_DIR / "summary.csv", index=False)
    recall.to_csv(RUNS_DIR / "per_grade_recall.csv")
    print("\n=== kappa by run and hospital ===")
    print(table.pivot(index="run", columns="test set", values="kappa").to_string())


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "train":
        train(sys.argv[2])
    elif step == "evaluate" and len(sys.argv) > 2:
        evaluate(sys.argv[2:])
    elif step == "cache":
        cache(sys.argv[2] if len(sys.argv) > 2 else "v2")
    else:
        {"clean": clean, "split": split, "cache": cache, "evaluate": evaluate}[step]()
