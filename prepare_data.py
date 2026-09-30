"""Converts any Kaggle download layout into train.csv plus a folder of image symlinks."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# column names used by different mirrors
ID_COLUMNS = ("id_code", "image", "image_id", "id", "filename", "name")
LABEL_COLUMNS = ("diagnosis", "level", "label", "class", "grade", "dr_grade")

# folder names used when labels are directories
FOLDER_LABELS = {
    "0": 0, "no_dr": 0, "nodr": 0, "no dr": 0, "0 - no_dr": 0,
    "1": 1, "mild": 1, "1 - mild": 1,
    "2": 2, "moderate": 2, "2 - moderate": 2,
    "3": 3, "severe": 3, "3 - severe": 3,
    "4": 4, "proliferate_dr": 4, "proliferative": 4, "proliferative_dr": 4,
    "proliferate dr": 4, "4 - proliferate_dr": 4, "pdr": 4,
}


def index_images(root: Path) -> dict[str, Path]:
    """Map every image file stem to its path, searching the whole tree."""
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            # first match wins
            index.setdefault(path.stem, path)
    return index


def labels_from_csvs(root: Path) -> pd.DataFrame | None:
    """Merge every CSV in the tree that has an id column and a label column."""
    frames = []
    for csv_path in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        lower = {c.lower().strip(): c for c in df.columns}
        id_col = next((lower[c] for c in ID_COLUMNS if c in lower), None)
        label_col = next((lower[c] for c in LABEL_COLUMNS if c in lower), None)
        if id_col is None or label_col is None:
            continue

        frame = df[[id_col, label_col]].copy()
        frame.columns = ["id_code", "diagnosis"]
        frame["source"] = csv_path.name
        frames.append(frame)
        print(f"  read {csv_path.relative_to(root)}: {len(frame)} rows")

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    # strip any file extension from the id
    merged["id_code"] = (
        merged["id_code"].astype(str).str.strip()
        .str.replace(r"\.(png|jpe?g|tiff?)$", "", regex=True, case=False)
    )
    merged["diagnosis"] = pd.to_numeric(merged["diagnosis"], errors="coerce")
    return merged


def labels_from_folders(root: Path) -> pd.DataFrame | None:
    """Derive labels from class-named folders, for mirrors with no CSV."""
    rows = []
    for path in root.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
            continue
        grade = FOLDER_LABELS.get(path.parent.name.lower().strip())
        if grade is not None:
            rows.append({"id_code": path.stem, "diagnosis": grade, "source": path.parent.name})

    if not rows:
        return None
    print(f"  derived {len(rows)} labels from folder names")
    return pd.DataFrame(rows)


def prepare(root: Path, out: Path) -> pd.DataFrame:
    """Read `root`; write train.csv and an all_images symlink folder into `out`."""
    if not root.exists():
        raise SystemExit(f"{root} does not exist. Download the dataset there first.")

    print(f"scanning {root}")
    images = index_images(root)
    if not images:
        raise SystemExit(f"no image files found under {root}")
    print(f"  found {len(images)} image files")

    labels = labels_from_csvs(root)
    if labels is None:
        labels = labels_from_folders(root)
    if labels is None:
        raise SystemExit(
            f"no labels found under {root}. Expected a CSV with an id column and a "
            "grade column, or images sorted into folders named per class."
        )

    # keep valid grades whose image exists
    before = len(labels)
    labels = labels[labels["diagnosis"].between(0, 4)]
    labels["diagnosis"] = labels["diagnosis"].astype(int)
    labels = labels.drop_duplicates(subset="id_code", keep="first")

    missing = labels[~labels["id_code"].isin(images)]
    if len(missing):
        print(f"  warning: {len(missing)} labelled ids have no image file, dropping them")
        print(f"    examples: {missing['id_code'].head(3).tolist()}")
    labels = labels[labels["id_code"].isin(images)]

    if labels.empty:
        raise SystemExit(
            "no labelled image survived matching. The CSV ids and the image "
            f"filenames may not correspond. CSV example: "
            f"{pd.concat([l for l in [labels_from_csvs(root)] if l is not None])['id_code'].head(1).tolist()}, "
            f"file example: {list(images)[:1]}"
        )

    print(f"  matched {len(labels)} of {before} labelled rows to image files")

    # rebuild the symlink folder from scratch
    out.mkdir(parents=True, exist_ok=True)
    link_dir = out / "all_images"
    if link_dir.exists():
        shutil.rmtree(link_dir)
    link_dir.mkdir()

    for id_code in labels["id_code"]:
        source = images[id_code].resolve()
        (link_dir / f"{id_code}{source.suffix}").symlink_to(source)

    final = labels[["id_code", "diagnosis"]].sort_values("id_code").reset_index(drop=True)
    final.to_csv(out / "train.csv", index=False)

    print(f"\nwrote {out / 'train.csv'} ({len(final)} rows)")
    print(f"wrote {link_dir} ({len(list(link_dir.iterdir()))} symlinks)")
    print("\nclass distribution:")
    counts = final["diagnosis"].value_counts().sort_index()
    names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    for grade, count in counts.items():
        print(f"  {grade} {names[grade]:<14} {count:>5}  ({count / len(final):.1%})")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="data/raw/aptos", help="where the download landed")
    parser.add_argument("--out", default=None, help="where to write train.csv (default: --root)")
    args = parser.parse_args()

    root = Path(args.root)
    prepare(root, Path(args.out) if args.out else root)


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        # three fake download layouts must all normalise
        import tempfile

        import cv2
        import numpy as np

        def fake_image(path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((40, 40, 3), 120, np.uint8))

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # A: three CSVs and three folders
            a = base / "a"
            for folder, csv_name, ids in (
                ("train_images", "train_1.csv", ["p1", "p2"]),
                ("val_images", "valid.csv", ["p3"]),
                ("test_images", "test.csv", ["p4"]),
            ):
                for i in ids:
                    fake_image(a / folder / f"{i}.png")
                pd.DataFrame({"id_code": ids, "diagnosis": [i % 5 for i in range(len(ids))]}).to_csv(
                    a / csv_name, index=False
                )
            out_a = prepare(a, a)
            assert len(out_a) == 4, f"layout A: expected 4 rows, got {len(out_a)}"
            assert (a / "all_images" / "p3.png").is_symlink()

            # B: one CSV, ids with a file extension
            b = base / "b"
            for i in ["q1", "q2"]:
                fake_image(b / "train_images" / f"{i}.jpg")
            pd.DataFrame({"image": ["q1.jpg", "q2.jpg"], "level": [0, 3]}).to_csv(
                b / "labels.csv", index=False
            )
            out_b = prepare(b, b)
            assert list(out_b["id_code"]) == ["q1", "q2"], out_b["id_code"].tolist()
            assert list(out_b["diagnosis"]) == [0, 3]

            # C: labels as folder names
            c = base / "c"
            fake_image(c / "colored_images" / "No_DR" / "r1.png")
            fake_image(c / "colored_images" / "Severe" / "r2.png")
            out_c = prepare(c, c)
            assert dict(zip(out_c["id_code"], out_c["diagnosis"])) == {"r1": 0, "r2": 3}

            # a labelled id without an image is dropped
            d = base / "d"
            fake_image(d / "img" / "s1.png")
            pd.DataFrame({"id_code": ["s1", "ghost"], "diagnosis": [1, 2]}).to_csv(
                d / "train.csv", index=False
            )
            out_d = prepare(d, d)
            assert list(out_d["id_code"]) == ["s1"]

        print("\nprepare_data.py self-check passed")
    else:
        main()
