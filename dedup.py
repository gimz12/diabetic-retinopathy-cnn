"""Finds exact and near-duplicate photos (file hash, pHash, CNN features), confirmed by vessel-pattern correlation."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dr import data as D
from dr import evaluate as E
from dr import preprocess as P

RAW = {"aptos": "data/raw/aptos", "ddr": "data/raw/ddr", "idrid": "data/raw/idrid", "messidor": "data/raw/messidor"}
PROC = {"aptos": "data/processed/256", "ddr": "data/processed/ddr256",
        "idrid": "data/processed/idrid256", "messidor": "data/processed/messidor256"}
OUT = Path("runs/dedup")


def phash(img_gray: np.ndarray, hash_size: int = 8, highfreq: int = 4) -> np.ndarray:
    """64-bit perceptual hash (DCT low band of a 32x32 grey thumbnail)."""
    small = cv2.resize(img_gray, (hash_size * highfreq,) * 2, interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:hash_size, :hash_size]
    return (dct > np.median(dct)).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance from one hash to many."""
    return (a[None, :] != b).sum(1)


def thumb(path: Path, size: int = 128) -> np.ndarray:
    """High-pass, standardised grey thumbnail of the inner retina, for pixel comparison."""
    g = cv2.cvtColor(P.crop_circle(P.read_rgb(path)), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 3)
    # compare only the inner disc; the crop edge is the same in every photo
    yy, xx = np.mgrid[:size, :size]
    inner = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= (0.42 * size) ** 2
    v = g[inner]
    out = np.zeros_like(g)
    out[inner] = (v - v.mean()) / (v.std() + 1e-6)
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation over the inner disc (1.0 = identical)."""
    inner = a != 0
    return float((a[inner] * b[inner]).mean()) if inner.any() else 0.0


def load_ids(name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(RAW[name]) / "train.csv").astype({"id_code": str})
    df["dataset"] = name
    return df


def embeddings(name: str, ids: list[str]) -> np.ndarray:
    """Unit-normalised 2048-d features from the APTOS-only ResNet50."""
    model, cfg = E.load_checkpoint("runs/best_aptos/best.pt")
    model.fc = torch.nn.Identity()  # keep the 2048-d features
    tmp = OUT / f"_{name}_ids.csv"
    pd.DataFrame({"id_code": ids, "diagnosis": 0}).to_csv(tmp, index=False)
    loader = DataLoader(D.FundusDataset(tmp, PROC[name], D.eval_tfms(cfg["img_size"])), batch_size=64, num_workers=4)
    feats = []
    with torch.no_grad():
        for images, _ in loader:
            feats.append(model(images.to(E.get_device())).cpu().numpy())
    tmp.unlink()
    f = np.concatenate(feats)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_ids(n) for n in names], ignore_index=True)
    keys = list(zip(df.dataset, df.id_code))
    print(f"{len(df)} photos across {names}")

    # exact copies by file hash
    raw_paths = {}
    for n in names:
        for p in Path(RAW[n]).joinpath("all_images").iterdir():
            raw_paths[(n, p.stem)] = p
    md5 = np.array([hashlib.md5(raw_paths[k].read_bytes()).hexdigest() for k in keys])
    exact_groups = pd.Series(range(len(keys))).groupby(md5).apply(list)
    exact_groups = [g for g in exact_groups if len(g) > 1]
    exact_pairs = {tuple(sorted(p)) for g in exact_groups for p in combinations(g, 2)}
    print(f"exact duplicate groups: {len(exact_groups)}  pairs: {len(exact_pairs)}")

    # perceptual hash
    hashes = np.stack([phash(cv2.imread(f"{PROC[n]}/{i}.png", cv2.IMREAD_GRAYSCALE)) for n, i in keys])
    rng = np.random.default_rng(0)
    neg = rng.integers(0, len(keys), (5000, 2))
    neg = neg[neg[:, 0] != neg[:, 1]]
    neg_d = (hashes[neg[:, 0]] != hashes[neg[:, 1]]).sum(1)
    pos = np.array(list(exact_pairs))
    pos_d = (hashes[pos[:, 0]] != hashes[pos[:, 1]]).sum(1) if len(pos) else np.array([0])
    print(f"phash distance  | exact dups: max {pos_d.max()}  | random pairs: min {neg_d.min()}, 1st pct {np.percentile(neg_d, 1):.0f}, median {np.median(neg_d):.0f}")

    # CNN embeddings
    feats = np.concatenate([embeddings(n, df[df.dataset == n].id_code.tolist()) for n in names])
    sim = feats @ feats.T
    np.fill_diagonal(sim, 0)
    pos_s = sim[pos[:, 0], pos[:, 1]] if len(pos) else np.array([1.0])
    neg_s = sim[neg[:, 0], neg[:, 1]]
    print(f"embedding cosine| exact dups: min {pos_s.min():.3f} | random pairs: max {neg_s.max():.3f}, 99th pct {np.percentile(neg_s, 99):.3f}, median {np.median(neg_s):.3f}")

    # candidates: close hash or very high embedding similarity
    ph_thresh = 8
    emb_thresh = max(0.95, float(np.percentile(neg_s, 99.9)))
    cand = set()
    for i in range(len(keys)):
        d = hamming(hashes[i], hashes[i + 1:])
        for j in np.flatnonzero(d <= ph_thresh) + i + 1:
            cand.add((i, int(j)))
    hi = np.argwhere(np.triu(sim, 1) >= emb_thresh)
    cand |= {(int(a), int(b)) for a, b in hi}
    print(f"candidates from hash/embedding: {len(cand)}  [phash<={ph_thresh} or cosine>={emb_thresh:.3f}]")

    # verify every candidate by pixel correlation of the raw photos
    thumbs: dict[int, np.ndarray] = {}
    def T(i):
        if i not in thumbs:
            thumbs[i] = thumb(raw_paths[keys[i]])
        return thumbs[i]
    pos_c = np.array([ncc(T(a), T(b)) for a, b in pos]) if len(pos) else np.array([1.0])
    neg_c = np.array([ncc(T(a), T(b)) for a, b in neg[:800]])
    print(f"pixel NCC       | exact dups: min {pos_c.min():.3f} | random pairs: max {neg_c.max():.3f}, 99th pct {np.percentile(neg_c, 99):.3f}, median {np.median(neg_c):.3f}")
    # above anything random pairs reach, but low enough to keep re-compressed copies (~0.9)
    ncc_thresh = min(0.90, max(0.60, float(np.percentile(neg_c, 99.9)) + 0.15))
    if ncc_thresh <= float(neg_c.max()):
        print(f"  warning: random pairs reach {neg_c.max():.3f}, above threshold {ncc_thresh:.2f}; inspect the contact sheet")
    verified = {(a, b): ncc(T(a), T(b)) for a, b in cand}
    cand = {k for k, v in verified.items() if v >= ncc_thresh or k in exact_pairs}
    near = sorted(cand - exact_pairs)
    print(f"verified duplicate pairs: {len(cand)} (exact {len(exact_pairs)}, near {len(near)})  [NCC>={ncc_thresh:.2f}]")

    rows = []
    for a, b in sorted(cand):
        rows.append({
            "dataset_a": keys[a][0], "id_a": keys[a][1], "grade_a": int(df.diagnosis[a]),
            "dataset_b": keys[b][0], "id_b": keys[b][1], "grade_b": int(df.diagnosis[b]),
            "exact": (a, b) in exact_pairs,
            "phash_dist": int((hashes[a] != hashes[b]).sum()),
            "cosine": round(float(sim[a, b]), 4),
            "ncc": round(verified.get((a, b), 1.0), 4),
        })
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / f"pairs_{'_'.join(names)}.csv", index=False)

    cross = pairs[pairs.dataset_a != pairs.dataset_b]
    conflict = pairs[pairs.grade_a != pairs.grade_b]
    print(f"cross-dataset pairs: {len(cross)} | pairs with conflicting grades: {len(conflict)}")

    # contact sheet of near-duplicates for human review
    show = pairs[~pairs.exact].sort_values("ncc", ascending=False).head(16)
    if len(show):
        fig, axes = plt.subplots(len(show), 2, figsize=(7, 3.4 * len(show)))
        axes = np.atleast_2d(axes)
        for r, (_, p) in enumerate(show.iterrows()):
            for c, (dsn, idc, g) in enumerate([(p.dataset_a, p.id_a, p.grade_a), (p.dataset_b, p.id_b, p.grade_b)]):
                axes[r, c].imshow(P.crop_circle(P.read_rgb(raw_paths[(dsn, idc)])))
                axes[r, c].set_title(f"{dsn} {idc[:12]} grade {g}\nncc {p.ncc:.3f} phash {p.phash_dist}", fontsize=8)
                axes[r, c].axis("off")
        plt.tight_layout()
        fig.savefig(OUT / f"near_duplicates_{'_'.join(names)}.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"contact sheet: {OUT}/near_duplicates_{'_'.join(names)}.png")

    # union-find groups of everything flagged
    parent = list(range(len(keys)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in cand:
        parent[find(a)] = find(b)
    groups: dict[int, list] = {}
    for i in range(len(keys)):
        groups.setdefault(find(i), []).append(f"{keys[i][0]}:{keys[i][1]}")
    groups = [g for g in groups.values() if len(g) > 1]
    (OUT / f"groups_{'_'.join(names)}.json").write_text(json.dumps(groups, indent=1))
    print(f"duplicate groups (exact + near): {len(groups)}, photos involved: {sum(map(len, groups))}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["aptos"])
