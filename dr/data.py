"""Splits (stratified 70/15/15), datasets, augmentation and class balancing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from . import CLASSES

# ImageNet normalisation, as used to pretrain the backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_splits(
    csv_path: str | Path,
    out_dir: str | Path,
    id_col: str = "id_code",
    label_col: str = "diagnosis",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split of a label CSV, saved once; existing CSVs are reused unless overwrite."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / f"{name}.csv" for name in ("train", "val", "test")}

    if all(p.exists() for p in paths.values()) and not overwrite:
        return {name: pd.read_csv(p) for name, p in paths.items()}

    df = pd.read_csv(csv_path)[[id_col, label_col]].rename(
        columns={id_col: "id_code", label_col: "diagnosis"}
    )

    # test first, then val from the remainder
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["diagnosis"], random_state=seed
    )
    val_share = val_frac / (1.0 - test_frac)  # val as a fraction of what is left
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=train_val["diagnosis"], random_state=seed
    )

    splits = {"train": train, "val": val, "test": test}
    for name, frame in splits.items():
        frame.reset_index(drop=True).to_csv(paths[name], index=False)
    return splits


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts per grade per split."""
    rows = {
        name: frame["diagnosis"].value_counts().reindex(range(len(CLASSES)), fill_value=0)
        for name, frame in splits.items()
    }
    table = pd.DataFrame(rows)
    table.index = [f"{i} {name}" for i, name in enumerate(CLASSES)]
    table.loc["total"] = table.sum()
    return table


def train_tfms(img_size: int = 224) -> transforms.Compose:
    """v1 training augmentation: flips, rotation, mild zoom, light colour jitter."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FundusArtefacts:
    """Random camera artefacts (halo, dark patch, bright spots), each with probability p; after GDRNet FundusAug."""

    def __init__(self, p: float = 0.5):
        self.p = p

    @staticmethod
    def _u(lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi))

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img).astype(np.float32)
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        r_norm = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (min(h, w) / 2)

        if self._u(0, 1) < self.p:  # halo: bright ring
            ring = np.exp(-((r_norm - self._u(0.7, 0.95)) ** 2) / (2 * 0.08 ** 2))
            arr += ring[..., None] * self._u(20, 80)
        if self._u(0, 1) < self.p:  # hole: soft dark disc
            cy, cx, rad = self._u(0.3, 0.7) * h, self._u(0.3, 0.7) * w, self._u(0.2, 0.45) * min(h, w)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            arr -= np.clip(1 - d / rad, 0, 1)[..., None] * self._u(10, 40)
        if self._u(0, 1) < self.p:  # spots: small bright blobs
            for _ in range(int(self._u(5, 11))):
                cy, cx, rad = self._u(0.1, 0.9) * h, self._u(0.1, 0.9) * w, self._u(0.01, 0.05) * min(h, w)
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                arr += np.clip(1 - d / rad, 0, 1)[..., None] * self._u(30, 90)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def train_tfms_v2(img_size: int = 448) -> transforms.Compose:
    """v2 training augmentation: v1 geometry plus strong colour jitter, camera artefacts, blur and sharpness."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            FundusArtefacts(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.1, 3.0))], p=0.5),
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_tfms(img_size: int = 224) -> transforms.Compose:
    """Validation/test transform: resize and normalise only."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FundusDataset(Dataset):
    """Yields (image tensor, label) from a split CSV and the cached preprocessed PNGs."""

    def __init__(
        self,
        split_csv: str | Path,
        img_dir: str | Path,
        tfms: transforms.Compose,
        id_col: str = "id_code",
        label_col: str = "diagnosis",
    ):
        self.df = pd.read_csv(split_csv)
        self.img_dir = Path(img_dir)
        self.tfms = tfms
        self.ids = self.df[id_col].astype(str).tolist()
        self.labels = self.df[label_col].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.img_dir / f"{self.ids[idx]}.png").convert("RGB")
        return self.tfms(img), self.labels[idx]


def class_counts(labels: list[int] | np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Number of images per grade."""
    return np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)


def class_weights(labels: list[int] | np.ndarray, n_classes: int = 5) -> torch.Tensor:
    """Balanced loss weights: total / (n_classes * count), for CrossEntropyLoss(weight=...)."""
    counts = class_counts(labels, n_classes).astype(float)
    counts[counts == 0] = 1.0  # avoid divide-by-zero on an absent class
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sampler(labels: list[int], n_classes: int = 5) -> WeightedRandomSampler:
    """Sampler that draws each grade about equally often (probability 1/class count)."""
    counts = class_counts(labels, n_classes).astype(float)
    counts[counts == 0] = 1.0
    per_image = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(per_image, num_samples=len(labels), replacement=True)


def domain_class_sampler(df: pd.DataFrame, beta: float = 0.5) -> WeightedRandomSampler:
    """Sampler that softly balances (dataset, grade) pairs with weight 1/count^beta (GDRNet DCR)."""
    domain = df["domain"] if "domain" in df else pd.Series("single", index=df.index)
    counts = df.groupby([domain, df["diagnosis"]]).size()
    weights = [1.0 / counts[(d, g)] ** beta for d, g in zip(domain, df["diagnosis"])]
    return WeightedRandomSampler(weights, num_samples=len(df), replacement=True)


def make_loaders(
    splits_dir: str | Path,
    img_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    balance: str = "sampler",
    pipeline: str = "v1",
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders; balance is "sampler", "domain_class" or "none"; pipeline "v1" or "v2"."""
    splits_dir, img_dir = Path(splits_dir), Path(img_dir)
    aug = train_tfms_v2(img_size) if pipeline == "v2" else train_tfms(img_size)
    sets = {
        "train": FundusDataset(splits_dir / "train.csv", img_dir, aug),
        "val": FundusDataset(splits_dir / "val.csv", img_dir, eval_tfms(img_size)),
        "test": FundusDataset(splits_dir / "test.csv", img_dir, eval_tfms(img_size)),
    }

    if balance == "sampler":
        train_loader = DataLoader(
            sets["train"],
            batch_size=batch_size,
            sampler=balanced_sampler(sets["train"].labels),
            num_workers=num_workers,
        )
    elif balance == "domain_class":
        train_loader = DataLoader(
            sets["train"],
            batch_size=batch_size,
            sampler=domain_class_sampler(sets["train"].df),
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            sets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    return {
        "train": train_loader,
        "val": DataLoader(sets["val"], batch_size=batch_size, num_workers=num_workers),
        "test": DataLoader(sets["test"], batch_size=batch_size, num_workers=num_workers),
    }


if __name__ == "__main__":
    # self-check: disjoint stratified splits, balancing favours the rare class
    import tempfile

    rng = np.random.default_rng(0)
    fake = pd.DataFrame(
        {
            "id_code": [f"img{i}" for i in range(1000)],
            # lopsided like the real data
            "diagnosis": rng.choice([0, 1, 2, 3, 4], size=1000, p=[0.5, 0.1, 0.27, 0.05, 0.08]),
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "train.csv"
        fake.to_csv(csv, index=False)
        splits = make_splits(csv, Path(tmp) / "splits")

        ids = {name: set(frame["id_code"]) for name, frame in splits.items()}
        assert not ids["train"] & ids["val"], "train and val overlap"
        assert not ids["train"] & ids["test"], "train and test overlap"
        assert not ids["val"] & ids["test"], "val and test overlap"
        assert sum(len(s) for s in ids.values()) == len(fake), "images lost in the split"

        # each split keeps the overall class mix
        overall = fake["diagnosis"].value_counts(normalize=True).sort_index()
        for name, frame in splits.items():
            share = frame["diagnosis"].value_counts(normalize=True).sort_index()
            assert np.allclose(share, overall, atol=0.03), f"{name} split is not stratified"

        # re-running returns the same split
        again = make_splits(csv, Path(tmp) / "splits")
        assert set(again["test"]["id_code"]) == ids["test"], "splits changed on re-run"

    labels = fake["diagnosis"].tolist()
    weights = class_weights(labels)
    counts = class_counts(labels)
    assert weights[int(np.argmin(counts))] > weights[int(np.argmax(counts))], (
        "the rarest class must carry the largest weight"
    )

    sampler = balanced_sampler(labels)
    drawn = np.bincount([labels[i] for i in sampler], minlength=5)
    assert drawn.min() > counts.min(), "sampler did not oversample the rare class"
    assert drawn.max() < counts.max(), "sampler did not thin the common class"

    # v2 augmentation keeps the tensor shape
    pil = Image.fromarray(np.full((512, 512, 3), 120, dtype=np.uint8))
    out = train_tfms_v2(448)(pil)
    assert out.shape == (3, 448, 448), out.shape
    art = FundusArtefacts(p=1.0)(pil)
    assert art.size == (512, 512) and np.asarray(art).std() > 0, "artefacts should change the image"

    # rare (domain, grade) pair is drawn more than its share
    mixed = pd.DataFrame({
        "diagnosis": [0] * 900 + [4] * 100 + [0] * 50 + [4] * 5,
        "domain": ["big"] * 1000 + ["small"] * 55,
    })
    drawn = pd.Series([i for i in domain_class_sampler(mixed)])
    share_small4 = ((mixed.domain[drawn] == "small") & (mixed.diagnosis[drawn] == 4)).mean()
    natural = 5 / len(mixed)  # 0.5 % of the data
    assert share_small4 > 5 * natural, f"rare pair drawn {share_small4:.3f}, natural share {natural:.3f}"
    assert share_small4 < 0.25, "soft balancing (beta=0.5) must not swamp the batch with the rare pair"

    print("data.py self-check passed")
