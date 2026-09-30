"""Fundus preprocessing. v1 crop + enhance, v2 crop + pad, v3 (final) v2 + common outline, v4 v3 + enhance."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 256


def crop_circle(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the black border so the retina fills the frame; returns the input if the photo is almost all dark."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = grey > tol
    if not mask.any():
        return img

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    cropped = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    # a sliver means the threshold caught noise, not retina
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return img
    return cropped


def resize(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize to a fixed square (INTER_AREA is best for shrinking)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def ben_graham(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Ben Graham blur-subtraction: removes uneven illumination and sharpens edges."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def clahe_green(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the green channel, where vessels and lesions have the most contrast."""
    out = img.copy()
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    out[:, :, 1] = clahe.apply(out[:, :, 1])
    return out


def denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Small median blur to remove single-pixel sensor noise."""
    return cv2.medianBlur(img, ksize)


def preprocess(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """v1 pipeline: crop, resize, Ben Graham, CLAHE green, denoise."""
    img = crop_circle(img)
    img = resize(img, size)
    img = ben_graham(img)
    img = clahe_green(img)
    img = denoise(img)
    return img


def pad_square(img: np.ndarray) -> np.ndarray:
    """Centre the image on a black square canvas so its proportions are kept."""
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def preprocess_v2(img: np.ndarray, size: int = 512) -> np.ndarray:
    """v2 pipeline: crop border, pad square, resize; no enhancement."""
    return cv2.resize(pad_square(crop_circle(img)), (size, size), interpolation=cv2.INTER_AREA)


FOV_BAND = 0.70


def common_band(img: np.ndarray, ratio: float = FOV_BAND) -> np.ndarray:
    """Black out the top and bottom so every photo has the same flat-cut outline (height = 0.70 x width)."""
    out = img.copy()
    h, w = out.shape[:2]
    keep = int(round(ratio * w))
    if h > keep:
        top = (h - keep) // 2
        out[:top] = 0
        out[top + keep:] = 0
    return out


def circle_mask(img: np.ndarray) -> np.ndarray:
    """Black out the corners outside the circle inscribed in the photo width."""
    out = img.copy()
    h, w = out.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    outside = (xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2 > (w / 2) ** 2
    out[outside] = 0
    return out


def preprocess_v3(img: np.ndarray, size: int = 512) -> np.ndarray:
    """v3 pipeline: crop, common band, pad square, resize."""
    return cv2.resize(pad_square(common_band(crop_circle(img))), (size, size), interpolation=cv2.INTER_AREA)


def preprocess_v4(img: np.ndarray, size: int = 512) -> np.ndarray:
    """v4 pipeline: v3 geometry plus inscribed circle, Ben Graham (sigma = radius/30), CLAHE green and a 5 % rim re-mask."""
    base = cv2.resize(pad_square(common_band(circle_mask(crop_circle(img)))), (size, size),
                      interpolation=cv2.INTER_AREA)
    fov = (cv2.cvtColor(base, cv2.COLOR_RGB2GRAY) > 7).astype(np.uint8)
    radius = size / 2  # retina spans the full width after pad + resize
    enhanced = clahe_green(ben_graham(base, sigma=radius / 30))
    k = max(1, int(0.05 * size))
    # borderValue=0 so the rim is also cleaned where the retina touches the image edge
    inner = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    enhanced[inner == 0] = 0
    return enhanced


def steps(img: np.ndarray, size: int = IMG_SIZE) -> dict[str, np.ndarray]:
    """Return the image after each v1 step, for the before/after figure."""
    out = {"original": img}
    out["1 cropped"] = crop_circle(img)
    out["2 resized"] = resize(out["1 cropped"], size)
    out["3 ben graham"] = ben_graham(out["2 resized"])
    out["4 clahe green"] = clahe_green(out["3 ben graham"])
    out["5 denoised"] = denoise(out["4 clahe green"])
    return out


def read_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, img: np.ndarray) -> None:
    """Save an RGB array as PNG."""
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def cache_dataset(
    src_dir: str | Path,
    dst_dir: str | Path,
    ids: list[str] | None = None,
    ext: str = ".png",
    size: int = IMG_SIZE,
    overwrite: bool = False,
    fn=preprocess,
) -> dict[str, int]:
    """Run `fn` over a folder once and save PNGs; skips existing files unless overwrite. Returns {done, skipped, failed}."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if ids is None:
        paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    else:
        # match by stem: datasets mix .png/.JPG/.jpg in one folder
        by_stem = {p.stem: p for p in src_dir.iterdir() if p.is_file()}
        paths = [by_stem.get(str(i), src_dir / f"{i}{ext}") for i in ids]

    counts = {"done": 0, "skipped": 0, "failed": 0}
    for path in paths:
        out_path = dst_dir / f"{path.stem}.png"
        if out_path.exists() and not overwrite:
            counts["skipped"] += 1
            continue
        try:
            write_rgb(out_path, fn(read_rgb(path), size))
            counts["done"] += 1
        except Exception as exc:  # one corrupt file must not stop the run
            print(f"failed on {path.name}: {exc}")
            counts["failed"] += 1
    return counts


if __name__ == "__main__":
    # self-check on a synthetic fundus: bright circle on black
    fake = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(fake, (300, 200), 180, (180, 90, 60), -1)

    cropped = crop_circle(fake)
    assert cropped.shape[0] < fake.shape[0], "crop_circle did not remove the black border"
    assert abs(cropped.shape[0] - cropped.shape[1]) <= 2, "crop should be roughly square"

    out = preprocess(fake)
    assert out.shape == (IMG_SIZE, IMG_SIZE, 3), f"unexpected shape {out.shape}"
    assert out.dtype == np.uint8, f"unexpected dtype {out.dtype}"

    every = steps(fake)
    assert len(every) == 6, "steps() should return the original plus five stages"

    # v2: wide ellipse -> centred square
    wide = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.ellipse(wide, (300, 150), (280, 140), 0, 0, 360, (180, 90, 60), -1)
    sq = pad_square(crop_circle(wide))
    assert sq.shape[0] == sq.shape[1], "pad_square must return a square"
    v2 = preprocess_v2(wide, 512)
    assert v2.shape == (512, 512, 3) and v2.dtype == np.uint8
    # centre bright, top rows black
    assert v2[256, 256].sum() > 100 and v2[5, 256].sum() == 0

    # v3: full circle and flat-cut versions end with the same outline
    circle = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(circle, (250, 250), 240, (170, 95, 65), -1)
    flat = circle.copy(); flat[:70] = 0; flat[430:] = 0  # natural flat cut, h/w ~0.74
    outline = lambda im: (cv2.cvtColor(preprocess_v3(im, 256), cv2.COLOR_RGB2GRAY) > 7)
    diff = (outline(circle) != outline(flat)).mean()
    assert diff < 0.01, f"outlines should match after the common band, differ on {diff:.1%}"
    very_flat = circle.copy(); very_flat[:110] = 0; very_flat[390:] = 0  # flatter than the band
    kept = common_band(crop_circle(very_flat))
    assert kept.shape[0] < 0.70 * kept.shape[1] and kept.any(), "photos flatter than the band stay untouched"

    # v4: same outline, rim cleaned, interior enhanced
    v4c, v4f = preprocess_v4(circle, 256), preprocess_v4(flat, 256)
    assert v4c.shape == (256, 256, 3) and v4c.dtype == np.uint8
    out4 = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) > 0
    assert (out4(v4c) != out4(v4f)).mean() < 0.01, "v4 outlines must match"
    assert v4c[128, 5].sum() == 0, "outside the field of view must be black"
    assert not np.array_equal(v4c[128, 128], preprocess_v3(circle, 256)[128, 128]), "v4 must enhance the interior"
    framefill = np.full((300, 400, 3), (170, 95, 65), dtype=np.uint8)  # retina fills the frame
    assert (out4(preprocess_v4(framefill, 256)) != out4(v4c)).mean() < 0.02, "frame-filling photos must match too"
    assert np.array_equal(circle_mask(crop_circle(circle)), crop_circle(circle)) or \
        (circle_mask(crop_circle(circle)) != crop_circle(circle)).any(axis=2).mean() < 0.01, "full circles barely change"

    # all-black image must not crash
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    assert preprocess(black).shape == (IMG_SIZE, IMG_SIZE, 3)

    # cache_dataset must find files whatever the extension
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src", Path(tmp) / "dst"
        src.mkdir()
        write_rgb(src / "a.png", fake)
        cv2.imwrite(str(src / "b.JPG"), cv2.cvtColor(fake, cv2.COLOR_RGB2BGR))
        result = cache_dataset(src, dst, ids=["a", "b"], ext=".png")
        assert result == {"done": 2, "skipped": 0, "failed": 0}, result
        assert (dst / "a.png").exists() and (dst / "b.png").exists()

    print("preprocess.py self-check passed")
