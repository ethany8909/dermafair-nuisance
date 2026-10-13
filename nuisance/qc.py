"""The checks run before authorization (PREREGISTRATION.md section 4). None reads a diagnosis label or a
classifier output.

    Q1  identity: severity 0, through the harness and DermaFair's frozen encoder, reproduces the stored features
    Q2  colour preservation: how far each grid cell moves skin ITA, a* and b*; CP cells must stay within limits
    Q3  contact sheets for a visual check (local only, never committed)
    Q4  the unit-test suite (tests/)
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import base, spec
from . import transforms as T

CELLS = spec.cells()


def _cache(name: str = "internal_256.npy") -> np.ndarray:
    return np.load(base.input_path(name), mmap_mode="r")


# --------------------------------------------------------------------------- #
# Q1 identity
# --------------------------------------------------------------------------- #
def q1_identity(log=print) -> dict:
    from extract_features import dinov2, encode  # DermaFair, commit 89be54e

    imgs = _cache()
    stored = np.load(base.input_path("dinov2_b14_clean.npz"))["int"]
    ident = np.stack([T.cell(np.asarray(imgs[i]), "A", 0) for i in range(len(imgs))])  # severity 0 = identity
    log(f"Q1: encoding {len(ident)} internal images through the severity-0 path")
    feats = encode(dinov2("b"), ident)
    cos = (feats * stored).sum(1) / (np.linalg.norm(feats, axis=1) * np.linalg.norm(stored, axis=1))
    out = {
        "n_images": int(len(feats)),
        "min_cosine": float(cos.min()),
        "median_cosine": float(np.median(cos)),
        "max_abs_difference": float(np.abs(feats - stored).max()),
        "threshold_min_cosine": spec.Q1_MIN_COSINE,
    }
    out["pass"] = bool(out["min_cosine"] >= spec.Q1_MIN_COSINE)
    return out


# --------------------------------------------------------------------------- #
# Q2 colour preservation
# --------------------------------------------------------------------------- #
def skin_stats(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(median skin ITA, median a*, median b*) over the mask, with DermaFair's fallback to every pixel when the
    mask has 500 pixels or fewer (dermafair.data.skin_tone.ita)."""
    from dermafair.data.skin_tone import rgb2lab

    lab = rgb2lab(img).reshape(-1, 3)
    m = mask.reshape(-1)
    if m.sum() > 500:
        lab = lab[m]
    ita = np.degrees(np.arctan2(lab[:, 0] - 50, lab[:, 2]))
    return np.array([np.median(ita), np.median(lab[:, 1]), np.median(lab[:, 2])])


def q2_image(i: int) -> np.ndarray:
    """Changes (ITA, a*, b*) for every grid cell of internal image row i: (len(CELLS), 3).

    The DermaFair skin mask is computed once on the reference image and applied to both images, so the same
    pixels are compared. The reference is the clean image, except for zoom (family C), where it is the identically
    cropped clean image before resizing, so zoom measures resampling rather than content selection.
    """
    base.activate()
    from dermafair.data.skin_tone import skin_mask

    img = np.asarray(_cache()[i])
    h, w = img.shape[:2]
    mask = skin_mask(img) > 0.5
    ref = skin_stats(img, mask)
    out = np.zeros((len(CELLS), 3))
    for j, (fam, sev) in enumerate(CELLS):
        moved = T.cell(img, fam, sev, "internal", i)
        if fam == "C":
            top, left, side = T.zoom_box(T.param("C", sev), h, w)
            crop = img[top : top + side, left : left + side]
            crop_mask = skin_mask(crop) > 0.5
            up = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            out[j] = skin_stats(moved, np.asarray(up) > 127) - skin_stats(crop, crop_mask)
        else:
            out[j] = skin_stats(moved, mask) - ref
    return out


def q2_colour(log=print, workers: int | None = None) -> dict:
    tone = base.internal_tone()
    n = len(_cache())
    if len(tone) != n:
        raise RuntimeError(f"tone vector ({len(tone)}) does not match the cache ({n})")
    log(f"Q2: {len(CELLS)} cells x {n} internal images")
    with ProcessPoolExecutor(max_workers=workers, initializer=base.activate) as ex:
        delta = np.stack(list(ex.map(q2_image, range(n), chunksize=8)))  # (n, cells, 3)
    rows = []
    for j, (fam, sev) in enumerate(CELLS):
        d = delta[:, j, :]
        row = {
            "cell": spec.cell_name(fam, sev),
            "family": fam,
            "class": spec.FAMILIES[fam].cls,
            "severity": sev,
            "param": T.param(fam, sev),
            "abs_ita": float(np.median(np.abs(d[:, 0]))),
            "abs_a": float(np.median(np.abs(d[:, 1]))),
            "abs_b": float(np.median(np.abs(d[:, 2]))),
        }
        for g, code in (("lighter", 0), ("darker", 1)):
            m = tone == code
            row[f"ita_{g}"] = float(np.median(d[m, 0]))
            row[f"a_{g}"] = float(np.median(d[m, 1]))
            row[f"b_{g}"] = float(np.median(d[m, 2]))
        if row["class"] == "CP":
            row["pass"] = bool(
                row["abs_ita"] <= spec.Q2_MAX_ITA and row["abs_a"] <= spec.Q2_MAX_AB and row["abs_b"] <= spec.Q2_MAX_AB
            )
        rows.append(row)
    cp = [r for r in rows if r["class"] == "CP"]
    return {
        "n_images": int(n),
        "n_lighter": int((tone == 0).sum()),
        "n_darker": int((tone == 1).sum()),
        "limits": {"abs_ita": spec.Q2_MAX_ITA, "abs_a": spec.Q2_MAX_AB, "abs_b": spec.Q2_MAX_AB},
        "rows": rows,
        "cp_failures": [r["cell"] for r in cp if not r["pass"]],
        "pass": all(r["pass"] for r in cp),
    }


# --------------------------------------------------------------------------- #
# Q3 contact sheets (local only)
# --------------------------------------------------------------------------- #
def q3_sheets(out_dir: Path, log=print, thumb: int = 160) -> dict:
    """One sheet per family: rows = 3 lighter- and 3 darker-skin internal images (fixed draw), columns = clean and
    severities 1, 3, 5. Written under local/ (git-ignored); only this summary is committed."""
    imgs = _cache()
    tone = base.internal_tone()
    rng = np.random.default_rng([spec.BASE_SEED, spec.Q3_STREAM])
    pick = [
        int(i) for code in (0, 1) for i in rng.choice(np.flatnonzero(tone == code), spec.Q3_PER_GROUP, replace=False)
    ]
    small = lambda a: np.asarray(Image.fromarray(a).resize((thumb, thumb), Image.BICUBIC))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for fam in spec.ALL:
        rows = []
        for i in pick:
            img = np.asarray(imgs[i])
            cols = [small(img)] + [small(T.cell(img, fam, s, "internal", i)) for s in spec.Q3_SEVERITIES]
            rows.append(np.concatenate(cols, 1))
        name = f"Q3_{fam}_LOCAL_ONLY.jpg"
        Image.fromarray(np.concatenate(rows, 0)).save(out_dir / name, quality=90)
        files.append(name)
    log(f"Q3: {len(files)} contact sheets written to {out_dir}")
    return {"families": len(files), "images": len(pick), "severities": list(spec.Q3_SEVERITIES), "files": files}


# --------------------------------------------------------------------------- #
# Q4 unit tests
# --------------------------------------------------------------------------- #
def q4_tests(repo: Path) -> dict:
    """Run the suite (pyproject.toml adds -q) and keep pytest's final count line, e.g. '125 passed in 8s'."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"], cwd=repo, capture_output=True, text=True
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1] if proc.stdout.strip() else ""
    return {
        "command": "python -m pytest",
        "summary": tail,
        "returncode": proc.returncode,
        "pass": proc.returncode == 0 and "passed" in tail and "skipped" not in tail,
    }
