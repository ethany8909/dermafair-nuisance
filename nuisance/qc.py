"""The checks run before authorization (PREREGISTRATION.md section 4). None reads a diagnosis label or a
classifier output.

    Q1  identity: severity 0, through the harness and DermaFair's frozen encoder, reproduces the stored features
    Q2  colour preservation by tone group (Addendum 1): grid cells and CP midpoints; admissible training intervals
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
# Q2 colour preservation (protocol section 4, specified by Addendum 1, section 2)
# --------------------------------------------------------------------------- #
# Checked points: the 55 grid cells ("s", severity) and the 4 midpoints of each CP family ("m", j for j..j+1).
POINTS = [(fam, "s", sev, T.param(fam, sev)) for fam, sev in CELLS] + [
    (fam, "m", j, spec.MIDPOINTS[fam][j - 1]) for fam in spec.CP for j in (1, 2, 3, 4)
]


def point_name(fam: str, kind: str, j: int) -> str:
    return spec.cell_name(fam, j) if kind == "s" else f"{fam}{j}-{j + 1}"


def adequate(mask: np.ndarray) -> bool:
    """Addendum 1, section 2.2: a reference mask with 500 pixels or fewer is inadequate."""
    return int(mask.sum()) > spec.MASK_MAX_INADEQUATE_PIXELS


def skin_stats(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(median ITA, median a*, median b*) over the masked pixels; ITA = atan2(L* - 50, b*) in degrees.

    No fallback to the whole image: callers pass adequate masks only."""
    from dermafair.data.skin_tone import rgb2lab

    lab = rgb2lab(img).reshape(-1, 3)[mask.reshape(-1)]
    ita = np.degrees(np.arctan2(lab[:, 0] - 50, lab[:, 2]))
    return np.array([np.median(ita), np.median(lab[:, 1]), np.median(lab[:, 2])])


def q2_image(i: int) -> np.ndarray:
    """(len(POINTS), 3) changes in (ITA, a*, b*) for internal image row i; NaN where the reference mask is inadequate.

    The DermaFair skin mask is computed once on the reference image and applied to both the reference and the
    transformed image, so the same pixels are compared. The reference is the clean image; for zoom it is the
    identically cropped clean image before resizing, whose mask is resized to the output by nearest neighbour.
    """
    base.activate()
    from dermafair.data.skin_tone import skin_mask

    img = np.asarray(_cache()[i])
    h, w = img.shape[:2]
    mask = skin_mask(img) > 0.5
    ref = skin_stats(img, mask) if adequate(mask) else None
    out = np.full((len(POINTS), 3), np.nan)
    for j, (fam, kind, k, value) in enumerate(POINTS):
        moved = T.cell(img, fam, k, "internal", i) if kind == "s" else T.midpoint(img, fam, k, "internal", i)
        if fam == "C":
            top, left, side = T.zoom_box(value, h, w)
            crop = img[top : top + side, left : left + side]
            crop_mask = skin_mask(crop) > 0.5
            if adequate(crop_mask):
                up = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
                out[j] = skin_stats(moved, np.asarray(up) > 127) - skin_stats(crop, crop_mask)
        elif ref is not None:
            out[j] = skin_stats(moved, mask) - ref
    return out


def summarize_point(d: np.ndarray, groups: dict) -> dict:
    """Per group: mask adequacy and the medians of |change| and of the signed change over adequate-mask images."""
    out = {}
    for g, m in groups.items():
        dd = d[m]
        ok = ~np.isnan(dd[:, 0])
        s = {"n": int(m.sum()), "n_adequate": int(ok.sum()), "share_adequate": float(ok.mean()) if m.any() else np.nan}
        for q, key in enumerate(("ita", "a", "b")):
            s[f"abs_{key}"] = float(np.median(np.abs(dd[ok, q]))) if ok.any() else np.nan
            s[key] = float(np.median(dd[ok, q])) if ok.any() else np.nan
        out[g] = s
    return out


def point_passes(summary: dict) -> bool:
    """Addendum 1, section 2.3: evaluable and within every limit in each tone group."""
    for g in spec.TONE_GROUPS:
        s = summary[g]
        if not s["share_adequate"] >= spec.MASK_MIN_ADEQUATE_SHARE:
            return False
        if not (s["abs_ita"] <= spec.Q2_MAX_ITA and s["abs_a"] <= spec.Q2_MAX_AB and s["abs_b"] <= spec.Q2_MAX_AB):
            return False
    return True


def admissibility(passes: dict) -> dict:
    """Addendum 1, section 1: the passing prefix of each CP family.

    passes[(family, "s", j)] is the result for severity j, passes[(family, "m", j)] for the midpoint j..j+1."""
    families, surviving, reclassified = {}, [], []
    for fam in spec.CP:
        k = 0
        for j in spec.SEVERITIES:
            if not passes[(fam, "s", j)] or (j > 1 and not passes[(fam, "m", j - 1)]):
                break
            k = j
        levels = spec.FAMILIES[fam].levels
        families[fam] = {"k": k, "interval": sorted((levels[0], levels[k - 1])) if k else None}
        surviving += [spec.cell_name(fam, s) for s in spec.SEVERITIES if s <= k]
        reclassified += [spec.cell_name(fam, s) for s in spec.SEVERITIES if s > k]
    return {
        "families": families,
        "m1_families": [f for f in spec.CP if families[f]["k"] > 0],
        "surviving_cp": surviving,
        "reclassified_to_ca": reclassified,
        "n_surviving": len(surviving),
        "stage1_eligible": len(surviving) >= spec.CP_MIN_SURVIVING,
    }


def q2_colour(log=print, workers: int | None = None) -> dict:
    tone = base.internal_tone()
    n = len(_cache())
    if len(tone) != n:
        raise RuntimeError(f"tone vector ({len(tone)}) does not match the cache ({n})")
    log(f"Q2: {len(POINTS)} checked points x {n} internal images")
    with ProcessPoolExecutor(max_workers=workers, initializer=base.activate) as ex:
        delta = np.stack(list(ex.map(q2_image, range(n), chunksize=8)))  # (n, points, 3)
    groups = {"all": np.ones(n, bool), **{g: tone == code for g, code in spec.TONE_GROUPS.items()}}
    rows, passes = [], {}
    for j, (fam, kind, k, value) in enumerate(POINTS):
        summary = summarize_point(delta[:, j, :], groups)
        row = {
            "point": point_name(fam, kind, k),
            "family": fam,
            "kind": "cell" if kind == "s" else "midpoint",
            "class": spec.FAMILIES[fam].cls,
            "param": value,
            **summary,
        }
        if row["class"] == "CP":
            row["pass"] = point_passes(summary)
            passes[(fam, kind, k)] = row["pass"]
        rows.append(row)
    adm = admissibility(passes)
    return {
        "n_images": int(n),
        "n_lighter": int((tone == 0).sum()),
        "n_darker": int((tone == 1).sum()),
        "n_unknown_tone": int((tone == -1).sum()),
        "limits": {
            "abs_ita": spec.Q2_MAX_ITA,
            "abs_a": spec.Q2_MAX_AB,
            "abs_b": spec.Q2_MAX_AB,
            "inadequate_mask_pixels_at_most": spec.MASK_MAX_INADEQUATE_PIXELS,
            "min_adequate_share": spec.MASK_MIN_ADEQUATE_SHARE,
        },
        "rows": rows,
        "admissible": adm,
        "cp_failures": [r["point"] for r in rows if r.get("pass") is False],
        "pass": all(r["pass"] for r in rows if "pass" in r),
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
