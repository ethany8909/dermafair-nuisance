"""The 11 frozen nuisance-transform families (PREREGISTRATION.md section 4).

Every transform takes and returns an H x W x 3 RGB uint8 array (256 x 256 in the study).

    cell(img, family, severity, cohort, i)   one grid cell; severity 0 is the identity
    draw_view(img, rng, families)            one training view for M1 or C1

Random numbers: family E draws its noise from `noise_rng(cohort, i, severity)` on the grid, and from the view's
own generator during training (after the family and the parameter have been drawn).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

from . import spec

LUMA = np.array([0.2126, 0.7152, 0.0722])  # luminance of linear sRGB (Rec. 709 primaries)


def _pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# colour-preserving (CP)
# --------------------------------------------------------------------------- #
def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """A: re-encode with Pillow's JPEG encoder, 4:2:0 chroma subsampling, no optimisation."""
    buf = io.BytesIO()
    _pil(img).save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def resolution(img: np.ndarray, s: int) -> np.ndarray:
    """B: bicubic downscale to s x s (Pillow antialiases when shrinking), bicubic upscale back."""
    h, w = img.shape[:2]
    small = _pil(img).resize((int(s), int(s)), Image.BICUBIC)
    return np.asarray(small.resize((w, h), Image.BICUBIC), dtype=np.uint8)


def zoom_box(f: float, h: int, w: int) -> tuple[int, int, int]:
    """(top, left, side) of the centred square crop with side f * min(h, w)."""
    side = int(round(float(f) * min(h, w)))
    return (h - side) // 2, (w - side) // 2, side


def zoom(img: np.ndarray, f: float) -> np.ndarray:
    """C: centred square crop of side f * 256, bicubic resize back to the full size."""
    h, w = img.shape[:2]
    top, left, side = zoom_box(f, h, w)
    crop = _pil(img[top : top + side, left : left + side])
    return np.asarray(crop.resize((w, h), Image.BICUBIC), dtype=np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """D: Pillow GaussianBlur with radius = sigma (the standard deviation, in pixels)."""
    return np.asarray(_pil(img).filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.uint8)


def luma_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """E: one N(0, sigma^2) draw per pixel, added equally to R, G and B (luma changes, chroma does not)."""
    n = rng.normal(0.0, float(sigma), size=img.shape[:2])
    return np.clip(np.rint(img.astype(np.float64) + n[..., None]), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# linear light (IEC 61966-2-1)
# --------------------------------------------------------------------------- #
def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    s = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# lightness-altering (LA) and chromaticity-altering (CA)
# --------------------------------------------------------------------------- #
def exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """F-/F+: linear-light gain 2^EV, clipped at black and white."""
    return linear_to_srgb(srgb_to_linear(img) * 2.0 ** float(ev))


def cast_gains(c: float) -> np.ndarray:
    """Linear-light channel gains (2^(c/2), 1, 2^(-c/2)), divided by their effect on grey luminance."""
    g = np.array([2.0 ** (c / 2.0), 1.0, 2.0 ** (-c / 2.0)])
    return g / float(LUMA @ g)


def cast(img: np.ndarray, c: float) -> np.ndarray:
    """G-: warm cast with c > 0; G+: cool cast, applied as -c."""
    return linear_to_srgb(srgb_to_linear(img) * cast_gains(float(c)))


def saturation(img: np.ndarray, k: float) -> np.ndarray:
    """H-/H+: CIELAB a* and b* scaled by k with L* fixed, using DermaFair's CIELAB (D65) conversion.

    Out-of-gamut colours are clipped by that conversion.
    """
    from dermafair.data.skin_tone import lab2rgb, rgb2lab  # the frozen DermaFair code (base.activate())

    lab = rgb2lab(img)
    lab[..., 1:] *= float(k)
    return lab2rgb(lab)


# --------------------------------------------------------------------------- #
# grid cells and training views
# --------------------------------------------------------------------------- #
def apply(img: np.ndarray, family: str, param: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply one family at a given parameter value (signed EV for F, c >= 0 for G, k for H)."""
    if family == "A":
        return jpeg(img, param)
    if family == "B":
        return resolution(img, param)
    if family == "C":
        return zoom(img, param)
    if family == "D":
        return blur(img, param)
    if family == "E":
        if rng is None:
            raise ValueError("family E needs a random generator")
        return luma_noise(img, param, rng)
    if family in ("F-", "F+"):
        return exposure(img, param)
    if family == "G-":
        return cast(img, param)
    if family == "G+":
        return cast(img, -param)
    if family in ("H-", "H+"):
        return saturation(img, param)
    raise KeyError(f"unknown family {family!r}")


def param(family: str, severity: int) -> float:
    return spec.FAMILIES[family].levels[severity - 1]


def noise_rng(cohort: str, i: int, severity: int) -> np.random.Generator:
    """Generator for the evaluation noise of family E (section 8)."""
    return np.random.default_rng([spec.BASE_SEED, spec.NOISE_STREAM, spec.COHORT_IDS[cohort], int(i), int(severity)])


def cell(img: np.ndarray, family: str, severity: int, cohort: str = "internal", i: int = 0) -> np.ndarray:
    """Grid cell (family, severity) for image row i of a cohort's cache. Severity 0 returns an unchanged copy."""
    if severity == 0:
        return img.copy()
    rng = noise_rng(cohort, i, severity) if family == "E" else None
    return apply(img, family, param(family, severity), rng)


def view_rng(draw: int, i: int, v: int) -> np.random.Generator:
    """Generator for training view v of internal image row i in draw r (section 8)."""
    return np.random.default_rng([spec.BASE_SEED, int(draw), int(i), int(v)])


def draw_view(img: np.ndarray, rng: np.random.Generator, families: tuple[str, ...]) -> tuple[np.ndarray, str, float]:
    """One training view: family uniform over `families`, parameter uniform over its training range.

    The generator is consumed in a fixed order: the family, then the parameter, then (family E only) the noise.
    """
    fid = families[int(rng.integers(len(families)))]
    fam = spec.FAMILIES[fid]
    lo, hi = fam.train
    p = int(rng.integers(lo, hi + 1)) if fam.integer else float(rng.uniform(lo, hi))
    return apply(img, fid, p, rng), fid, p
