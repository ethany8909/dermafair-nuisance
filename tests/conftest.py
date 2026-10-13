"""Shared fixtures. Every test runs on synthetic arrays; no patient image, label or prediction is read."""

from __future__ import annotations

import numpy as np
import pytest

from nuisance import base


def _base_available() -> bool:
    try:
        base.activate()
        return True
    except Exception:
        return False


BASE = _base_available()
needs_base = pytest.mark.skipif(not BASE, reason="needs DERMAFAIR_BASE, DERMAFAIR_DATA_DIR, DERMAFAIR_RESULTS_DIR")


def synthetic_image(seed: int = 0, size: int = 256) -> np.ndarray:
    """A skin-like synthetic photograph: graded skin tone, a red plaque, texture and a dark background corner."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    skin = np.stack([190 - 40 * yy, 140 - 35 * yy, 120 - 30 * xx], -1)
    plaque = np.exp(-(((xx - 0.55) ** 2 + (yy - 0.45) ** 2) / 0.02))[..., None] * np.array([45, -30, -20])
    texture = rng.normal(0, 6, (size, size, 1))
    img = skin + plaque + texture
    img[: size // 5, : size // 5] = [25, 22, 20]
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.fixture
def img() -> np.ndarray:
    return synthetic_image()
