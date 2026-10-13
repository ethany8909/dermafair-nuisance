"""Q4: the stage-1 pipeline end to end on a synthetic cohort (synthetic images, a toy encoder, synthetic labels).

This exercises transforms -> features -> B0 and M1 per fold -> tau -> flips -> bootstrap -> verdict without
touching any real image, label or prediction.
"""

from __future__ import annotations

import numpy as np
from conftest import needs_base, synthetic_image

from nuisance import endpoints as E
from nuisance import spec, stages
from nuisance import transforms as T

pytestmark = needs_base


def toy_encoder(imgs: np.ndarray) -> np.ndarray:
    """Colour means, spreads and a coarse 4 x 4 luma map: cheap features that react to every family."""
    x = imgs.astype(float) / 255
    luma = x @ np.array([0.299, 0.587, 0.114])
    n, h, w = luma.shape
    coarse = luma.reshape(n, 4, h // 4, 4, w // 4).mean((2, 4)).reshape(n, -1)
    sharp = np.abs(np.diff(luma, axis=1)).mean((1, 2))[:, None]
    return np.hstack([x.mean((1, 2)), x.std((1, 2)), coarse, sharp])


def test_stage1_pipeline_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 120
    y = (rng.random(n) < 0.35).astype(int)
    imgs = np.stack([synthetic_image(seed=i, size=64) for i in range(n)]).astype(float)
    imgs[y == 1, 20:44, 20:44, 0] += 40  # psoriasis images get a redder plaque
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    folds, tone, subject = np.arange(n) % 5, (rng.random(n) < 0.4).astype(int), np.arange(n) // 2

    X = toy_encoder(imgs)
    cp = spec.cells(("CP",))
    names = [spec.cell_name(f, s) for f, s in cp]
    F = {
        spec.cell_name(f, s): toy_encoder(np.stack([T.cell(imgs[i], f, s, "internal", i) for i in range(n)]))
        for f, s in cp
    }
    V = {
        r: toy_encoder(
            np.stack(
                [
                    T.draw_view(imgs[i], T.view_rng(r, i, v), spec.M1_FAMILIES)[0]
                    for i in range(n)
                    for v in range(spec.K_VIEWS)
                ]
            )
        ).reshape(n, spec.K_VIEWS, -1)
        for r in spec.M1_DRAWS
    }
    b0, _ = stages.score_internal(X, y, folds, F)
    m1 = [stages.score_internal(X, y, folds, F, V[r])[0] for r in spec.M1_DRAWS]
    coh = E.Cohort(y, tone, subject)
    res = E.primary(coh, [b0], m1, names, E.bootstrap_indices(coh.clusters, 100, 0))
    assert res["verdict"] in ("Supported", "Not supported") or res["verdict"].startswith("Rejected: harm")
    assert set(res["fails"]) == {"R1", "R2", "R3", "R4"} and 0 <= res["B0"]["mfr_all"] <= 1
    assert len(b0.cells) == 25 and not np.isnan(b0.clean).any()
