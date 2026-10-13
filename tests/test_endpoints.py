"""Q4: endpoint arithmetic and the decision rule, on synthetic scores only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from nuisance import endpoints as E


def test_tau_is_the_largest_threshold_reaching_80_percent():
    rng = np.random.default_rng(0)
    for n_pos in (5, 40, 219):
        y = np.r_[np.ones(n_pos), np.zeros(300)].astype(int)
        s = np.round(rng.random(len(y)), 2)  # ties included
        tau = E.tau_at_sensitivity(s, y)
        sens = lambda t, s=s, y=y: (s[y == 1] >= t).mean()
        assert sens(tau) >= 0.8
        assert all(sens(t) < 0.8 for t in np.unique(s) if t > tau)


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    S = np.round(rng.random((500, 4)) + 0.3 * y[:, None], 1)
    assert np.allclose(E.auroc_columns(y, S), [roc_auc_score(y, S[:, j]) for j in range(4)])


def test_flip_matrix():
    r = E.Scored(np.array([0.2, 0.6, 0.9]), {"x": np.array([0.7, 0.4, 0.95])}, tau=0.5)
    assert r.flip_matrix(["x"])[:, 0].tolist() == [True, True, False]


def test_bootstrap_resamples_whole_clusters():
    clusters = np.repeat(np.arange(30), 3)
    boots = E.bootstrap_indices(clusters, n_boot=20, seed=5)
    assert len(boots) == 20
    for idx in boots:
        counts = np.bincount(idx, minlength=len(clusters)).reshape(30, 3)
        assert (counts == counts[:, :1]).all()  # members of a cluster appear together
    assert all(np.array_equal(a, b) for a, b in zip(boots, E.bootstrap_indices(clusters, n_boot=20, seed=5)))


def _cohort(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    tone = (rng.random(n) < 0.35).astype(int)
    clean = np.clip(0.35 * y + rng.normal(0.35, 0.18, n), 0, 1)
    return E.Cohort(y, tone, np.arange(n) // 2), clean, rng


def _scored(clean, y, noise, rng, names, darker_extra=None, tone=None):
    cells = {}
    for nm in names:
        sd = noise + (darker_extra * (tone == 1) if darker_extra is not None else 0)
        cells[nm] = np.clip(clean + rng.normal(0, 1, len(clean)) * sd, 0, 1)
    return E.Scored(clean, cells, E.tau_at_sensitivity(clean, y))


NAMES = [f"A{s}" for s in range(1, 6)]


def test_verdict_supported_when_m1_flips_less_at_equal_discrimination():
    coh, clean, rng = _cohort()
    b0 = _scored(clean, coh.y, 0.15, rng, NAMES)
    m1 = [_scored(clean, coh.y, 0.03, rng, NAMES) for _ in range(3)]
    res = E.primary(coh, [b0], m1, NAMES, E.bootstrap_indices(coh.clusters, 300, 1))
    assert res["G"] and res["verdict"] == "Supported", res["verdict"]


def test_verdict_not_supported_for_identical_methods():
    coh, clean, rng = _cohort()
    b0 = _scored(clean, coh.y, 0.1, rng, NAMES)
    res = E.primary(coh, [b0], [b0], NAMES, E.bootstrap_indices(coh.clusters, 300, 1))
    assert not res["G"] and res["verdict"] == "Not supported"


def test_r1_rejects_a_robust_but_less_discriminating_method():
    coh, clean, rng = _cohort()
    b0 = _scored(clean, coh.y, 0.15, rng, NAMES)
    worse = np.clip(0.5 * clean + rng.normal(0.25, 0.2, len(clean)), 0, 1)
    m1 = [_scored(worse, coh.y, 0.0, rng, NAMES)]
    res = E.primary(coh, [b0], m1, NAMES, E.bootstrap_indices(coh.clusters, 300, 1))
    assert res["fails"]["R1"] and res["verdict"].startswith("Rejected: harm")


def test_r4_rejects_more_flips_on_darker_skin():
    coh, clean, rng = _cohort()
    b0 = _scored(clean, coh.y, 0.05, rng, NAMES)
    m1 = [_scored(clean, coh.y, 0.0, rng, NAMES, darker_extra=0.3, tone=coh.tone)]
    res = E.primary(coh, [b0], m1, NAMES, E.bootstrap_indices(coh.clusters, 200, 1))
    assert res["fails"]["R4"] and "R4" in res["verdict"]


def test_cell_table_and_summary_shapes():
    coh, clean, rng = _cohort(n=200)
    b0 = _scored(clean, coh.y, 0.1, rng, NAMES)
    boot = E.bootstrap_indices(coh.clusters, 50, 2)
    rows = E.cell_table(coh, [b0], NAMES, boot)
    assert len(rows) == 3 * len(NAMES) and {"fr", "auc", "sens", "spec", "fr_lo", "fr_hi"} <= set(rows[0])
    summ = E.summary_table(coh, {"B0": [b0]}, {"CP": NAMES}, boot)
    assert len(summ) == 1 and 0 <= summ[0]["mfr_all"] <= 1
