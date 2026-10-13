"""Endpoints, intervals and the decision rule (PREREGISTRATION.md sections 6-7).

A fitted method is a list of `Scored` replicates (one for B0 and C1, one per view draw for M1). Every endpoint is
computed per replicate and then averaged over replicates; predictions are never ensembled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from . import spec

GROUPS = ("all", "lighter", "darker")


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def tau_at_sensitivity(scores: np.ndarray, y: np.ndarray, target: float = spec.SENSITIVITY) -> float:
    """The largest threshold t at which the share of psoriasis images with score >= t is still >= target."""
    pos = np.sort(np.asarray(scores, dtype=float)[np.asarray(y) == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("no psoriasis images to set the threshold on")
    k = max(1, int(np.ceil(target * len(pos) - 1e-9)))
    return float(pos[k - 1])


def auroc_columns(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUROC of every column of `scores` (n x m) against labels y (Mann-Whitney, ties at half weight)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(scores, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(s.shape[1], np.nan)
    ranks = rankdata(s, axis=0)
    return (ranks[y].sum(0) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(auroc_columns(y, scores)[0])


def group_mask(tone: np.ndarray, group: str) -> np.ndarray:
    if group == "all":
        return np.ones(len(tone), bool)
    return tone == (0 if group == "lighter" else 1)


def bootstrap_indices(clusters: np.ndarray, n_boot: int = spec.N_BOOT, seed: int = spec.BASE_SEED) -> list[np.ndarray]:
    """Index arrays of a paired bootstrap that resamples clusters (patients, cases or near-duplicate groups)."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    return [np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))]) for _ in range(n_boot)]


def ci(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


# --------------------------------------------------------------------------- #
# one model's scores
# --------------------------------------------------------------------------- #
@dataclass
class Scored:
    """One fitted replicate's scores on one cohort: clean images and every scored grid cell."""

    clean: np.ndarray  # (n,)
    cells: dict  # cell name -> (n,) scores on the perturbed images
    tau: float

    def decisions(self, scores: np.ndarray) -> np.ndarray:
        return scores >= self.tau

    def flip_matrix(self, names: list[str]) -> np.ndarray:
        """(n, len(names)) booleans: the decision at tau differs between clean and perturbed."""
        c = self.decisions(self.clean)
        return np.stack([self.decisions(self.cells[nm]) != c for nm in names], 1)


@dataclass
class Cohort:
    """The scored images of one cohort."""

    y: np.ndarray  # 1 = psoriasis, 0 = eczema
    tone: np.ndarray  # 0 lighter, 1 darker, -1 unknown
    clusters: np.ndarray  # bootstrap unit of each image

    def __post_init__(self):
        self.y = np.asarray(self.y).astype(int)
        self.tone = np.asarray(self.tone).astype(int)
        self.clusters = np.asarray(self.clusters)


def method_endpoints(coh: Cohort, reps: list[Scored], flips: list[np.ndarray], idx: np.ndarray) -> dict:
    """Seed-averaged clean AUROC, gap and mFR (over the columns of `flips`) on the images in idx."""
    keys = [f"auc_{g}" for g in GROUPS] + ["gap"] + [f"mfr_{g}" for g in GROUPS]
    acc = {k: [] for k in keys}
    y, tone = coh.y[idx], coh.tone[idx]
    for rep, fm in zip(reps, flips):
        s = rep.clean[idx]
        per_image = fm[idx].mean(1)
        for g in GROUPS:
            m = group_mask(tone, g)
            acc[f"auc_{g}"].append(auroc(y[m], s[m]))
            acc[f"mfr_{g}"].append(float(per_image[m].mean()) if m.any() else np.nan)
        acc["gap"].append(acc["auc_lighter"][-1] - acc["auc_darker"][-1])
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# the primary comparison and the verdict (section 6)
# --------------------------------------------------------------------------- #
def r1_category(lo: float, hi: float) -> str:
    """Addendum 1, section 3: how a 95% CI for the clean-AUROC change relates to the R1 gate (unchanged)."""
    if lo >= spec.R1_LOWER_BOUND:
        return "passes"
    if hi >= 0:
        return "R1: non-inferiority not shown (interval too wide)"
    if hi >= spec.R1_LOWER_BOUND:
        return "R1: clean-performance loss demonstrated"
    return "R1: material clean-performance loss demonstrated"


def failure_reasons(fails: dict, cis: dict) -> dict:
    """Addendum 1, section 3: wording for each failing rule. 'Loss' is said only when its CI excludes zero."""
    out = {}
    if fails["R1"]:
        out["R1"] = r1_category(*cis["d_auc_all"])
    if fails["R2"]:
        demonstrated = cis["d_auc_darker"][1] < 0
        out["R2"] = (
            "R2: darker-skin loss demonstrated" if demonstrated else "R2: darker-skin guard triggered (point estimate)"
        )
    if fails["R3"]:
        out["R3"] = "R3: tone-gap guard triggered (point estimate)"
    if fails["R4"]:
        out["R4"] = "R4: darker-skin robustness guard triggered (point estimate)"
    return out


def primary(coh: Cohort, b0: list[Scored], m1: list[Scored], cp_names: list[str], boot: list[np.ndarray]) -> dict:
    """Invariance gain G, failure rules R1-R4 and the verdict for M1 against B0."""
    fb = [r.flip_matrix(cp_names) for r in b0]
    fm = [r.flip_matrix(cp_names) for r in m1]
    every = np.arange(len(coh.y))
    pb, pm = method_endpoints(coh, b0, fb, every), method_endpoints(coh, m1, fm, every)

    def deltas(eb, em):
        return {
            "d_mfr": eb["mfr_all"] - em["mfr_all"],  # positive: M1 flips less often
            "d_auc_all": em["auc_all"] - eb["auc_all"],
            "d_auc_darker": em["auc_darker"] - eb["auc_darker"],
            "d_auc_lighter": em["auc_lighter"] - eb["auc_lighter"],
            "d_gap": em["gap"] - eb["gap"],
            "d_mfr_darker": eb["mfr_darker"] - em["mfr_darker"],
            "d_mfr_lighter": eb["mfr_lighter"] - em["mfr_lighter"],
        }

    point = deltas(pb, pm)
    reps = {k: [] for k in point}
    eb_reps = {k: [] for k in pb}
    em_reps = {k: [] for k in pm}
    for idx in boot:
        eb, em = method_endpoints(coh, b0, fb, idx), method_endpoints(coh, m1, fm, idx)
        for k, v in deltas(eb, em).items():
            reps[k].append(v)
        for k in pb:
            eb_reps[k].append(eb[k])
            em_reps[k].append(em[k])
    cis = {k: ci(v) for k, v in reps.items()}

    rel = point["d_mfr"] / pb["mfr_all"] if pb["mfr_all"] > 0 else np.nan
    gain = bool(cis["d_mfr"][0] > 0 and not np.isnan(rel) and rel >= spec.G_MIN_RELATIVE)
    fails = {
        "R1": bool(cis["d_auc_all"][0] < spec.R1_LOWER_BOUND),
        "R2": bool(point["d_auc_darker"] < spec.R2_POINT or cis["d_auc_darker"][1] < 0),
        "R3": bool(point["d_gap"] > spec.R3_GAP),
        "R4": bool(pm["mfr_darker"] > pb["mfr_darker"]),
    }
    reasons = failure_reasons(fails, cis)
    if reasons:
        verdict = "Rejected: " + "; ".join(reasons.values())
    else:
        verdict = "Supported" if gain else "Not supported"
    return {
        "B0": pb,
        "M1": pm,
        "B0_ci": {k: ci(v) for k, v in eb_reps.items()},
        "M1_ci": {k: ci(v) for k, v in em_reps.items()},
        "delta": point,
        "delta_ci": cis,
        "relative_mfr_reduction": float(rel),
        "G": gain,
        "fails": fails,
        "failure_reasons": reasons,
        "verdict": verdict,
        "baseline_already_robust": bool(pb["mfr_all"] < spec.BASELINE_ROBUST_BELOW),
        "n_images": int(len(coh.y)),
        "n_clusters": int(len(np.unique(coh.clusters))),
        "n_boot": len(boot),
    }


# --------------------------------------------------------------------------- #
# secondary endpoints (section 6, S1-S5): per-cell tables with paired bootstrap CIs
# --------------------------------------------------------------------------- #
def cell_table(coh: Cohort, reps: list[Scored], names: list[str], boot: list[np.ndarray]) -> list[dict]:
    """Per cell and tone group: flip rate, perturbed AUROC, flip direction, sensitivity and specificity at tau."""
    rows = []
    flips = [r.flip_matrix(names) for r in reps]
    for g in GROUPS:
        gm = group_mask(coh.tone, g)

        def stats(idx, gm=gm):
            sel = idx[gm[idx]]
            y = coh.y[sel]
            out = []
            for r, fm in zip(reps, flips):
                pert = np.stack([r.cells[nm][sel] for nm in names], 1)
                clean_call = r.decisions(r.clean[sel])
                pert_call = pert >= r.tau
                fr = fm[sel].mean(0)
                auc = auroc_columns(y, pert)
                pso, ecz = y == 1, y == 0
                p2e = (clean_call[pso, None] & ~pert_call[pso]).mean(0) if pso.any() else np.full(len(names), np.nan)
                e2p = (~clean_call[ecz, None] & pert_call[ecz]).mean(0) if ecz.any() else np.full(len(names), np.nan)
                sens = pert_call[pso].mean(0) if pso.any() else np.full(len(names), np.nan)
                spec_ = (~pert_call[ecz]).mean(0) if ecz.any() else np.full(len(names), np.nan)
                out.append(np.stack([fr, auc, p2e, e2p, sens, spec_]))
            return np.nanmean(np.stack(out), 0)  # (6, cells), averaged over replicates

        every = np.arange(len(coh.y))
        point = stats(every)
        samples = np.stack([stats(idx) for idx in boot])  # (boot, 6, cells)
        for j, nm in enumerate(names):
            row = {"group": g, "cell": nm}
            for q, key in enumerate(("fr", "auc", "pso_to_ecz", "ecz_to_pso", "sens", "spec")):
                row[key] = float(point[q, j])
                row[f"{key}_lo"], row[f"{key}_hi"] = ci(samples[:, q, j])
            rows.append(row)
    return rows


def summary_table(coh: Cohort, methods: dict, sets: dict, boot: list[np.ndarray]) -> list[dict]:
    """mFR over each cell set and clean AUROC by tone group for each method, with paired bootstrap CIs.

    methods: name -> list[Scored]; sets: name (e.g. "CP") -> list of cell names.
    """
    rows = []
    every = np.arange(len(coh.y))
    for mname, reps in methods.items():
        for sname, names in sets.items():
            flips = [r.flip_matrix(names) for r in reps]
            point = method_endpoints(coh, reps, flips, every)
            samples = [method_endpoints(coh, reps, flips, idx) for idx in boot]
            row = {"method": mname, "cells": sname}
            for k in point:
                row[k] = point[k]
                row[f"{k}_lo"], row[f"{k}_hi"] = ci([s[k] for s in samples])
            rows.append(row)
    return rows
