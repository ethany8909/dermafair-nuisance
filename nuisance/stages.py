"""Stage runners (PREREGISTRATION.md section 10, with Addendum 1).

    stage 0   checks Q1-Q4 (allowed after the freeze; no labels, no classifier outputs)
    stage 1   the primary comparison and verdict (internal cohort, clean + surviving CP cells, B0 and M1)
    stage 2   internal LA and CA grid, and the C1 contrast
    stage 3   SCIN, all cells (retrospective)
    stage 4   Fitzpatrick17k, all cells (retrospective)

Stages 1-4 refuse to start unless LOG.md has a dated authorization for them, the latest stage-0 run passed with
the same harness code, its Q2 admissibility file is unchanged and allows stage 1, the Q3 review is recorded, earlier
stages are complete (the stage-1 verdict committed before stage 2) and the compute cap is not used up. Stage 5
(PASSION) is not implemented: it needs its own access and authorization (section 9).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import base, qc, spec
from . import endpoints as E
from . import models as M
from . import provenance as P
from . import transforms as T

QC_DIR = P.REPO / "qc"
ADMISSIBLE = QC_DIR / "Q2_admissible.json"
FEATURES = P.LOCAL / "features"


# --------------------------------------------------------------------------- #
# stage 0: the checks
# --------------------------------------------------------------------------- #
def stage0(workers: int | None = None) -> None:
    with P.Run(0, "python -m nuisance stage0", {"workers": workers}) as run:
        base.activate()
        run.inputs = base.verify_inputs()
        run.log(f"all {len(run.inputs)} Appendix B inputs match their frozen SHA-256")
        q4 = qc.q4_tests(P.REPO)
        run.log(f"Q4 unit tests: {q4['summary']}")
        q1 = qc.q1_identity(run.log)
        run.log(f"Q1 identity: min cosine {q1['min_cosine']:.6f} ({'pass' if q1['pass'] else 'FAIL'})")
        q2 = qc.q2_colour(run.log, workers)
        run.log(f"Q2 colour preservation: {q2_status(q2)}")
        q3 = qc.q3_sheets(P.LOCAL / "qc", run.log)
        run.qc = {
            "Q1_identity": "pass" if q1["pass"] else "fail",
            "Q2_colour_preservation": q2_status(q2),
            "Q3_visual_check": "contact sheets written locally; review recorded in qc/Q3_review.md",
            "Q4_unit_tests": "pass" if q4["pass"] else "fail",
        }
        run.outputs = write_qc_reports(q1, q2, q3, q4, run.id)
        run.results["tables"] = sorted(run.outputs)
        run.summary = qc_summary(q1, q2, q3, q4, run.id)


def q2_status(q2: dict) -> str:
    adm = q2["admissible"]
    if q2["pass"]:
        return "pass"
    if adm["stage1_eligible"]:
        return f"pass with reclassification: {adm['n_surviving']} of 25 CP cells survive"
    return f"fail: only {adm['n_surviving']} of 25 CP cells survive (floor {spec.CP_MIN_SURVIVING})"


def write_qc_reports(q1: dict, q2: dict, q3: dict, q4: dict, run_id: str) -> dict:
    QC_DIR.mkdir(exist_ok=True)
    _write_json(QC_DIR / "Q1_identity.json", q1)
    _write_json(QC_DIR / "Q2_colour_preservation.json", q2)
    _write_json(ADMISSIBLE, q2["admissible"])
    _write_json(QC_DIR / "Q4_unit_tests.json", q4)
    P.write_lf(QC_DIR / "Q2_colour_preservation.md", q2_markdown(q2, run_id))
    P.write_lf(
        QC_DIR / "Q3_contact_sheets.md",
        f"# Q3: contact sheets ({run_id})\n\n"
        f"{q3['families']} sheets, one per family. Each shows {q3['images']} internal images (3 Fitzpatrick III–IV, "
        f"3 V–VI, a fixed seeded draw) clean and at severities {', '.join(map(str, q3['severities']))}.\n\n"
        "The sheets show patient photographs, so they stay in `local/qc/` and are never committed. The visual review "
        "is recorded in [Q3_review.md](Q3_review.md).\n",
    )
    return {f"qc/{p.name}": base.sha256_file(p) for p in sorted(QC_DIR.glob("*")) if p.name != "Q3_review.md"}


def _f(x, fmt="{:.2f}") -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def q2_markdown(q2: dict, run_id: str) -> str:
    adm = q2["admissible"]
    lines = [
        f"# Q2: colour preservation by tone group ({run_id})",
        "",
        f"Specified by [Addendum 1](../ADDENDUM-1.md). {q2['n_images']} cached internal images: Fitzpatrick III–IV "
        f"{q2['n_lighter']}, V–VI {q2['n_darker']} ({q2['n_unknown_tone']} of unknown type count only toward the overall "
        "summary in the JSON). ITA in degrees; a\\* and b\\* in CIELAB units. No diagnosis label, feature or classifier "
        "output was read.",
        "",
        "**Mask.** The DermaFair skin mask is computed on the reference image and applied to both images. The reference "
        "is the clean image, or for zoom (C) the identically cropped clean image before resizing. A reference mask with "
        "500 pixels or fewer is inadequate: that image is left out of the point's medians, with no fallback to the "
        "whole image.",
        "",
        "**Pass rule for colour-preserving points.** In each tone group: at least 90% adequate masks, median "
        "\\|ΔITA\\| ≤ 2°, median \\|Δa\\*\\| ≤ 1.0 and median \\|Δb\\*\\| ≤ 1.0.",
        "",
        f"**Result: {q2_status(q2)}.** Surviving CP cells: {adm['n_surviving']} of 25. "
        f"Stage 1 eligible: {'yes' if adm['stage1_eligible'] else 'no'}.",
        "",
        "## Admissible training intervals (Addendum 1, section 1)",
        "",
        "| Family | Passing prefix k | Admissible interval | Surviving cells | Reclassified to CA |",
        "|---|---|---|---|---|",
    ]
    for fam, v in adm["families"].items():
        surv = [c for c in adm["surviving_cp"] if c.startswith(fam)]
        recl = [c for c in adm["reclassified_to_ca"] if c.startswith(fam)]
        interval = f"[{v['interval'][0]}, {v['interval'][1]}]" if v["interval"] else "none (family dropped)"
        lines.append(f"| {fam} | {v['k']} | {interval} | {', '.join(surv) or '—'} | {', '.join(recl) or '—'} |")
    lines += [
        "",
        "## Colour-preserving points: grid cells and midpoints",
        "",
        "| Point | Parameter | III–IV adequate | III–IV \\|ΔITA\\| | \\|Δa\\*\\| | \\|Δb\\*\\| | V–VI adequate | V–VI \\|ΔITA\\| | \\|Δa\\*\\| | \\|Δb\\*\\| | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in (r for r in q2["rows"] if r["class"] == "CP"):
        lo, da = r["lighter"], r["darker"]
        lines.append(
            f"| {r['point']} | {r['param']} | {_f(lo['share_adequate'], '{:.1%}')} | {_f(lo['abs_ita'])} | "
            f"{_f(lo['abs_a'])} | {_f(lo['abs_b'])} | {_f(da['share_adequate'], '{:.1%}')} | {_f(da['abs_ita'])} | "
            f"{_f(da['abs_a'])} | {_f(da['abs_b'])} | {'pass' if r['pass'] else '**fail**'} |"
        )
    lines += [
        "",
        "## Lightness- and chromaticity-altering cells (descriptive; median signed change)",
        "",
        "| Cell | Class | Parameter | III–IV ΔITA | Δa\\* | Δb\\* | V–VI ΔITA | Δa\\* | Δb\\* | Adequate III–IV / V–VI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in (r for r in q2["rows"] if r["class"] != "CP"):
        lo, da = r["lighter"], r["darker"]
        lines.append(
            f"| {r['point']} | {r['class']} | {r['param']} | {_f(lo['ita'], '{:+.2f}')} | {_f(lo['a'], '{:+.2f}')} | "
            f"{_f(lo['b'], '{:+.2f}')} | {_f(da['ita'], '{:+.2f}')} | {_f(da['a'], '{:+.2f}')} | "
            f"{_f(da['b'], '{:+.2f}')} | {_f(lo['share_adequate'], '{:.1%}')} / {_f(da['share_adequate'], '{:.1%}')} |"
        )
    return "\n".join(lines) + "\n"


def qc_summary(q1: dict, q2: dict, q3: dict, q4: dict, run_id: str) -> str:
    ok = lambda b: "pass" if b else "**FAIL**"
    adm = q2["admissible"]
    return (
        f"# Stage 0 checks ({run_id})\n\n"
        "No diagnosis label and no classifier output was read.\n\n"
        "| Check | Result | Detail |\n|---|---|---|\n"
        f"| Q1 identity | {ok(q1['pass'])} | {q1['n_images']} images; min cosine {q1['min_cosine']:.6f} "
        f"(limit ≥ {q1['threshold_min_cosine']}); max abs difference {q1['max_abs_difference']:.2e} |\n"
        f"| Q2 colour preservation | {q2_status(q2)} | {len(q2['rows'])} checked points × {q2['n_images']} images, "
        f"by tone group; {adm['n_surviving']} of 25 CP cells survive |\n"
        f"| Q3 visual check | see qc/Q3_review.md | {q3['families']} local contact sheets |\n"
        f"| Q4 unit tests | {ok(q4['pass'])} | {q4['summary']} |\n"
    )


# --------------------------------------------------------------------------- #
# shared pieces for stages 1-4
# --------------------------------------------------------------------------- #
@dataclass
class Internal:
    rows: np.ndarray  # labelled rows of the internal cache
    y: np.ndarray
    folds: np.ndarray
    tone: np.ndarray
    subject: np.ndarray


def internal_data() -> Internal:
    """Labelled internal images (harmonized labels) through DermaFair's frozen load_bank."""
    from lab import load_bank

    b = load_bank()
    rows = np.flatnonzero(~np.isnan(b.yh))
    tone = np.where(b.fst >= 5, 1, np.where(b.fst >= 1, 0, -1))
    if not np.array_equal(tone, base.internal_tone()):
        raise RuntimeError("tone groups from load_bank and the manifest disagree")
    return Internal(rows, b.yh[rows].astype(int), b.fold[rows], tone[rows], b.subject[rows])


class Encoder:
    """DermaFair's frozen DINOv2 ViT-B/14 feature extractor ('clean' policy)."""

    def __init__(self):
        from extract_features import dinov2, encode

        self.model, self.encode = dinov2("b"), encode

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        return self.encode(self.model, np.ascontiguousarray(imgs))


def m1_sampling(adm: dict) -> tuple[tuple, dict]:
    """Addendum 1, section 1.3: M1's training families and their admissible intervals."""
    fams = tuple(adm["m1_families"])
    return fams, {f: tuple(adm["families"][f]["interval"]) for f in fams}


def cell_sets(adm: dict) -> dict:
    """Cell sets for the endpoints (Addendum 1, section 1.4)."""
    names = lambda cls: [spec.cell_name(f, s) for f, s in spec.cells((cls,))]
    sets = {"CP": list(adm["surviving_cp"]), "LA": names("LA"), "CA": names("CA") + list(adm["reclassified_to_ca"])}
    if adm["reclassified_to_ca"]:
        sets.update({"CP_frozen25": names("CP"), "CA_frozen20": names("CA"), "reclassified": adm["reclassified_to_ca"]})
    return sets


def grid_features(cohort: str, imgs, rows: np.ndarray, cells: list, enc, budget, run) -> dict:
    """Features of every image in `rows` under each grid cell, cached under local/features/<cohort>/."""
    out = {}
    for fam, sev in cells:
        name = spec.cell_name(fam, sev)
        path = FEATURES / cohort / f"{name}.npy"
        if not path.exists():
            moved = np.stack([T.cell(np.asarray(imgs[i]), fam, sev, cohort, int(i)) for i in rows])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, enc(moved))
            run.log(f"{cohort} {name}: {len(rows)} images encoded")
            budget.check()
        run.outputs[f"local/features/{cohort}/{name}.npy"] = base.sha256_file(path)
        out[name] = np.load(path)
    return out


def view_features(
    method: str, draw: int, families: tuple, imgs, rows, enc, budget, run, ranges: dict | None = None, chunk: int = 64
):
    """(n, K, d) features of K training views per internal image, cached under local/features/views/.

    The file name carries a tag of the families and ranges, so views drawn under other intervals are never reused."""
    tag = hashlib.sha256(json.dumps([list(families), ranges or {}], sort_keys=True).encode()).hexdigest()[:8]
    path = FEATURES / "views" / f"{method}_r{draw}_{tag}.npy"
    if not path.exists():
        parts = []
        for s in range(0, len(rows), chunk):
            batch = [
                T.draw_view(np.asarray(imgs[i]), T.view_rng(draw, int(i), v), families, ranges)[0]
                for i in rows[s : s + chunk]
                for v in range(spec.K_VIEWS)
            ]
            f = enc(np.stack(batch))
            parts.append(f.reshape(len(f) // spec.K_VIEWS, spec.K_VIEWS, f.shape[1]))
            budget.check()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.concatenate(parts))
        run.log(f"{method} draw {draw}: {len(rows)} x {spec.K_VIEWS} views encoded")
    run.outputs[f"local/features/views/{path.name}"] = base.sha256_file(path)
    return np.load(path)


def score_internal(X, y, folds, cell_feats: dict, V=None) -> tuple[E.Scored, list]:
    """Fit one replicate per fold; out-of-fold scores on clean and perturbed images; tau from the clean scores."""
    predictors = M.fit_folds(X, y, folds, V)
    clean = M.out_of_fold(predictors, folds, X)
    cells = {nm: M.out_of_fold(predictors, folds, F) for nm, F in cell_feats.items()}
    return E.Scored(clean, cells, E.tau_at_sensitivity(clean, y)), predictors


def score_external(predictors: list, X, y, cell_feats: dict) -> E.Scored:
    """Five fold models averaged; tau set on this cohort's clean scores (section 6, S4)."""
    clean = M.external(predictors, X)
    cells = {nm: M.external(predictors, F) for nm, F in cell_feats.items()}
    return E.Scored(clean, cells, E.tau_at_sensitivity(clean, y))


def guard(stage: int) -> tuple[dict, dict]:
    """Refuse to run a protected stage unless every precondition holds; return (authorization, admissibility)."""
    if stage not in P.authorized_stages():
        raise PermissionError(f"Stage {stage} has no dated authorization in LOG.md; it may not run.")
    s0 = [r for r in P.records(0) if r["status"] == "completed"]
    if not s0:
        raise PermissionError("No completed stage-0 run.")
    last = s0[-1]
    if any(str(v).startswith("fail") for v in last["qc"].values()):
        raise PermissionError(f"Stage-0 checks did not pass in {last['run_id']}: {last['qc']}")
    if P.code_changed_since(last["provenance"]["harness_commit"]):
        raise PermissionError("The harness code changed after the last stage-0 run; rerun stage 0 first.")
    if base.sha256_file(ADMISSIBLE) != last["outputs_sha256"].get("qc/Q2_admissible.json"):
        raise PermissionError("qc/Q2_admissible.json differs from the file the last stage-0 run wrote.")
    adm = json.loads(ADMISSIBLE.read_text(encoding="utf-8"))
    if not adm["stage1_eligible"]:
        raise PermissionError(f"Only {adm['n_surviving']} CP cells survive Q2; stage 1 may not run (Addendum 1, 1.4).")
    if "Reviewed: yes" not in (QC_DIR / "Q3_review.md").read_text(encoding="utf-8"):
        raise PermissionError("The Q3 visual review is not recorded in qc/Q3_review.md.")
    if stage >= 2:
        tracked = base.git(P.REPO, "ls-files", "runs")
        if not any(p.startswith("runs/stage1-") and p.endswith("verdict.json") for p in tracked.splitlines()):
            raise PermissionError("The stage-1 verdict must be committed before stage 2 or later.")
    for earlier in range(1, stage):
        if not any(r["status"] == "completed" for r in P.records(earlier)):
            raise PermissionError(f"Stage {earlier} has not completed; stages run in order (section 10).")
    return {"stages": sorted(P.authorized_stages()), "stage0_run": last["run_id"]}, adm


def _primary_markdown(res: dict, run_id: str, n_cells: int) -> str:
    d, c = res["delta"], res["delta_ci"]
    f = lambda k: f"{d[k]:+.4f} [{c[k][0]:+.4f}, {c[k][1]:+.4f}]"
    lines = [
        f"# Primary comparison: M1 vs B0 ({run_id})",
        "",
        f"**Verdict: {res['verdict']}**",
        "",
        f"{res['n_images']} internal images, {res['n_clusters']} patients, {res['n_boot']} paired patient-bootstrap "
        f"resamples; mFR_CP over {n_cells} surviving colour-preserving cells.",
        "",
        "| Quantity | B0 | M1 | Change (95% CI) |",
        "|---|---|---|---|",
        f"| mFR_CP, all | {res['B0']['mfr_all']:.4f} | {res['M1']['mfr_all']:.4f} | {f('d_mfr')} (B0 − M1) |",
        f"| mFR_CP, V–VI | {res['B0']['mfr_darker']:.4f} | {res['M1']['mfr_darker']:.4f} | {f('d_mfr_darker')} (B0 − M1) |",
        f"| Clean AUROC, all | {res['B0']['auc_all']:.4f} | {res['M1']['auc_all']:.4f} | {f('d_auc_all')} |",
        f"| Clean AUROC, V–VI | {res['B0']['auc_darker']:.4f} | {res['M1']['auc_darker']:.4f} | {f('d_auc_darker')} |",
        f"| Clean tone gap | {res['B0']['gap']:.4f} | {res['M1']['gap']:.4f} | {f('d_gap')} |",
        "",
        f"Relative mFR_CP reduction: {res['relative_mfr_reduction']:.3f} (G needs ≥ {spec.G_MIN_RELATIVE} and a CI above "
        f"0). G holds: {res['G']}. R1: {E.r1_category(*c['d_auc_all'])}. Failure rules: "
        + ", ".join(f"{k} {'fails' if v else 'passes'}" for k, v in res["fails"].items())
        + ".",
    ]
    if res["baseline_already_robust"]:
        lines += [
            "",
            f"mFR_CP(B0) is below {spec.BASELINE_ROBUST_BELOW}: the baseline is already insensitive at this operating point.",
        ]
    return "\n".join(lines) + "\n"


def _write_json(path: Path, obj) -> None:
    P.write_lf(path, json.dumps(obj, indent=2, default=float) + "\n")


# --------------------------------------------------------------------------- #
# stage 1: the primary comparison
# --------------------------------------------------------------------------- #
def stage1() -> None:
    auth, adm = guard(1)
    with P.Run(1, "python -m nuisance stage1") as run:
        run.authorization = auth
        budget = P.Budget()
        base.activate()
        run.inputs = base.verify_inputs()
        D = internal_data()
        imgs = qc._cache()
        X = np.load(base.input_path("dinov2_b14_clean.npz"))["int"][D.rows]
        enc = Encoder()
        names = list(adm["surviving_cp"])
        cp = [(f, s) for f, s in spec.cells(("CP",)) if spec.cell_name(f, s) in names]
        F = grid_features("internal", imgs, D.rows, cp, enc, budget, run)
        fams, ranges = m1_sampling(adm)
        V = {r: view_features("M1", r, fams, imgs, D.rows, enc, budget, run, ranges) for r in spec.M1_DRAWS}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b0, _ = score_internal(X, D.y, D.folds, F)
            m1 = [score_internal(X, D.y, D.folds, F, V[r])[0] for r in spec.M1_DRAWS]
        run.warnings += sorted({str(w.message) for w in caught})
        stored = np.load(base.input_path("final_preds.npz"))[spec.B0_STORED_KEY + "|oof"][D.rows]
        diff = float(np.abs(b0.clean - stored).max())
        run.log(f"B0 refit vs DermaFair's stored final-model predictions: max difference {diff:.2e}")
        if diff > 1e-6:
            raise RuntimeError("B0 does not reproduce DermaFair's stored final model")
        coh = E.Cohort(D.y, D.tone, D.subject)
        boot = E.bootstrap_indices(coh.clusters)
        res = E.primary(coh, [b0], m1, names, boot)
        res["cp_cells"] = names
        _write_json(run.dir / "verdict.json", res)
        P.write_lf(run.dir / "PRIMARY.md", _primary_markdown(res, run.id, len(names)))
        _write_json(
            run.dir / "S1_cells_CP.json",
            {"B0": E.cell_table(coh, [b0], names, boot), "M1": E.cell_table(coh, m1, names, boot)},
        )
        run.results.update({"verdict": res["verdict"], "tables": ["verdict.json", "PRIMARY.md", "S1_cells_CP.json"]})
        run.summary = _primary_markdown(res, run.id, len(names))
        run.log(f"VERDICT: {res['verdict']}")


# --------------------------------------------------------------------------- #
# stage 2: internal LA/CA cells and the C1 contrast
# --------------------------------------------------------------------------- #
def stage2() -> None:
    auth, adm = guard(2)
    with P.Run(2, "python -m nuisance stage2") as run:
        run.authorization = auth
        budget = P.Budget()
        base.activate()
        run.inputs = base.verify_inputs()
        D = internal_data()
        imgs = qc._cache()
        X = np.load(base.input_path("dinov2_b14_clean.npz"))["int"][D.rows]
        enc = Encoder()
        F = grid_features("internal", imgs, D.rows, spec.cells(), enc, budget, run)
        fams, ranges = m1_sampling(adm)
        V1 = {r: view_features("M1", r, fams, imgs, D.rows, enc, budget, run, ranges) for r in spec.M1_DRAWS}
        VC = {r: view_features("C1", r, spec.C1_FAMILIES, imgs, D.rows, enc, budget, run) for r in spec.C1_DRAWS}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            methods = {
                "B0": [score_internal(X, D.y, D.folds, F)[0]],
                "M1": [score_internal(X, D.y, D.folds, F, V1[r])[0] for r in spec.M1_DRAWS],
                "C1": [score_internal(X, D.y, D.folds, F, VC[r])[0] for r in spec.C1_DRAWS],
            }
        run.warnings += sorted({str(w.message) for w in caught})
        _secondary(run, E.Cohort(D.y, D.tone, D.subject), methods, cell_sets(adm), contrast=True)


# --------------------------------------------------------------------------- #
# stages 3-4: retrospective external cohorts
# --------------------------------------------------------------------------- #
def _external(stage: int, cohort: str) -> None:
    auth, adm = guard(stage)
    with P.Run(stage, f"python -m nuisance stage{stage}") as run:
        run.authorization = auth
        budget = P.Budget()
        base.activate()
        run.inputs = base.verify_inputs()
        D = internal_data()
        X = np.load(base.input_path("dinov2_b14_clean.npz"))["int"][D.rows]
        enc = Encoder()
        imgs = qc._cache()
        fams, ranges = m1_sampling(adm)
        V1 = {r: view_features("M1", r, fams, imgs, D.rows, enc, budget, run, ranges) for r in spec.M1_DRAWS}
        VC = {r: view_features("C1", r, spec.C1_FAMILIES, imgs, D.rows, enc, budget, run) for r in spec.C1_DRAWS}
        ext = scin_data() if cohort == "scin" else fitz17k_data()
        F = grid_features(cohort, ext["imgs"], ext["rows"], spec.cells(), enc, budget, run)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fits = {
                "B0": [M.fit_folds(X, D.y, D.folds)],
                "M1": [M.fit_folds(X, D.y, D.folds, V1[r]) for r in spec.M1_DRAWS],
                "C1": [M.fit_folds(X, D.y, D.folds, VC[r]) for r in spec.C1_DRAWS],
            }
        run.warnings += sorted({str(w.message) for w in caught})
        methods = {k: [score_external(p, ext["X"], ext["y"], F) for p in v] for k, v in fits.items()}
        _secondary(run, E.Cohort(ext["y"], ext["tone"], ext["clusters"]), methods, cell_sets(adm), contrast=False)


def scin_data() -> dict:
    from lab import load_bank

    b = load_bank()
    imgs = np.load(base.input_path("external_256.npy"), mmap_mode="r")
    X = np.load(base.input_path("dinov2_b14_clean.npz"))["ext"]
    f = b.fst_ext
    rows = np.arange(len(b.y_ext))
    return {
        "imgs": imgs,
        "rows": rows,
        "X": X,
        "y": b.y_ext.astype(int),
        "tone": np.where(f >= 5, 1, np.where(f >= 1, 0, -1)),
        "clusters": rows,  # cases (section 7)
    }


def fitz17k_data() -> dict:
    """The primary-label Fitzpatrick17k images, with near-duplicate clusters, as DermaFair's frozen code builds them."""
    from fitz17k_eval import DUP, clusters, cohort, label_sets
    from lab import features, load_bank

    d = cohort()
    imgs = np.load(base.input_path("fitz17k_256_255.npy"), mmap_mode="r")
    Zb = np.load(base.input_path("dinov2_b14_fitz17k_255.npz"))["X"]
    if len(d) != len(imgs) or len(d) != len(Zb):
        raise RuntimeError("Fitzpatrick17k cohort, cache and features are not aligned")
    fb = features(load_bank(), "dinov2_b14_clean")
    U = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
    leak = ((U(Zb) @ U(fb["int"]).T).max(1) >= DUP) | ((U(Zb) @ U(fb["ext"]).T).max(1) >= DUP)
    keep = np.flatnonzero(~leak)
    cl = clusters(Zb[keep])
    y_all = label_sets(d["label"].iloc[keep].reset_index(drop=True))["primary"]
    m = ~np.isnan(y_all)
    rows = keep[m]
    f = d["fitzpatrick_scale"].to_numpy()[rows]
    return {
        "imgs": imgs,
        "rows": rows,
        "X": Zb[rows],
        "y": y_all[m].astype(int),
        "tone": np.where(f >= 5, 1, np.where(f >= 1, 0, -1)),
        "clusters": cl[m],
    }


def _secondary(run, coh: E.Cohort, methods: dict, sets: dict, contrast: bool) -> None:
    """S1-S5 tables for one cohort (section 6), over the Addendum 1 cell sets."""
    boot = E.bootstrap_indices(coh.clusters)
    every = [spec.cell_name(f, s) for f, s in spec.cells()]
    _write_json(run.dir / "S_summary.json", E.summary_table(coh, methods, sets, boot))
    _write_json(run.dir / "S_cells.json", {m: E.cell_table(coh, reps, every, boot) for m, reps in methods.items()})
    tables = ["S_summary.json", "S_cells.json"]
    if contrast:  # S5: C1 against B0 and against M1, with the same machinery as the primary comparison
        s5 = {
            "C1_vs_B0": E.primary(coh, methods["B0"], methods["C1"], sets["CP"], boot),
            "C1_vs_M1": E.primary(coh, methods["M1"], methods["C1"], sets["CP"], boot),
            "note": "C1 is a secondary contrast; these 'verdict' fields are not verdicts and cannot be adopted.",
        }
        _write_json(run.dir / "S5_contrast.json", s5)
        tables.append("S5_contrast.json")
    run.results["tables"] = tables
    run.summary = f"# {run.id}\n\nSecondary endpoints written: {', '.join(tables)}. They have no role in the verdict.\n"
    run.log("secondary tables written")


def stage3() -> None:
    _external(3, "scin")


def stage4() -> None:
    _external(4, "fitz17k")
