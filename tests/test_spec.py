"""The code's constants must equal the frozen protocol's (PREREGISTRATION.md v1.1)."""

from __future__ import annotations

import re
from pathlib import Path

from nuisance import base, spec

PROTOCOL = Path(__file__).resolve().parents[1] / "PREREGISTRATION.md"
TEXT = PROTOCOL.read_text(encoding="utf-8")
NUM = re.compile(r"[−+-]?\d+(?:\.\d+)?")


def numbers(s: str) -> list[float]:
    return [float(x.replace("−", "-")) for x in NUM.findall(s)]


def grid_rows() -> dict[str, list[str]]:
    rows = {}
    for line in TEXT.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and re.fullmatch(r"[A-H][−+]?", cells[0]):
            rows[cells[0].replace("−", "-")] = cells
    return rows


def test_protocol_file_is_the_frozen_one():
    assert base.sha256_file(PROTOCOL) == spec.PREREG_SHA256


def test_grid_matches_protocol_table():
    rows = grid_rows()
    assert list(rows) == list(spec.FAMILIES)
    for fid, cells in rows.items():
        fam = spec.FAMILIES[fid]
        assert cells[2] == fam.cls, fid
        levels = numbers(rows["G-"][5] if cells[5] == "as G−" else cells[5])[:5]
        assert tuple(levels) == tuple(float(x) for x in fam.levels), fid
        train = cells[6].split(":", 1)[1]
        rng = numbers(rows["G-"][6].split(":", 1)[1] if "as G−" in train else train)
        assert (rng[0], rng[-1]) == tuple(float(x) for x in fam.train), fid


def test_m1_trains_on_colour_preserving_families_only():
    rows = grid_rows()
    for cells in rows.values():
        if cells[2] == "CP":
            assert cells[6].startswith("M1, C1:")
        else:
            assert cells[6].startswith("C1 only:")
    assert spec.M1_FAMILIES == spec.CP and set(spec.C1_FAMILIES) == set(spec.FAMILIES)


def test_constants_appear_in_protocol():
    for phrase in (
        "K = 8 views",
        "0.5 · w_i",
        "LogisticRegression(C=0.003, max_iter=3000)",
        "seeds 0–2",
        "with one draw (seed 0)",
        "reach 80% psoriasis sensitivity",
        "is at least 20%",
        "below −0.02",
        "below −0.01",
        "above +0.02",
        "below 1%",
        "2,000 paired bootstrap resamples",
        "`numpy.random.default_rng([20261009, r, i, v])`",
        "`default_rng([20261009, 99, cohort, i, severity])`",
        "Cohort ids: 0 internal,\n  1 SCIN, 2 Fitzpatrick17k, 3 PASSION",
        "Cap: 15 wall-clock hours",
        "cosine\n  similarity ≥ 0.9999",
        "median |ΔITA| ≤ 2°",
        "|Δb\\*| ≤ 1.0",
        spec.DERMAFAIR_COMMIT,
    ):
        assert phrase in TEXT, phrase
    assert (spec.K_VIEWS, spec.CLEAN_WEIGHT, spec.C, spec.MAX_ITER) == (8, 0.5, 0.003, 3000)
    assert (spec.M1_DRAWS, spec.C1_DRAWS, spec.SENSITIVITY, spec.N_BOOT) == ((0, 1, 2), (0,), 0.80, 2000)
    assert (spec.G_MIN_RELATIVE, spec.R1_LOWER_BOUND, spec.R2_POINT, spec.R3_GAP) == (0.20, -0.02, -0.01, 0.02)
    assert (spec.BASE_SEED, spec.NOISE_STREAM, spec.COMPUTE_CAP_HOURS) == (20261009, 99, 15.0)
    assert spec.COHORT_IDS == {"internal": 0, "scin": 1, "fitz17k": 2, "passion": 3}
    assert (spec.Q1_MIN_COSINE, spec.Q2_MAX_ITA, spec.Q2_MAX_AB) == (0.9999, 2.0, 1.0)


def test_appendix_b_hashes_match_protocol():
    table = {}
    for line in TEXT.split("## Appendix B", 1)[1].splitlines():
        ticks = re.findall(r"`([^`]+)`", line)
        digests = [t for t in ticks if re.fullmatch(r"[0-9a-f]{64}", t)]
        if digests:
            table[Path(ticks[0]).name] = digests[-1]
    assert table == {name: v[2] for name, v in spec.APPENDIX_B.items()}
