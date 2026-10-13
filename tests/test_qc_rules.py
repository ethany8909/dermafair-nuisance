"""Q4: the Addendum 1 rules for Q2 (passing prefix, admissible intervals, floor, tone-group pass rule, masks)."""

from __future__ import annotations

import numpy as np
import pytest

from nuisance import qc, spec


def all_pass() -> dict:
    return {(f, k, j): True for f in spec.CP for k, js in (("s", spec.SEVERITIES), ("m", (1, 2, 3, 4))) for j in js}


def test_everything_passes_reproduces_the_frozen_protocol():
    adm = qc.admissibility(all_pass())
    assert adm["n_surviving"] == 25 and adm["stage1_eligible"] and not adm["reclassified_to_ca"]
    assert adm["m1_families"] == list(spec.CP)
    for f in spec.CP:
        assert tuple(adm["families"][f]["interval"]) == spec.FAMILIES[f].train


def test_a_failing_most_severe_cell_shrinks_the_interval():
    p = all_pass()
    p[("A", "s", 5)] = False  # JPEG quality 10 fails
    adm = qc.admissibility(p)
    assert adm["families"]["A"] == {"k": 4, "interval": [20, 75]}
    assert adm["reclassified_to_ca"] == ["A5"] and adm["n_surviving"] == 24


def test_a_failing_midpoint_cuts_at_the_last_checked_point_below_it():
    p = all_pass()
    p[("C", "m", 2)] = False  # zoom midpoint between severities 2 and 3 fails
    adm = qc.admissibility(p)
    assert adm["families"]["C"] == {"k": 2, "interval": [0.8, 0.9]}
    assert adm["reclassified_to_ca"] == ["C3", "C4", "C5"]


def test_cells_beyond_the_first_failure_are_reclassified_even_if_they_pass():
    p = all_pass()
    p[("B", "s", 3)] = False  # B4 and B5 pass on their own
    adm = qc.admissibility(p)
    assert adm["families"]["B"]["k"] == 2 and adm["reclassified_to_ca"] == ["B3", "B4", "B5"]


def test_a_family_failing_at_severity_1_is_dropped_from_m1():
    p = all_pass()
    p[("E", "s", 1)] = False
    adm = qc.admissibility(p)
    assert adm["families"]["E"] == {"k": 0, "interval": None}
    assert "E" not in adm["m1_families"] and adm["n_surviving"] == 20


def test_floor_of_13_surviving_cells():
    p = all_pass()
    for f in ("A", "B", "C"):
        p[(f, "s", 1)] = False  # 15 cells lost, 10 survive
    assert not qc.admissibility(p)["stage1_eligible"]
    p = all_pass()
    for f in ("A", "B"):
        p[(f, "s", 2)] = False  # 8 lost, 17 survive
    assert qc.admissibility(p)["stage1_eligible"]


def _summary(lighter: dict, darker: dict) -> dict:
    base = {"share_adequate": 1.0, "abs_ita": 0.5, "abs_a": 0.2, "abs_b": 0.2}
    return {"lighter": {**base, **lighter}, "darker": {**base, **darker}}


@pytest.mark.parametrize(
    "lighter,darker,expected",
    [
        ({}, {}, True),
        ({}, {"abs_ita": 2.5}, False),  # fails in the darker group only
        ({"abs_a": 1.2}, {}, False),
        ({}, {"share_adequate": 0.85}, False),  # not evaluable in the darker group
        ({"abs_b": float("nan")}, {}, False),
        ({"abs_ita": 2.0, "abs_a": 1.0, "abs_b": 1.0}, {}, True),  # limits are inclusive
    ],
)
def test_point_passes_only_in_both_tone_groups(lighter, darker, expected):
    assert qc.point_passes(_summary(lighter, darker)) is expected


def test_inadequate_masks_are_left_out_not_replaced():
    d = np.array([[1.0, 0.1, 0.1], [np.nan, np.nan, np.nan], [3.0, 0.3, 0.3], [5.0, 0.5, 0.5]])
    s = qc.summarize_point(d, {"lighter": np.array([True, True, True, False]), "darker": np.array([0, 0, 0, 1], bool)})
    assert s["lighter"]["n"] == 3 and s["lighter"]["n_adequate"] == 2
    assert s["lighter"]["share_adequate"] == pytest.approx(2 / 3) and s["lighter"]["abs_ita"] == pytest.approx(2.0)
    assert s["darker"]["abs_ita"] == pytest.approx(5.0)


def test_mask_adequacy_threshold():
    mask = np.zeros(1000, bool)
    mask[:500] = True
    assert not qc.adequate(mask)
    mask[500] = True
    assert qc.adequate(mask)


def test_checked_points_cover_cells_and_cp_midpoints():
    names = [qc.point_name(f, k, j) for f, k, j, _ in qc.POINTS]
    assert len(names) == 55 + 20 and len(set(names)) == len(names)
    assert "A1-2" in names and "E4-5" in names and "F-1-2" not in names
