# Addendum 1: Q2 failure handling, colour QC by tone group, and R1 reporting

| | |
|---|---|
| Date | 2026-10-12 |
| Applies to | [PREREGISTRATION.md](PREREGISTRATION.md) v1.1 (tag `prereg-v1.1`, commit `986fbca`), which is not edited |
| Requested by | the research-track lead, before stage 1 |
| Status | No result-bearing comparison has been run. No Q2 result had been read when this addendum was written. |

**Timing.**
- A stage-0 run under harness commit `ad378e0` (run `stage0-20261013T010109Z`) was in progress when the request
  arrived. It finished before it could be stopped.
- Its outputs were moved, unread, to local storage, and this addendum was written and committed before any of them was
  read.
- Stage 0 is rerun under the amended harness. The superseded run is kept and logged in [LOG.md](LOG.md).

**Scope.** This addendum:
- fills in rules the protocol left open;
- makes the Q2 pass rule stricter, by tone group;
- changes how R1 failures are worded.

It does not change any model, transform, endpoint definition, threshold or seed. If every checked point passes Q2,
the protocol applies exactly as frozen.

## 1. The surviving colour-preserving cells, the primary endpoint and training draws

The protocol let a failing Q2 cell shrink M1's training range while still defining mFR_CP over 25 cells. It did not
say how five checked severities determine which values of a continuous training range are admissible. This section
fixes both.

### 1.1 Checked points

For each colour-preserving (CP) family, Q2 checks:
- the five grid severities;
- the four midpoints between adjacent severities.

Integer families use the midpoint rounded down. For family E, the midpoint between severities j and j + 1 draws its
noise from `default_rng([20261009, 99, 0, i, 10 + j])`.

| Family | Midpoint 1–2 | 2–3 | 3–4 | 4–5 |
|---|---|---|---|---|
| A, JPEG quality | 62 | 40 | 25 | 15 |
| B, resolution s | 144 | 112 | 80 | 56 |
| C, zoom f | 0.85 | 0.75 | 0.65 | 0.55 |
| D, blur σ | 0.75 | 1.25 | 1.75 | 2.5 |
| E, noise σ | 3 | 5 | 7.5 | 10.5 |

A checked point passes only if it passes in both tone groups (section 2.3).

### 1.2 The passing prefix

- For family f, **k_f** is the largest j from 0 to 5 such that severities 1 to j, and every midpoint between them, all
  pass. k_f = 0 when severity 1 fails.
- **Surviving cells:** severities 1 to k_f of each family. Together they form the surviving set CP*.
- **Reclassified cells:** every severity above k_f is reclassified as chromaticity-altering (CA) for every analysis,
  even if it passed on its own. This keeps the evaluated region equal to the trained region.

### 1.3 Admissible training draws

- The checked points are the only evidence about colour preservation along a family's parameter.
- **Admissible interval:** M1 draws family f's parameter uniformly over the closed interval from its severity-1 value
  to its severity-k_f value. Draws are integers for A and B. This interval replaces the §4 training range.
- **No extrapolation past a failure:** values between a passing and a failing checked point are not admissible, because
  the limit could be crossed anywhere between them.
- **Interpolation inside the interval** rests on two pieces of evidence:
  - every checked point inside it passes, including the midpoints, which test the most likely place for a
    non-monotone excursion;
  - Q4 tests that pixel change grows with severity.
- **Dropped families:** if k_f = 0, family f is dropped from M1's training families, and a view's family is uniform
  over the remaining CP families.
- **C1 is unaffected.** It treats every family as nuisance by design and keeps the §4 ranges. B0 is unaffected.
- **If every checked point passes,** each admissible interval equals the §4 training range and CP* has all 25 cells.

### 1.4 The primary endpoint

- **mFR_CP** is the unweighted mean of FR over the cells in CP*. Its denominator is |CP*|, which is 25 when nothing is
  reclassified. R4 uses the same set.
- **Reclassified cells join the CA set.** mFR_CA averages over the 20 frozen CA cells plus the reclassified ones. The
  20-cell mean and the reclassified cells are also reported separately.
- **Sensitivity analysis:** mFR over all 25 frozen CP cells, with no role in the verdict.
- **Floor:** if |CP*| is below 13 (fewer than half of 25), stage 1 does not run. The Q2 result goes to the
  research-track lead, and any redesign is a new protocol version.
- **The frozen result file:**
  - Stage 0 writes CP*, the admissible intervals and the reclassified cells to `qc/Q2_admissible.json`.
  - That file is committed with the stage-0 evidence.
  - Stages 1–4 read it, and refuse to run if its SHA-256 differs from the value in the stage-0 run record.

## 2. Colour preservation by tone group

### 2.1 Mask

- **The mask:** the DermaFair skin mask, `dermafair.data.skin_tone.skin_mask` at `89be54e`.
  - It keeps pixels whose YCrCb chroma lies in the box Cr 133–180, Cb 70–130.
  - It feathers that box with a Gaussian blur of radius 6 and keeps values above 0.5.
- **Same pixels compared:** the mask is computed once, on the reference image, and applied to both the reference and the
  transformed image.
- **The reference** is the clean cached image. For zoom (C) it is the identically cropped clean image before resizing,
  and its mask is resized to the output with nearest-neighbour interpolation.
- **Per-image statistics:** the medians over masked pixels of per-pixel ITA = atan2(L\* − 50, b\*) in degrees, of a\*,
  and of b\*. Changes are transformed minus reference.

### 2.2 Inadequate masks

- DermaFair's `ita()` helper falls back to the whole image when its mask has 500 pixels or fewer. **Q2 does not fall
  back.**
- A reference mask with 500 pixels or fewer is flagged **inadequate**, and that image is left out of that point's
  medians.
- Counts and shares of inadequate masks are reported for every checked point and tone group.
- **Evaluability:** a point is evaluable in a tone group only if at least 90% of the group's images have adequate
  reference masks for it. A point that is not evaluable in a group fails in that group.

### 2.3 Pass and fail, by tone group (frozen)

- **Tone groups:** every checked point is summarized for Fitzpatrick III–IV (782 cached internal images) and V–VI
  (340), and overall (1,125, including 3 of unknown type, which are left out of the group summaries).
- **The pass rule:** a CP point passes only if, **in each tone group**, it is evaluable and all three of these hold,
  taken over that group's adequate-mask images:
  - median |ΔITA| ≤ 2°;
  - median |Δa\*| ≤ 1.0;
  - median |Δb\*| ≤ 1.0.
- **Failing in either group is failing.** The overall summary is reported but does not decide.
- **LA and CA cells** get the same tables, with no pass or fail.

### 2.4 Independence from outcomes

Q2 reads only the cached images, the skin mask and the Fitzpatrick types in the manifest. It reads no diagnosis
label, no feature and no classifier output. Its result is committed before stage 1.

## 3. Reporting R1, and wording for R2–R4

**The R1 gate is unchanged.** R1 fails when the lower 95% bound of ΔAUROC_clean (M1 − B0, overall) is below −0.02. A
failing R1 is now reported as one of these:

| Report | When |
|---|---|
| R1: non-inferiority not shown (interval too wide) | upper bound ≥ 0, so the data are compatible with no loss |
| R1: clean-performance loss demonstrated | upper bound < 0, but the interval is not entirely below −0.02 |
| R1: material clean-performance loss demonstrated | upper bound < −0.02 |

**R2–R4 are guards on point estimates** (§7):
- R2 is reported as "darker-skin loss demonstrated" only when its 95% CI lies entirely below 0. Otherwise it is
  "darker-skin guard triggered (point estimate)".
- R3 and R4 are reported as guards triggered.

**The verdict wording:** "Rejected: harm" is reported as "Rejected", followed by these categories. "Harm" or "loss" is
said only when it was demonstrated. The decision itself is unchanged.

## 4. What changes in the harness

- **Q2** computes tone-group pass and fail, mask adequacy and the 20 CP midpoints, and writes `qc/Q2_admissible.json`.
- **Stages 1–4:**
  - M1's views are drawn from the admissible intervals.
  - mFR_CP and R4 use CP*.
  - The floor rule and the file check gate these stages.
- **Verdicts** carry the R1 categories and the guard wording.
- **Every run** checks this file's SHA-256 alongside PREREGISTRATION.md's.
