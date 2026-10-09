# Preregistration: less sensitivity to acquisition nuisance, without erasing skin-tone-relevant appearance

| | |
|---|---|
| Version | 1.1 |
| Frozen | 2026-10-09. The protocol is the commit that adds this version, tagged `prereg-v1.1`. Its SHA is sent to the research-track lead. |
| Author | Ethan Yu |
| Status | **Protocol only.** No result-bearing run of the comparison in §6 has been made. Stages 1–5 (§10) run only after written authorization from the research-track lead. |
| Builds on | DermaFair commit `89be54e4a5077d7d47ba6b6f4a6c7dd9a6e4866a` (§1) |
| Preparation | Drafted with help from an AI assistant (Claude, Anthropic), working from the DermaFair records. |

This file is never edited after the freeze. Every later change is a dated entry in [LOG.md](LOG.md).

## 0. Question and hypothesis

> Can we reduce sensitivity to acquisition nuisance (compression, framing, illumination/camera effects) without
> erasing clinically relevant appearance differences across skin-tone groups?

**H1.** Retrain the fixed image-only baseline's classifier (B0) with extra views made by *colour-preserving*
acquisition transforms; this is method M1 (§5). H1 says M1's decision changes less often under those transforms than
B0's, without a material loss of clean discrimination and without a worse darker-skin result (§6).

**Why lightness and colour are not treated as nuisance in M1.** In skin photographs, illumination changes in
lightness or colour move the same colour axes that carry skin tone (L\*, b\*) and erythema (a\*).
- These changes are measured as held-out stress tests.
- A secondary contrast, C1, measures what happens when they *are* treated as nuisance (§3–§5).

## 1. Frozen starting state

- **DermaFair commit `89be54e4a5077d7d47ba6b6f4a6c7dd9a6e4866a`** ([ethany8909/dermafair](https://github.com/ethany8909/dermafair), public since 2026-10-01).
  - Analyses run after this commit belong to a separate, unpublished collaboration. They are not part of this base.
    Appendix A says how they are handled.
- **Baseline code path** (all at this commit):
  - `g_backbones(bank, "harmonized")` in `experiments/improvements/run_experiments.py`, configuration
    "DINOv2 ViT-B/14 (86M)", which DermaFair calls its final model.
  - It fits `lab.logreg` with `lab.class_weights` on the `dinov2_b14_clean` features produced by
    `extract_features.py`.
  - Its stored clean predictions are `final_preds.npz` (internal and SCIN) and `fitz17k_preds.npz`
    (Fitzpatrick17k).
- **Local inputs** are identified by SHA-256 in Appendix B and are never distributed. They are:
  - the image caches, feature banks and manifests
  - the encoder weights
  - B0's stored clean predictions; B0's clean results are already public
- **This repository holds no patient data** (§11).

## 2. Datasets and their status

A search on 2026-10-09 found **no untouched public cohort with both eczema and psoriasis images and skin-tone
labels.** It covered the literature, dataset pages and a 2025 audit of about 30 public dermatology datasets. Every
cohort that can support the two-class endpoint has already been inspected in this project.

| Dataset | Used here | Psoriasis | Tone groups, n (psoriasis) | Tone label | Status | Role |
|---|---|---|---|---|---|---|
| DermaCon-IN ([doi:10.7910/DVN/W7OUZM](https://doi.org/10.7910/DVN/W7OUZM); CC BY-NC-SA 4.0) | 872 images from 511 patients (harmonized labels) | 219 | III–IV 597 (179); V–VI 275 (40) | dataset metadata | **Development.** Every DermaFair choice since July 2026 was made on it | Primary endpoint: 5-fold, patient-level, out of fold |
| SCIN ([google-research-datasets/scin](https://github.com/google-research-datasets/scin)) | 1,128 cases | 113 | I–IV 465 (39); V–VI 106 (18); unknown 557 | self-reported Fitzpatrick | **Retrospective.** Outcomes inspected since 2026-07-28 | Secondary |
| Fitzpatrick17k ([mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k); CC BY-NC-SA 3.0): Atlas Dermatológico images, primary label set | 204 images in 196 near-duplicate clusters | 104 | I–IV 140 (58); V–VI 57 (40); unknown 7 | `fitzpatrick_scale` | **Retrospective.** Tested once on 2026-09-25 and reused since. The never-downloaded DermaAmin images count as retrospective too: same curation and labels, and earlier decisions were conditioned on the seen subset | Secondary |
| PASSION (MICCAI 2024; [passionderm.github.io](https://passionderm.github.io/); PASSION data licence, access by request) | eczema images only | none: eczema, fungal infections, scabies, impetigo | described as pigmented skin; the skin-type field is checked on access | — | **Untouched.** Never requested, downloaded or inspected | Contingent one-class check (§9) |

**Not eligible:**
- DDI: biopsy-selected, mostly neoplastic lesions.
- A 2025 psoriasis segmentation compilation: psoriasis only, from public atlases that may overlap Fitzpatrick17k.
- Derm1M and SkinCAP: aggregated from, or derived from, Fitzpatrick17k and DDI.
- DermNet and SD-198: no skin-tone labels.

**Claims language, fixed now.**
- DermaCon-IN results are called "a preregistered comparison on development data".
- SCIN and Fitzpatrick17k results are called "retrospective evaluation on previously inspected cohorts".
- Neither is ever called external confirmation of the method.
- Only PASSION, used as §9 describes, may be called untouched, and only for its one-class endpoints.
- **No generated (synthetic) images are used anywhere in this protocol.** Any later use needs its own preregistered
  addendum and is secondary only.

## 3. Skin tone is a grouping variable, never a nuisance

1. **Groups:** Fitzpatrick I–IV vs V–VI. Internally this is III–IV vs V–VI, because the cohort has no I–II.
   - Each group comes from the cohort's own label (§2) and is fixed before any transform.
   - Unknown tone is excluded from tone-group results only.
2. **Groups are never re-estimated from pixels.** ITA is used only to check the transforms (Q2), because the
   transforms themselves move it.
3. **No transform used to train M1 may change apparent skin tone or colour.** Q2 enforces this with a numeric limit.
4. **Transforms that change lightness or colour (classes LA and CA, §4) are never pooled** with colour-preserving ones
   in any primary quantity. They are reported separately, by tone group.
5. **Excluded transforms:** tone shifts, melanin or ITA-targeted darkening, erythema recolouring, colour or style
   transfer, and hue rotation.
   - They deliberately change apparent skin tone or lesion colour, which is what this study protects.
   - DermaFair's September 2026 synthetic-image study also found darkened and generated images unhelpful.
6. **Every result is reported per tone group.** A pooled gain never stands in for a per-group result.

## 4. Nuisance transforms (frozen grid)

**Common to every transform.**
- **Input:** the 256 × 256 RGB uint8 image cache that DermaFair's frozen-feature models read (Appendix B).
- **Output:** a 256 × 256 RGB uint8 image. It then goes through B0's unchanged preprocessing, exactly as
  `extract_features.encode` does with the `clean` policy: a 224-px resize (torchvision `Resize`, bilinear,
  antialiased), then ImageNet normalization.
- **Severity 0** is the identity. It must reproduce the stored clean features (Q1).
- **Colour spaces:** "linear light" uses the IEC 61966-2-1 sRGB transfer function; CIELAB uses D65
  (`dermafair.data.skin_tone`).

**Classes.**
- **CP, colour-preserving:** meant to leave skin colour and lightness unchanged, which Q2 checks. These are the only
  nuisances M1 trains on.
- **LA, lightness-altering:** keeps chromaticity but changes lightness. Under-exposure lowers skin ITA the way darker
  skin does, so treating it as nuisance would treat skin lightness as nuisance.
- **CA, chromaticity-altering:** changes a\* (erythema) and b\* (melanin and yellowness).

| ID | Family | Class | Stands for | Implementation | Severity 1 → 5 | Training range |
|---|---|---|---|---|---|---|
| A | JPEG compression | CP | camera or app re-encoding | Pillow JPEG, 4:2:0 chroma subsampling, `optimize` off, decoded to RGB | quality 75, 50, 30, 20, 10 | M1, C1: quality U{10, …, 75} |
| B | Resolution loss | CP | low-resolution capture, down-sizing in transit | bicubic (antialiased) downscale to s × s, bicubic upscale to 256 | s = 160, 128, 96, 64, 48 | M1, C1: s U{48, …, 160} |
| C | Framing (digital zoom) | CP | camera distance and zoom | centred square crop of side f · 256, bicubic resize to 256 | f = 0.90, 0.80, 0.70, 0.60, 0.50 | M1, C1: f U[0.50, 0.90] |
| D | Defocus blur | CP | focus, camera shake | Pillow `GaussianBlur(radius=σ)`, σ in pixels | σ = 0.5, 1.0, 1.5, 2.0, 3.0 | M1, C1: σ U[0.5, 3.0] |
| E | Sensor noise, luminance only | CP | low-light sensor noise | one N(0, σ²) draw per pixel, added equally to R, G and B (changes luma, not chroma), then clipped | σ = 2, 4, 6, 9, 12 (8-bit units) | M1, C1: σ U[2, 12] |
| F− | Under-exposure | LA | dim light, exposure error | linear-light gain 2^EV, clipped | EV = −0.25, −0.5, −0.75, −1.0, −1.5 | C1 only: EV U[−1.5, −0.25] |
| F+ | Over-exposure | LA | bright light, flash | the same, clipping at white | EV = +0.25, +0.5, +0.75, +1.0, +1.5 | C1 only: EV U[0.25, 1.5] |
| G− | Warm cast | CA | uncorrected warm illuminant | linear-light gains (2^(c/2), 1, 2^(−c/2)), divided by their effect on grey luminance | c = 0.15, 0.3, 0.5, 0.75, 1.0 stops of red/blue ratio | C1 only: c U[0.15, 1.0] |
| G+ | Cool cast | CA | uncorrected cool illuminant | the same with −c | as G− | C1 only: as G− |
| H− | Desaturation | CA | camera colour rendering | CIELAB a\* and b\* × k, L\* fixed | k = 0.8, 0.6, 0.4, 0.2, 0.0 | C1 only: k U[0.0, 0.8] |
| H+ | Over-saturation | CA | "vivid" phone rendering | the same; out-of-gamut values clipped | k = 1.10, 1.20, 1.35, 1.50, 1.75 | C1 only: k U[1.10, 1.75] |

That gives 55 perturbed cells (CP 25, LA 10, CA 20) plus the clean image.

**Checks before authorization.** These use no labels and no classifier outputs.
- **Q1, identity:** severity-0 features of every internal image must match the stored clean features, with cosine
  similarity ≥ 0.9999 per image. If Q1 fails, the code is fixed until it passes; the stored features remain the
  clean reference.
- **Q2, colour preservation:**
  - **Measure:** for every cell, over all 1,125 cached internal images, the median absolute change in median skin
    ITA, a\* and b\*, using the DermaFair skin mask.
  - **Zoom** is compared with the identically cropped clean image, so it measures resampling, not content selection.
  - **Pass:** a CP cell passes if median |ΔITA| ≤ 2° and median |Δa\*| and |Δb\*| ≤ 1.0.
  - **If a CP cell fails,** only two reactions are permitted, and both are logged:
    - it is reclassified as CA for every analysis;
    - M1's training range for that family is cut to the passing severities.
  - **LA and CA cells:** their statistics are reported by tone group, to document how far each family moves apparent
    skin tone.
- **Q3, visual check:** a local-only contact sheet (never committed). It shows 6 internal images (3 per tone group)
  under every family at severities 1, 3 and 5.
- **Q4, unit tests** on synthetic arrays:
  - identity at severity 0
  - determinism under fixed seeds
  - dtype and shape
  - monotone change with severity

## 5. Models

**B0, the fixed image-only baseline.** This is DermaFair's public final model (September 2026).
- **Features:** the frozen DINOv2 ViT-B/14 encoder, [CLS ‖ mean patch token], 1,536-d.
- **Head:** `StandardScaler` fit on the clean training-fold images, then scikit-learn
  `LogisticRegression(C=0.003, max_iter=3000)` with the library defaults (L2 penalty, lbfgs solver) and balanced
  class weights n / (2 · n_class), passed as sample weights.
- **Training:** harmonized labels, clean internal training folds only, no metadata and no test-time augmentation.
- **External predictions** average the five fold models' probabilities.
- **What is known:** its clean results are public. They are:
  - internal 0.880
  - SCIN 0.672 (I–IV 0.703, V–VI 0.666)
  - Fitzpatrick17k 0.725 (I–IV 0.756, V–VI 0.641)
- **What is not known:** its results under the grid in §4 have not been computed.

**M1, the one proposed method: tone-preserving nuisance augmentation.**
- **Same as B0** in features, scaler (fit on clean training images), C, class weights and folds. The only change is
  extra training views.
- **Views:** each training image also contributes K = 8 views from the CP families, encoded by the same frozen
  DINOv2 ViT-B/14. A view's family is uniform over A–E, and its parameter is uniform over the training range in §4.
- **Weights:** if w_i is image i's class weight, its clean row gets 0.5 · w_i and each view gets 0.5 · w_i / 8.
  - So every image keeps its B0 total weight, and half of it stays on the clean image.
- **Leakage and testing:** an image's views are used only when that image is in the training folds. There is no
  test-time augmentation.
- **Replicates:** three independent view draws (seeds 0–2) give three M1 models per fold.

**C1, a secondary contrast.** C1 is not a candidate: it can never be adopted or rescue M1.
- **Same as M1,** except that views come uniformly from all 11 families (CP, LA and CA), with one draw (seed 0).
- C1 treats lightness and colour as nuisance, i.e. the "skin tone as style" design. It is included only to test
  whether doing so erases tone-relevant information.

## 6. Endpoints and decision rule

**Definitions.**
- **AUROC** takes psoriasis as the positive class. **Gap** = AUROC(lighter) − AUROC(darker).
- **Operating threshold τ:** the largest score at which a model's pooled clean internal out-of-fold predictions still
  reach 80% psoriasis sensitivity, a screening-oriented operating point. A score ≥ τ is called psoriasis.
  - It is computed from clean predictions only, before any perturbed prediction is read.
  - Each M1 seed has its own τ.
- **Flip:** the image's decision at τ differs between its clean and perturbed versions. **FR(cell)** is the share of
  images that flip.
- **mFR** is the unweighted mean of FR over a set of cells: mFR_CP over 25 cells, mFR_LA over 10, mFR_CA over 20.
- **M1 seeds:** every M1 endpoint is computed per seed, then averaged over seeds. Predictions are never ensembled, so
  M1 gets no ensemble advantage over the single B0.
- **Already-robust baseline:** if mFR_CP(B0) is below 1%, the report says the baseline is already insensitive at
  this operating point. The verdict is still computed as written.

**Primary endpoints** (internal, out of fold, 872 images):
- **Robustness:** mFR_CP, overall.
- **Clean discrimination:** clean AUROC, overall.
- **Subgroup result:**
  - darker-skin (V–VI) clean AUROC
  - the clean tone gap
  - darker-skin mFR_CP

**Primary comparison: M1 vs B0.**

**Invariance gain (G).** ΔmFR = mFR_CP(B0) − mFR_CP(M1). G holds if both:
- the 95% CI of ΔmFR lies above 0;
- the relative reduction ΔmFR / mFR_CP(B0) is at least 20% (point estimates).

**Failure rule.** The gain is rejected if any one of these holds:

| Rule | Rejects when |
|---|---|
| R1, clean discrimination damaged | the lower 95% bound of ΔAUROC_clean (M1 − B0, overall) is below −0.02 |
| R2, darker-skin discrimination worse | the point estimate of ΔAUROC_clean in V–VI is below −0.01, or its 95% CI lies entirely below 0 |
| R3, tone gap wider | the point estimate of Δgap_clean (M1 − B0) is above +0.02 |
| R4, darker-skin robustness worse | mFR_CP(M1) > mFR_CP(B0) within V–VI (point estimates) |

**Verdict (exactly one).**

| Verdict | Condition |
|---|---|
| **Supported** | G holds and R1–R4 all pass |
| **Rejected: harm** | any of R1–R4 fails (each failure is named), whatever G shows |
| **Not supported** | G fails and R1–R4 all pass |

R1–R4 are evaluated and reported even when G fails.

**Secondary endpoints.** These are reported with CIs, play no part in the verdict and get no multiplicity correction.
- **S1:** FR and AUROC per family and severity, overall and by tone, with the flip direction:
  - psoriasis → eczema, among psoriasis images
  - eczema → psoriasis, among eczema images
- **S2, spillover:** mFR_LA and mFR_CA for M1 vs B0, by tone. M1 never trains on these families. A change here shows
  whether colour-preserving invariance leaks into how the model treats colour or lightness.
- **S3, accuracy under perturbation:**
  - mean perturbed AUROC over the CP cells, and its drop from clean
  - sensitivity and specificity at τ for each family at severities 3 and 5, by tone
- **S4, retrospective cohorts** (SCIN and Fitzpatrick17k):
  - clean AUROC by tone
  - mFR_CP, mFR_LA and mFR_CA by tone
  - Here τ is set per cohort, at 80% sensitivity on each model's own clean predictions there, because score
    distributions differ between cohorts. It only defines decisions for counting flips.
- **S5, the C1 contrast:** every endpoint above, for C1 vs B0 and C1 vs M1. **Stated in advance:** C1 will lower
  mFR_LA and mFR_CA more than M1 does, and is more likely than M1 to lower darker-skin clean AUROC or widen the gap.

## 7. Statistical analysis

- **Intervals:** percentile 95% CIs from 2,000 paired bootstrap resamples.
  - **Resampling unit:** patients internally (511 clusters); cases on SCIN; near-duplicate clusters on
    Fitzpatrick17k (DINOv2 cosine ≥ 0.95).
  - Each resample recomputes every endpoint for every model, with τ held fixed. Gap differences use the same paired
    resamples.
- **Missing data:** none internally. An image that fails to transform or encode counts as missing for that cell and
  is reported. None are expected.
- **No imputation, no outlier removal, no interim looks.**
- **Power:**
  - The sample size is fixed by the available data.
  - The darker-skin group is small: 275 images, 40 of them psoriasis. So R2–R4 are guards on point estimates, not
    powered tests.
  - Passing them is not evidence of equal performance.

## 8. Seeds, model selection and compute cap

**Seeds.** Every seed is reported, and there is no seed selection.
- **Base seed:** 20261009.
- **Training views:** M1 uses draws r ∈ {0, 1, 2}; C1 uses r = 0. Image i, view v uses
  `numpy.random.default_rng([20261009, r, i, v])`, where i is the image's row in its cohort cache.
- **Evaluation noise** (family E): `default_rng([20261009, 99, cohort, i, severity])`. Cohort ids: 0 internal,
  1 SCIN, 2 Fitzpatrick17k, 3 PASSION.
- **Bootstrap:** `default_rng(20261009)`.

**Model selection: none.**
- Every setting of B0, M1 and C1 is fixed above.
- C = 0.003 was chosen in September 2026 by inner patient-grouped CV on clean internal data, and is not revisited.
- K = 8, the 0.5 / 0.5 clean-to-view weight split and the training ranges were set without seeing any outcome.
- Nothing is tuned on held-out folds, perturbed outcomes or external cohorts.

**Compute cap.**
- **Machine:** AMD Ryzen 7 8700G (8 cores), 31 GB RAM, no CUDA, PyTorch on CPU.
- **Estimate:** about 150,000 image encodings at about 0.1 s each, roughly 4–5 hours.
- **Cap: 15 wall-clock hours** of result-bearing computation across stages 1–5, failed runs included, as logged in the
  run records.
- **If the cap is reached,** the current stage stops. Unfinished stages are reported as not run and are not completed
  later under this protocol.
- **No stage is shortened, subsampled or re-gridded** to fit the cap.

## 9. How test and external outcomes are kept from choosing the method

1. **Order:**
   - this protocol, frozen at its commit SHA
   - harness code with Q1–Q4 passed, committed; its code SHA is sent with the authorization request
   - written authorization
   - stage 1
   - **the primary verdict, committed**
   - only then any LA/CA, C1 or external perturbed outcome
2. **External cohorts never select, tune or reweight anything,** and cannot change the verdict.
3. **No setting changes once any perturbed or M1 outcome exists.** A variant suggested by results is a new protocol,
   labelled exploratory. It never replaces the verdict.
4. **Bug fixes** may only bring the code into line with this protocol.
   - Each fix is logged with its diff and whether any outcome had been seen.
   - If outcomes had been seen, both runs are reported, and the verdict uses the run that follows the protocol.
5. **The harness refuses to run** unless all three hold, and it writes all three into the run record:
   - its working tree is clean;
   - PREREGISTRATION.md matches the frozen file's SHA-256;
   - a DermaFair checkout is at `89be54e` with a clean tree.

**The untouched cohort (PASSION) is contingent.**
- **It is used only if all three hold:**
  - access is granted under its licence;
  - the research-track lead authorizes it;
  - the stage-1 verdict has been committed.
- **Before scoring,** only its file list and its class × skin-type counts may be read.
- **The run:** one run of the frozen B0 and M1, with the five fold models averaged, on eczema images only.
- **Endpoints:**
  - specificity (eczema called eczema) at the internal τ;
  - mFR_CP, mFR_LA and mFR_CA, overall and by skin type if recorded.
- **Bootstrap unit:** patients if IDs are provided, otherwise images.
- **Reporting:** the result is reported whatever it shows. One-class results cannot test discrimination.

## 10. Stages and the authorization gate

| Stage | Content | May start | Estimated compute |
|---|---|---|---|
| 0 | Harness, unit tests, Q1–Q4 | after this freeze | about 1 h |
| 1 | **Primary:** M1 training views; internal clean and CP grid for B0 and M1; the verdict | after written authorization | about 1.5 h |
| 2 | Internal LA and CA grid; C1 | after the stage-1 verdict is committed | about 1 h |
| 3 | SCIN, all cells | after stage 2 | about 2 h |
| 4 | Fitzpatrick17k, all cells | after stage 3 | about 0.5 h |
| 5 | PASSION (contingent, §9) | after its own authorization | small |

The date and wording of each authorization are recorded in [LOG.md](LOG.md) before its stage runs.

## 11. Evidence artefact and data policy

**Every result-bearing run** leaves `runs/<run_id>/run_record.json` and a short `summary.md` in this repository.
- This covers stages 1–5, including failed or aborted runs.
- The template is [templates/run_record.json](templates/run_record.json).

The record contains:
- run id, stage, start and end (UTC), wall-clock hours, and the machine
- the protocol commit SHA, the SHA-256 of PREREGISTRATION.md, the DermaFair SHA and the harness commit SHA (a clean
  working tree is required)
- the command line and the effective configuration
- software versions, including Pillow's libjpeg-turbo, which must not change between Q1–Q4 and stage 1
- the seeds
- the SHA-256 of every input (the Appendix B files and encoder weights) and every output, local-only arrays included
- the status of Q1–Q4, and warnings such as solver non-convergence
- aggregate results: the endpoint tables with CIs
- deviations raised during the run
- the console log

**Kept locally,** git-ignored and unaltered until the work is published or abandoned:
- per-image predictions for every model × cell, which are needed to recompute every number
- the perturbed feature banks
- the QC contact sheets

**Repository policy:**
- Never committed: images, image-derived arrays, per-image predictions, per-image metadata, and image or patient
  identifiers. `.gitignore` blocks image, array and table formats.
- Only aggregate tables and figures are committed.

## 12. Interpretation fixed in advance

- **Supported:** we say "On development data, tone-preserving nuisance augmentation of a frozen-feature classifier
  reduced decision flips under colour-preserving acquisition transforms by X%, without a material loss of clean or
  darker-skin discrimination." We do not claim robustness to real acquisition shift, a fairness improvement, or
  external confirmation.
- **Rejected: harm:** reported as a negative result that names the rule that failed. It is not rescued by tuning.
- **Not supported:** colour-preserving augmentation of a frozen-feature head did not materially reduce nuisance
  sensitivity. Learning invariance inside the encoder would need a new protocol.
- **C1 vs M1:**
  - If C1 lowers darker-skin clean AUROC or widens the gap more than M1 while cutting LA/CA flips, that supports
    keeping lightness and colour out of the nuisance set.
  - If not, that premise is not supported in these data, and the report says so.
- **Limitations:**
  - Transforms are applied digitally to photographs that were already captured and compressed, so they model
    acquisition only in part.
  - Zoom assumes lesions sit near the centre of the image.
  - A frozen encoder limits what a linear head can learn.
  - The primary endpoint uses development data.
  - The darker-skin group is small.
  - SCIN skin tone is self-reported.

---

## Appendix A. Prior looks relevant to this question

Everything below was seen before this protocol was written.

| Date | Analysis | Cohorts | What was seen |
|---|---|---|---|
| 2026-07-28 | First external validation | SCIN | SCIN outcomes first inspected |
| 2026-08-07 | Centre-crop framing probes (ImageNet ResNet-50 linear probes, single split; code in DermaFair `experiments/external/crop_*`) | internal | Tighter crops removed body-site information faster than diagnostic information |
| 2026-09-24 | B0 chosen as DermaFair's final model. Augmentation (flips, rotation, random crops, colour jitter, blur) with test-time augmentation was tested against it | internal, SCIN | Internal −0.010 (−0.023, +0.004); SCIN +0.035 (+0.005, +0.065); SCIN V–VI 0.000. Not adopted |
| 2026-09-24/25 | Synthetic darker-skin images: colour darkening and image generators | internal, SCIN, Fitzpatrick17k | None adopted. Text-to-image images lowered internal darker-skin AUROC (−0.040) |
| 2026-09-25 | One-shot Fitzpatrick17k test | Fitzpatrick17k | Fitzpatrick17k outcomes first inspected. B0: 0.725 (I–IV 0.756, V–VI 0.641) |

**Unpublished analyses.**
- Between 2026-10-03 and 2026-10-07, further analyses of these cohorts were run for a separate, unpublished
  collaboration. Their results are not reported here.
- They informed three choices in this protocol:
  - the flip rate as the robustness endpoint;
  - exposure, colour-cast and saturation changes as stress tests rather than training transforms;
  - the exclusion of tone- and colour-transfer augmentations.

**Never measured before this protocol:**
- compression, resolution loss, zoom, blur or luminance noise as test-time perturbations of any model
- flip rates under perturbation at the 80%-sensitivity threshold
- M1 and C1 as specified here

**How public earlier results shaped the design:**
- **What M1 leaves out.** In September, augmentation that included colour jitter lowered internal AUROC slightly.
  So M1 drops colour and lightness transforms and keeps half its weight on clean images.
- **Why tone and colour-transfer augmentations are excluded.** The September synthetic-image results support
  excluding them.

## Appendix B. Frozen inputs and environment

SHA-256 of the local files the protocol reads. The files themselves are never distributed.

| Kind | File | Bytes | SHA-256 |
|---|---|---|---|
| metadata | `Skin_Metadata-1.csv` (DermaCon-IN) | 1,494,349 | `43ac90d147cba2eac0608093770b6319fad74c087ea06d0eba67da684417f67b` |
| folds | `manifest_clean.csv` | 64,186 | `18694a9aaf99dbf154adf42a2f118fcffa02ab7067306ce67e71b1836e0ec7c7` |
| metadata | `internal_canonical_manifest.csv` | 67,216 | `9575ff8a8bb017912ebb45c87ca9d3a513fe2cae87ef7a930d6c8f9adb78cb3a` |
| metadata | `external_scin_manifest.csv` | 169,271 | `b4114710aa0baee510da27d6a7e55bd98f04bf7ef0be272c4188784daeb6d520` |
| labels | `fitzpatrick17k/fitzpatrick17k.csv` | 4,104,341 | `851dfaabf1791c2a8b1c2957619f3e0673cf0002f4ce924a5c02d185c7d4a1cd` |
| image cache | `internal_256.npy` (1,125 images) | 221,184,128 | `06139ba65c9a0606323744e9a320648935cd48a53cc6e79c50ef8025e962eb7b` |
| image cache | `external_256.npy` (SCIN, 1,128) | 221,773,952 | `5fce2218aa698597bf1d30554471c0d09554d04d9aecd9ea2c91bb77e6fbe09f` |
| image cache | `fitz17k_256_255.npy` (255) | 50,135,168 | `64c7d5c642dd59b7450bc836d862746296e0c6f9b9f651e4f37836cc177371b8` |
| features | `dinov2_b14_clean.npz` | 13,842,930 | `324ed028404e4f71b1b1b2aee1fb0d335f90952fb4e15f69323a7f000b2e2790` |
| features | `dinov2_b14_fitz17k_255.npz` | 1,566,976 | `d161b8504364b20487c65563fb08adce35948270fc93bfa0ea04d36b30fcb5d6` |
| parameter | `params.json` (C = 0.003) | 18 | `7d6299f016464d5f8c8e9c284e4d52ae010bd3ff5e0ac694521a7268468f030b` |
| B0 predictions | `final_preds.npz` (clean; internal out of fold, SCIN) | 42,642 | `1136834fee53dd03dbabfa73805d106f88a8b29641991cb5047adb291dd4f584` |
| B0 predictions | `fitz17k_preds.npz` (clean; Fitzpatrick17k) | 16,846 | `132e42951ea3d429196536e92c3c8509731871dcac2920a21c1e6341eaa167cd` |
| weights | `dinov2_vitb14_pretrain.pth` (torch.hub `facebookresearch/dinov2`) | 346,378,731 | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |

**Environment at freeze:**
- Windows 11 (10.0.26200); AMD Ryzen 7 8700G (8 cores, 16 threads); 31 GB RAM; no CUDA.
- Python 3.14.6, PyTorch 2.12.1+cpu, torchvision 0.27.1+cpu, scikit-learn 1.9.0, NumPy 2.5.0, Pillow 12.3.0
  (libjpeg-turbo 3.1.4.1).
