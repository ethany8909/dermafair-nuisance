# DermaFair: acquisition-nuisance invariance

Can an eczema–psoriasis image classifier become less sensitive to how a photo was taken (compression, framing,
camera blur and noise) without erasing appearance differences that matter across skin tones?

This repository holds the preregistered protocol for that question, the analysis harness that implements it, and
aggregate evidence from each run. It builds on [DermaFair](https://github.com/ethany8909/dermafair) at commit
`89be54e4a5077d7d47ba6b6f4a6c7dd9a6e4866a`.

## Status

| Date | Status |
|---|---|
| 2026-10-09 | Protocol v1.1 frozen, tag `prereg-v1.1`. |
| 2026-10-12 | [Addendum 1](ADDENDUM-1.md) committed (tag `addendum-1`). Harness `16c4bae`; stage-0 checks run (`stage0-20261013T012829Z`): Q1 pass, Q2 24 of 25 colour-preserving cells survive (JPEG quality 10 reclassified), Q3 confirmed by the author, Q4 149 passed. See [qc/](qc). **The protected comparison (stages 1–5) has not been run** and waits for written authorization from the research-track lead. |

## Contents

| Path | Purpose |
|---|---|
| [PREREGISTRATION.md](PREREGISTRATION.md) | The frozen protocol |
| [ADDENDUM-1.md](ADDENDUM-1.md) | Addendum 1 (tag `addendum-1`): Q2 failure handling, colour QC by tone group, R1 reporting |
| [LOG.md](LOG.md) | Authorizations, deviations and the run index, recorded after the freeze |
| [nuisance/](nuisance) | The analysis harness |
| [tests/](tests) | The unit tests (check Q4); synthetic data only |
| [qc/](qc) | Aggregate results of the stage-0 checks |
| [runs/](runs) | One evidence record per run ([template](templates/run_record.json)) |

## The harness

| Module | What it does |
|---|---|
| `spec.py` | Every value the protocol freezes. `tests/test_spec.py` reads PREREGISTRATION.md and fails if any differs. |
| `base.py` | Loads DermaFair's code from a checkout at `89be54e`, refusing any other commit or a modified tree, and verifies the Appendix B input hashes. |
| `transforms.py` | The 11 transform families of §4, the grid cells and the seeded training-view sampler. |
| `models.py` | B0 (DermaFair's frozen `lab.logreg`), and M1/C1 (the same head plus 8 views per image, half the weight on the clean image). |
| `endpoints.py` | The 80%-sensitivity threshold, decision flips, AUROC, the paired patient bootstrap, G, R1–R4 and the verdict. |
| `qc.py` | Checks Q1–Q4, which read no diagnosis label and no classifier output. Q2 follows Addendum 1: grid cells and CP midpoints, pass and fail in each tone group, inadequate masks flagged, admissible training intervals written to `qc/Q2_admissible.json`. |
| `provenance.py` | Protocol and repository checks, the authorization and compute ledgers, and the run record. |
| `stages.py` | Stage 0 (the checks) and stages 1–4 (the protected comparison). |

**How it enforces the protocol.** Every run refuses to start unless all of these hold:
- PREREGISTRATION.md and ADDENDUM-1.md match their frozen SHA-256;
- the harness working tree is clean;
- DermaFair is at `89be54e` with a clean tree;
- every input matches Appendix B.

Stages 1–4 also refuse unless all of these hold:
- `LOG.md` has a dated authorization naming the stage;
- the latest stage-0 run completed, with the same code. An older pass never stands in for a newer failure;
- that run explicitly records Q1 and Q4 as passing, records the Q3 handoff, and has a Q2 verdict that matches the
  admissibility file;
- every software version it recorded (Python, PyTorch, torchvision, scikit-learn, NumPy, Pillow, libjpeg-turbo) is
  unchanged;
- `qc/Q2_admissible.json` is unchanged, is marked eligible with a real `true`, and leaves at least 13
  colour-preserving cells;
- the Q3 review is recorded;
- earlier stages are complete, and the stage-1 verdict is committed before stage 2;
- compute remains under the 15-hour cap.

## Running it

It needs DermaFair's Python environment (PyTorch, torchvision, scikit-learn, SciPy, pandas, Pillow), a checkout of
DermaFair at the frozen commit, and the cohort data laid out as in DermaFair's
[docs/DATA.md](https://github.com/ethany8909/dermafair/blob/main/docs/DATA.md).

```powershell
git -C path\to\dermafair worktree add --detach ..\dermafair-base 89be54e4a5077d7d47ba6b6f4a6c7dd9a6e4866a
$env:DERMAFAIR_BASE = "path\to\dermafair-base"
$env:DERMAFAIR_DATA_DIR = "path\to\data"
$env:DERMAFAIR_RESULTS_DIR = "path\to\dermafair\results"   # holds the frozen image caches and features
python -m nuisance status
python -m nuisance stage0
```

Per-image arrays, feature banks and contact sheets go to `local/`, which git ignores.

## Data

No patient data is ever committed here. That means no images, image-derived arrays, per-image predictions,
per-image metadata, or image and patient identifiers. Only aggregate tables and figures are committed.

The cohorts (DermaCon-IN, SCIN, Fitzpatrick17k) are public, but licensed for non-commercial research and must not be
redistributed. See DermaFair's docs/DATA.md for how to obtain them.

## License

MIT
