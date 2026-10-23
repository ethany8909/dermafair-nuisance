# Log

[PREREGISTRATION.md](PREREGISTRATION.md) is frozen at tag `prereg-v1.1` and is never edited. This file records
everything that happens after the freeze, newest entries last.

## Freeze

| Date | Event |
|---|---|
| 2026-10-09 | Protocol v1.1 committed and tagged `prereg-v1.1`. Built on DermaFair `89be54e4a5077d7d47ba6b6f4a6c7dd9a6e4866a`. |
| 2026-10-09 | An earlier v1.0 draft was public briefly and withdrawn before it was shared with anyone, because it included unpublished material from a separate collaboration. v1.1 is the first version shared. |
| 2026-10-12 | [Addendum 1](ADDENDUM-1.md) committed and tagged `addendum-1`, at the research-track lead's request before stage 1. It covers Q2 failure handling, colour QC by tone group, and R1 reporting. It was written before any Q2 result was read. PREREGISTRATION.md is unchanged. |

## Authorizations

Each stage in §10 of the protocol runs only after an entry here.

| Date received | From | Stage(s) authorized | Wording (quoted) |
|---|---|---|---|
| 2026-10-12 | research-track lead | 0 only | Paraphrased: proceed with stage 0 preparation only; stages 1–5 stay on hold until the harness SHA and aggregate Q1–Q4 evidence are sent, as §9 requires. |

## Deviations

Each entry gives the date, what changed, why, and whether any outcome had been seen before the change.

None. Addendum 1 adds rules the protocol left open; it is listed under Freeze.

## Harness changes after review

| Date | Change |
|---|---|
| 2026-10-22 | **Guard repair adopted** from the research-track lead's review of 2026-10-18. The review found four defects in `guard()`, which decides whether stages 1–4 may start: (a) a newer failed stage-0 run could be skipped in favour of an older pass; (b) missing or unknown QC entries could pass; (c) recorded software versions were not compared, although §11 requires them unchanged; (d) the eligibility flag was tested for truthiness, so the string "false" passed. The proposed patch was applied with its logic unchanged (the formatter only reflowed two lines), and the review's 21 synthetic guard tests were added as `tests/test_guard.py`. A stale verdict assertion in `tests/test_pipeline.py` was corrected to the Addendum 1 wording. Nothing else changed: no transform, training range, endpoint, threshold, seed, budget, dataset status, protocol text or addendum. Stage 0 is rerun under the repaired harness. The review also notes, correctly, that colour preservation between checked points is an interpolation assumption (Addendum 1, §1.3); it stays stated in any report. |

## Stage-0 runs (checks; not result-bearing)

| Run id | Harness | Status | Note |
|---|---|---|---|
| stage0-20261013T010109Z | `ad378e0` | superseded | Finished while Addendum 1 was being requested. Its outputs were moved unread to local storage and read only after Addendum 1 was committed and pushed. They showed: Q1 pass; Q2 (one median over all images, before Addendum 1) failing only A5; Q4 125 passed. Kept in `runs/superseded/`; its recorded SHA-256 values also refer to Windows (CRLF) copies, as for the next run. |
| stage0-20261013T011646Z | `67caeb7` | superseded | Its recorded SHA-256 values were computed on Windows (CRLF) copies of its text outputs, while the repository stores LF, so they cannot be checked against the committed files. The harness now writes LF and stage 0 was rerun. Kept in `runs/superseded/`. Its results were: Q1 pass (cosine 1.000000; features reproduced exactly). Q2: 24 of 25 CP cells survive. A5 (JPEG quality 10) and the midpoint at quality 15 fail, so A5 is reclassified as CA and JPEG training draws are limited to quality 20–75 (Addendum 1). Every other CP family passes at all severities and midpoints, with at least 99.4% adequate masks in both tone groups. Q3 pre-reviewed; author confirmation pending. Q4: 148 passed. |
| stage0-20261013T012829Z | `16c4bae` | completed | The stage-0 evidence for harness `16c4bae`; replaced as the guard's reference by the rerun under `2364ab5` (next row). Q1 pass (cosine 1.000000). Q2: 24 of 25 CP cells survive; A5 is reclassified as CA and JPEG training draws are limited to quality 20–75. Content is identical to the superseded run apart from run ids. Q3 reviewed and confirmed by the author on 2026-10-17. Q4: 149 passed. Every recorded SHA-256 matches its committed (LF) file. |
| stage0-20261023T014050Z | `2364ab5` | completed | **The stage-0 evidence for the repaired harness.** Results identical to the previous run apart from run ids: Q1 pass (cosine 1.000000); Q2 24 of 25 CP cells survive (A5 reclassified; JPEG training quality 20–75); Q3 contact sheets byte-identical to those the author confirmed on 2026-10-17; Q4 170 passed, including the 21 guard tests. Software versions unchanged. Every recorded SHA-256 matches its committed file. With only the authorization check mocked, the repaired guard admits this record; without it, stage 1 is refused. |

## Run index

One row per result-bearing run, including failed or aborted runs. Records are in `runs/<run_id>/`.

| Run id | Stage | Status | Wall-clock hours | Record |
|---|---|---|---|---|
| — | — | — | — | — |

Compute used against the 15-hour cap: 0 h.
