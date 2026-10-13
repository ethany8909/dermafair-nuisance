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

## Stage-0 runs (checks; not result-bearing)

| Run id | Harness | Status | Note |
|---|---|---|---|
| stage0-20261013T010109Z | `ad378e0` | superseded | Finished while Addendum 1 was being requested. Its outputs were moved unread to local storage and read only after Addendum 1 was committed and pushed. They showed: Q1 pass; Q2 (one median over all images, before Addendum 1) failing only A5; Q4 125 passed. Kept in `runs/superseded/`. |
| stage0-20261013T011646Z | `67caeb7` | completed | Q1 pass (cosine 1.000000; features reproduced exactly). Q2: 24 of 25 CP cells survive. A5 (JPEG quality 10) and the midpoint at quality 15 fail, so A5 is reclassified as CA and JPEG training draws are limited to quality 20–75 (Addendum 1). Every other CP family passes at all severities and midpoints, with at least 99.4% adequate masks in both tone groups. Q3 pre-reviewed; author confirmation pending. Q4: 148 passed. |

## Run index

One row per result-bearing run, including failed or aborted runs. Records are in `runs/<run_id>/`.

| Run id | Stage | Status | Wall-clock hours | Record |
|---|---|---|---|---|
| — | — | — | — | — |

Compute used against the 15-hour cap: 0 h.
