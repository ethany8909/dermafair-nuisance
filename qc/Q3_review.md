# Q3: visual review of the contact sheets

| | |
|---|---|
| Run | `stage0-20261013T012829Z` (harness `16c4bae`). The sheets are identical to those of the superseded run `stage0-20261013T011646Z`, since the transforms did not change |
| Sheets | 11, one per family, kept in `local/qc/` and never committed |
| Images | 6 internal images (3 Fitzpatrick III–IV, 3 V–VI, a fixed seeded draw), shown clean and at severities 1, 3 and 5 |

Reviewed: yes (Ethan Yu, 2026-10-17)

| Family | What the sheet shows | As specified in §4 |
|---|---|---|
| A, JPEG | Severities 1 and 3 look like the clean image at this size. Severity 5 (quality 10) shows 8 × 8 blocking and pink-magenta colour blotches | Yes. The blotches match A5's Q2 failure |
| B, resolution | Progressive loss of detail; severity 5 is visibly soft | Yes |
| C, zoom | Centred crops that tighten around the image centre, with no shift, flip or rotation | Yes |
| D, blur | Progressive blur | Yes |
| E, noise | Fine luminance grain at severity 5, with no colour tint | Yes |
| F−, under-exposure | Progressively darker | Yes |
| F+, over-exposure | Progressively brighter; severity 5 is washed out | Yes |
| G−, warm cast | Progressively orange | Yes |
| G+, cool cast | Progressively blue | Yes |
| H−, desaturation | Progressively greyer; severity 5 is grayscale | Yes |
| H+, over-saturation | Progressively more vivid | Yes |

**Pre-review:** Claude (AI assistant), 2026-10-12, at the author's request. No transform behaved differently from §4.

**Author confirmation (Ethan Yu):** 2026-10-17. Ethan reviewed the contact sheets and confirmed that every family behaves as §4 specifies.
