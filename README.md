# DermaFair: acquisition-nuisance invariance

Can an eczema–psoriasis image classifier become less sensitive to how a photo was taken (compression, framing,
camera blur and noise) without erasing appearance differences that matter across skin tones?

This repository holds the preregistered protocol for that question. Its analysis harness and aggregate results will
follow. It builds on [DermaFair](https://github.com/ethany8909/dermafair) at commit
`89be54e4a5077d7d47ba6b6f4a6c7dd9a6e4866a`.

## Status

| Date | Status |
|---|---|
| 2026-10-09 | Protocol v1.1 frozen, tag `prereg-v1.1`. **Protocol only:** no result-bearing comparison has been run. The protected comparison waits for written authorization from the research-track lead. |

## Contents

| File | Purpose |
|---|---|
| [PREREGISTRATION.md](PREREGISTRATION.md) | The frozen protocol: datasets and their status, transform grid, models, endpoints, failure rule, seeds, compute cap, evidence |
| [LOG.md](LOG.md) | Authorizations, deviations and the run index, all recorded after the freeze |
| [templates/run_record.json](templates/run_record.json) | The evidence record that every result-bearing run leaves |

## Data

No patient data is ever committed here. That means no images, image-derived arrays, per-image predictions, per-image
metadata, or image and patient identifiers. Only aggregate tables and figures are committed.

The cohorts (DermaCon-IN, SCIN, Fitzpatrick17k) are public, but licensed for non-commercial research and must not be
redistributed. See DermaFair's [docs/DATA.md](https://github.com/ethany8909/dermafair/blob/main/docs/DATA.md) for
how to obtain them.

## License

MIT
