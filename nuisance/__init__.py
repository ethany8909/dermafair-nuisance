"""Analysis harness for the preregistered nuisance-invariance study (PREREGISTRATION.md v1.1).

Modules:
    spec        every value the protocol freezes
    base        access to the frozen DermaFair code (commit 89be54e) and its inputs
    transforms  the 11 nuisance-transform families and the training-view sampler
    models      the baseline B0, the proposed method M1 and the contrast C1
    endpoints   operating threshold, decision flips, AUROC, bootstrap, failure rule and verdict
    qc          the checks run before authorization (Q1-Q3; Q4 is the test suite)
    provenance  hashes, repository state, authorization and compute ledger, run records
    stages      stage 0 (checks) and stages 1-4 (the protected comparison, which needs authorization)

Run ``python -m nuisance --help``.
"""
