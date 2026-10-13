"""Command line: python -m nuisance {status, stage0, stage1, stage2, stage3, stage4}."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m nuisance", description=__doc__)
    p.add_argument("command", choices=["status", "stage0", "stage1", "stage2", "stage3", "stage4"])
    p.add_argument("--workers", type=int, default=None, help="processes for Q2 (stage 0 only)")
    a = p.parse_args()

    from . import provenance as P
    from . import spec, stages

    if a.command == "status":
        head, clean = P.harness_state()
        print(f"protocol {spec.PROTOCOL_TAG} (PREREGISTRATION.md SHA-256 check: {P.check_prereg()[:12]}...)")
        print(f"harness {head[:7]}, working tree {'clean' if clean else 'has changes'}")
        print(f"authorized stages in LOG.md: {sorted(P.authorized_stages()) or 'none'}")
        print(f"compute used: {P.compute_used_hours():.2f} of {spec.COMPUTE_CAP_HOURS} h")
        for r in P.records():
            print(f"  {r['run_id']}: {r['status']}  qc={r['qc'] or '-'}")
        return
    if a.command == "stage0":
        stages.stage0(a.workers)
    else:
        getattr(stages, a.command)()


if __name__ == "__main__":
    main()
