"""The checks every run makes and the evidence record it leaves (PREREGISTRATION.md sections 9 and 11)."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

from . import base, spec

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
LOCAL = REPO / "local"  # git-ignored: per-image arrays, feature banks, contact sheets


# --------------------------------------------------------------------------- #
# repository state
# --------------------------------------------------------------------------- #
def check_prereg() -> str:
    digest = base.sha256_file(REPO / "PREREGISTRATION.md")
    if digest != spec.PREREG_SHA256:
        raise RuntimeError("PREREGISTRATION.md differs from the frozen file (SHA-256 mismatch).")
    return digest


def protocol_commit() -> str:
    return base.git(REPO, "rev-list", "-n", "1", spec.PROTOCOL_TAG)


def harness_state() -> tuple[str, bool]:
    return base.git(REPO, "rev-parse", "HEAD"), not base.git(REPO, "status", "--porcelain")


def code_changed_since(commit: str) -> bool:
    """True if the harness code or tests differ between `commit` and HEAD."""
    out = base.git(REPO, "diff", "--name-only", commit, "HEAD", "--", "nuisance", "tests", "pyproject.toml")
    return bool(out)


def versions() -> dict:
    import numpy
    import PIL
    import sklearn
    import torch
    import torchvision
    from PIL import features

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
    }


# --------------------------------------------------------------------------- #
# authorization and compute ledgers (LOG.md, runs/)
# --------------------------------------------------------------------------- #
_STAGES = re.compile(r"^\s*(\d+(\s*[–-]\s*\d+)?)(\s*,\s*\d+(\s*[–-]\s*\d+)?)*(\s+only)?\s*$")


def parse_stages(column: str) -> set[int]:
    """'1' -> {1}; '2–4' -> {2, 3, 4}; '1, 3' -> {1, 3}; '0 only' -> {0}. Anything else is an error."""
    if not _STAGES.match(column):
        raise ValueError(f"unreadable 'Stage(s) authorized' entry: {column!r}")
    out = set()
    for part in column.replace("only", "").split(","):
        bounds = [int(x) for x in re.split(r"[–-]", part.strip())]
        out.update(range(bounds[0], bounds[-1] + 1))
    return out


def authorized_stages(log: Path = REPO / "LOG.md") -> set[int]:
    """Stages named in a dated row of LOG.md's Authorizations table."""
    text = log.read_text(encoding="utf-8")
    section = text.split("## Authorizations", 1)[1].split("\n## ", 1)[0]
    stages: set[int] = set()
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]):
            stages |= parse_stages(cells[2])
    return stages


def records(stage: int | None = None) -> list[dict]:
    out = []
    for f in sorted(RUNS.glob("*/run_record.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if stage is None or r["stage"] == stage:
            out.append(r)
    return out


def compute_used_hours() -> float:
    """Result-bearing compute so far: stages 1-5, failed runs included."""
    return float(sum(r["wall_clock_hours"] for r in records() if r["stage"] >= 1))


class Budget:
    """Stops a stage when the 15-hour cap would be exceeded (section 8)."""

    def __init__(self):
        self.t0 = time.time()
        self.left = spec.COMPUTE_CAP_HOURS - compute_used_hours()
        if self.left <= 0:
            raise RuntimeError("The compute cap is used up; no further stages may run under this protocol.")

    def check(self):
        if (time.time() - self.t0) / 3600 >= self.left:
            raise RuntimeError("Compute cap reached: this stage stops here and is reported as not run.")


# --------------------------------------------------------------------------- #
# the run record (section 11)
# --------------------------------------------------------------------------- #
class Run:
    """Context manager: refuses to start unless the protocol, the harness and DermaFair are in their frozen state,
    then writes runs/<run_id>/run_record.json, summary.md and console.log whatever the outcome."""

    def __init__(self, stage: int, command: str, config: dict | None = None):
        self.stage, self.command, self.config = stage, command, config or {}
        self.lines: list[str] = []
        self.inputs: dict = {}
        self.outputs: dict = {}
        self.qc: dict = {}
        self.warnings: list[str] = []
        self.deviations: list[str] = []
        self.results: dict = {"summary": "summary.md", "tables": []}
        self.summary = ""
        self.authorization: dict = {}

    def log(self, *parts):
        line = f"{time.strftime('%H:%M:%S')} " + " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def __enter__(self):
        self.prereg = check_prereg()
        self.harness, clean = harness_state()
        if not clean:
            raise RuntimeError("The harness has uncommitted changes; commit them before a run.")
        self.base = base.check_base()
        self.start = dt.datetime.now(dt.timezone.utc)
        self.t0 = time.time()
        self.id = f"stage{self.stage}-{self.start:%Y%m%dT%H%M%SZ}"
        self.dir = RUNS / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.log(
            f"run {self.id}: protocol {spec.PROTOCOL_TAG}, harness {self.harness[:7]}, DermaFair {spec.DERMAFAIR_COMMIT[:7]}"
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "completed" if exc is None else "failed"
        if exc is not None:
            self.log(f"FAILED: {exc_type.__name__}: {exc}")
        end = dt.datetime.now(dt.timezone.utc)
        record = {
            "run_id": self.id,
            "stage": self.stage,
            "status": status,
            "started_utc": self.start.isoformat(timespec="seconds"),
            "ended_utc": end.isoformat(timespec="seconds"),
            "wall_clock_hours": round((time.time() - self.t0) / 3600, 4),
            "machine": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "cuda": False,
            },
            "authorization": self.authorization,
            "provenance": {
                "protocol_commit": protocol_commit(),
                "preregistration_sha256": self.prereg,
                "dermafair_commit": spec.DERMAFAIR_COMMIT,
                "dermafair_tree_clean": True,
                "harness_commit": self.harness,
                "harness_tree_clean": True,
            },
            "command": self.command,
            "config": self.config,
            "software": versions(),
            "seeds": {"base": spec.BASE_SEED, "bootstrap": spec.BASE_SEED},
            "inputs_sha256": self.inputs,
            "outputs_sha256": self.outputs,
            "qc": self.qc,
            "warnings": self.warnings,
            "results": self.results,
            "deviations": self.deviations,
            "console_log": "console.log",
        }
        (self.dir / "run_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (self.dir / "console.log").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        (self.dir / "summary.md").write_text(self.summary or f"# {self.id}\n\nStatus: {status}\n", encoding="utf-8")
        print(f"run record written to {self.dir.relative_to(REPO)}", file=sys.stderr)
        return False  # never swallow the exception
