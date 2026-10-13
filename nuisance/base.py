"""Access to the frozen DermaFair code (commit 89be54e) and to the frozen inputs of Appendix B.

Three environment variables point the harness at the local copies:

    DERMAFAIR_BASE         a checkout of DermaFair at commit 89be54e with a clean working tree
    DERMAFAIR_DATA_DIR     the cohort data, laid out as in DermaFair's docs/DATA.md
    DERMAFAIR_RESULTS_DIR  DermaFair's results folder, which holds the frozen image caches and features

`activate()` refuses to run unless the checkout is at the frozen commit with no local changes, then puts its
code first on sys.path, so `lab`, `extract_features`, `fitz17k_eval` and `dermafair` are the frozen versions.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import spec


def _env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set (see README.md, 'Running the harness').")
    return Path(value).expanduser().resolve()


def git(path: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def check_base() -> Path:
    """The DermaFair checkout, verified to be at the frozen commit with a clean tree."""
    base = _env("DERMAFAIR_BASE")
    head = git(base, "rev-parse", "HEAD")
    if head != spec.DERMAFAIR_COMMIT:
        raise RuntimeError(f"DermaFair checkout is at {head}, not the frozen {spec.DERMAFAIR_COMMIT}.")
    if git(base, "status", "--porcelain"):
        raise RuntimeError("DermaFair checkout has local changes; the protocol requires a clean tree.")
    return base


@lru_cache(maxsize=1)
def activate() -> Path:
    """Verify the checkout and put the frozen DermaFair code first on sys.path (once per process)."""
    base = check_base()
    data_dir(), results_dir()  # fail early if unset
    for p in (base / "experiments" / "improvements", base):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import dermafair

    if Path(dermafair.__file__).resolve().parents[1] != base:
        raise RuntimeError(f"`dermafair` was imported from {dermafair.__file__}, not the frozen checkout.")
    return base


def data_dir() -> Path:
    return _env("DERMAFAIR_DATA_DIR")


def results_dir() -> Path:
    return _env("DERMAFAIR_RESULTS_DIR")


def hub_dir() -> Path:
    return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def input_path(name: str) -> Path:
    root, rel, _ = spec.APPENDIX_B[name]
    return {"data": data_dir, "results": results_dir, "hub": hub_dir}[root]() / rel


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_inputs(names: list[str] | None = None) -> dict[str, str]:
    """SHA-256 of each Appendix B input; raises if any differs from the frozen value."""
    out, bad = {}, []
    for name in names or list(spec.APPENDIX_B):
        digest = sha256_file(input_path(name))
        out[name] = digest
        if digest != spec.APPENDIX_B[name][2]:
            bad.append(name)
    if bad:
        raise RuntimeError(f"Inputs differ from Appendix B: {', '.join(bad)}")
    return out


def internal_tone() -> np.ndarray:
    """Tone group per row of the internal image cache: 0 = Fitzpatrick III-IV, 1 = V-VI, -1 = unknown.

    Reads only the image name and Fitzpatrick type, so no diagnosis label is loaded. Rows follow the cache,
    which DermaFair built in image-name order (common.load_cohorts).
    """
    import pandas as pd

    m = pd.read_csv(input_path("manifest_clean.csv"), usecols=["image_name", "fitzpatrick"])
    fst = m.sort_values("image_name")["fitzpatrick"].to_numpy()
    return np.where(fst >= 5, 1, np.where(fst >= 1, 0, -1))
