"""Every value the protocol freezes (PREREGISTRATION.md v1.1, sections 4-8 and Appendix B).

tests/test_spec.py reads the protocol's tables and checks them against these values, so the code cannot
drift from the frozen document.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_VERSION = "1.1"
PROTOCOL_TAG = "prereg-v1.1"
PREREG_SHA256 = "901d5e2c49e9815a63079fbf02c4a91760392bc727d96c3cd573c8169875ba37"
DERMAFAIR_COMMIT = "89be54e4a5077d7d47ba6b6f4a6c7dd9a6e4866a"

# --------------------------------------------------------------------------- #
# Section 4: the transform grid
# --------------------------------------------------------------------------- #
SIZE = 256
SEVERITIES = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Family:
    id: str  # ASCII id; the protocol writes the minus sign as U+2212 ("F−")
    name: str
    cls: str  # "CP" colour-preserving, "LA" lightness-altering, "CA" chromaticity-altering
    levels: tuple  # the parameter at severities 1..5
    train: tuple  # (low, high) of the uniform training range
    integer: bool = False  # integer-valued parameter (JPEG quality, resolution)


FAMILIES = {
    f.id: f
    for f in (
        Family("A", "JPEG compression", "CP", (75, 50, 30, 20, 10), (10, 75), integer=True),
        Family("B", "Resolution loss", "CP", (160, 128, 96, 64, 48), (48, 160), integer=True),
        Family("C", "Framing (digital zoom)", "CP", (0.90, 0.80, 0.70, 0.60, 0.50), (0.50, 0.90)),
        Family("D", "Defocus blur", "CP", (0.5, 1.0, 1.5, 2.0, 3.0), (0.5, 3.0)),
        Family("E", "Sensor noise, luminance only", "CP", (2, 4, 6, 9, 12), (2, 12)),
        Family("F-", "Under-exposure", "LA", (-0.25, -0.5, -0.75, -1.0, -1.5), (-1.5, -0.25)),
        Family("F+", "Over-exposure", "LA", (0.25, 0.5, 0.75, 1.0, 1.5), (0.25, 1.5)),
        Family("G-", "Warm cast", "CA", (0.15, 0.3, 0.5, 0.75, 1.0), (0.15, 1.0)),
        Family("G+", "Cool cast", "CA", (0.15, 0.3, 0.5, 0.75, 1.0), (0.15, 1.0)),
        Family("H-", "Desaturation", "CA", (0.8, 0.6, 0.4, 0.2, 0.0), (0.0, 0.8)),
        Family("H+", "Over-saturation", "CA", (1.10, 1.20, 1.35, 1.50, 1.75), (1.10, 1.75)),
    )
}
CP = tuple(k for k, f in FAMILIES.items() if f.cls == "CP")
LA = tuple(k for k, f in FAMILIES.items() if f.cls == "LA")
CA = tuple(k for k, f in FAMILIES.items() if f.cls == "CA")
ALL = tuple(FAMILIES)


def cells(classes: tuple[str, ...] = ("CP", "LA", "CA")) -> list[tuple[str, int]]:
    """Grid cells (family, severity) in protocol order."""
    return [(k, s) for k, f in FAMILIES.items() if f.cls in classes for s in SEVERITIES]


def cell_name(family: str, severity: int) -> str:
    return f"{family}{severity}"


# --------------------------------------------------------------------------- #
# Section 4: checks before authorization
# --------------------------------------------------------------------------- #
Q1_MIN_COSINE = 0.9999
Q2_MAX_ITA = 2.0  # degrees, median |change in median skin ITA|
Q2_MAX_AB = 1.0  # CIELAB units, median |change in median skin a*| and |b*|
Q3_PER_GROUP = 3
Q3_SEVERITIES = (1, 3, 5)

# --------------------------------------------------------------------------- #
# Section 5: models
# --------------------------------------------------------------------------- #
C = 0.003
MAX_ITER = 3000
K_VIEWS = 8
CLEAN_WEIGHT = 0.5  # share of each training image's weight kept on its clean row
M1_DRAWS = (0, 1, 2)
C1_DRAWS = (0,)
M1_FAMILIES = CP
C1_FAMILIES = ALL

# --------------------------------------------------------------------------- #
# Section 6: endpoints and the decision rule
# --------------------------------------------------------------------------- #
SENSITIVITY = 0.80
BASELINE_ROBUST_BELOW = 0.01
G_MIN_RELATIVE = 0.20
R1_LOWER_BOUND = -0.02
R2_POINT = -0.01
R3_GAP = 0.02

# --------------------------------------------------------------------------- #
# Sections 7-8: statistics, seeds, compute
# --------------------------------------------------------------------------- #
N_BOOT = 2000
BASE_SEED = 20261009
NOISE_STREAM = 99
Q3_STREAM = 3  # stream for choosing the Q3 contact-sheet images (an implementation detail of Q3)
COHORT_IDS = {"internal": 0, "scin": 1, "fitz17k": 2, "passion": 3}
COMPUTE_CAP_HOURS = 15.0

# --------------------------------------------------------------------------- #
# Appendix B: frozen inputs. root is "data" (DERMAFAIR_DATA_DIR), "results" (DERMAFAIR_RESULTS_DIR)
# or "hub" (the torch.hub checkpoint folder).
# --------------------------------------------------------------------------- #
APPENDIX_B = {
    "Skin_Metadata-1.csv": (
        "data",
        "Skin_Metadata-1.csv",
        "43ac90d147cba2eac0608093770b6319fad74c087ea06d0eba67da684417f67b",
    ),
    "manifest_clean.csv": (
        "data",
        "manifest_clean.csv",
        "18694a9aaf99dbf154adf42a2f118fcffa02ab7067306ce67e71b1836e0ec7c7",
    ),
    "internal_canonical_manifest.csv": (
        "data",
        "internal_canonical_manifest.csv",
        "9575ff8a8bb017912ebb45c87ca9d3a513fe2cae87ef7a930d6c8f9adb78cb3a",
    ),
    "external_scin_manifest.csv": (
        "data",
        "external_scin_manifest.csv",
        "b4114710aa0baee510da27d6a7e55bd98f04bf7ef0be272c4188784daeb6d520",
    ),
    "fitzpatrick17k.csv": (
        "data",
        "fitzpatrick17k/fitzpatrick17k.csv",
        "851dfaabf1791c2a8b1c2957619f3e0673cf0002f4ce924a5c02d185c7d4a1cd",
    ),
    "internal_256.npy": (
        "results",
        "improvements/cache/internal_256.npy",
        "06139ba65c9a0606323744e9a320648935cd48a53cc6e79c50ef8025e962eb7b",
    ),
    "external_256.npy": (
        "results",
        "improvements/cache/external_256.npy",
        "5fce2218aa698597bf1d30554471c0d09554d04d9aecd9ea2c91bb77e6fbe09f",
    ),
    "fitz17k_256_255.npy": (
        "results",
        "improvements/cache/fitz17k_256_255.npy",
        "64c7d5c642dd59b7450bc836d862746296e0c6f9b9f651e4f37836cc177371b8",
    ),
    "dinov2_b14_clean.npz": (
        "results",
        "improvements/features/dinov2_b14_clean.npz",
        "324ed028404e4f71b1b1b2aee1fb0d335f90952fb4e15f69323a7f000b2e2790",
    ),
    "dinov2_b14_fitz17k_255.npz": (
        "results",
        "improvements/features/dinov2_b14_fitz17k_255.npz",
        "d161b8504364b20487c65563fb08adce35948270fc93bfa0ea04d36b30fcb5d6",
    ),
    "params.json": (
        "results",
        "improvements/results/params.json",
        "7d6299f016464d5f8c8e9c284e4d52ae010bd3ff5e0ac694521a7268468f030b",
    ),
    "final_preds.npz": (
        "results",
        "improvements/results/final_preds.npz",
        "1136834fee53dd03dbabfa73805d106f88a8b29641991cb5047adb291dd4f584",
    ),
    "fitz17k_preds.npz": (
        "results",
        "improvements/results/fitz17k_preds.npz",
        "132e42951ea3d429196536e92c3c8509731871dcac2920a21c1e6341eaa167cd",
    ),
    "dinov2_vitb14_pretrain.pth": (
        "hub",
        "dinov2_vitb14_pretrain.pth",
        "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    ),
}

# Name of B0's stored predictions inside final_preds.npz (DermaFair's September final model).
B0_STORED_KEY = "final: DINOv2 ViT-B/14 (86M) + SCIN-matched labels + class weights"
