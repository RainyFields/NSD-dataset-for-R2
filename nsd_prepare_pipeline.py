#!/usr/bin/env python3
"""
NSD Dataset Preparation Pipeline (full spec).
Reference: https://cvnlab.slite.com/api/s/channel/CPyFRAyDYpxdkPK6YbB5R1/NSD%20Data%20Manual
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Package bootstrap (0.1) — MUST run before importing matplotlib:
# matplotlib loads PIL from matplotlib.colors at import time.
# Order: Pillow and numeric stack first, then matplotlib.
# -----------------------------------------------------------------------------
REQUIRED_PKGS = [
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("h5py", "h5py"),
    ("nibabel", "nibabel"),
    ("cv2", "opencv-python"),
    # nilearn: install via pip/conda but import lazily (see Part 8B) — importing
    # nilearn early pulls numpy.testing and can fail on a broken mixed numpy install.
    ("nilearn", "nilearn"),
]


def ensure_packages() -> None:
    for mod, pip_name in REQUIRED_PKGS:
        name = "PIL" if mod == "PIL" else mod
        try:
            __import__(name)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet"]
            )
    import matplotlib as _mpl

    _mpl.use("Agg")

    import h5py as _h5  # noqa: F401
    import nibabel as _nib  # noqa: F401
    import cv2 as _cv2  # noqa: F401
    from PIL import Image as _Img  # noqa: F401

    print("[0.1] Package versions:")
    import numpy as _np

    print(f"  Python {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print(
            "  WARNING: Pipeline targets Python 3.8+. Use conda env 'fmri' "
            "or upgrade Python."
        )
    print(f"  h5py {_h5.__version__}")
    print(f"  nibabel {_nib.__version__}")
    print(f"  numpy {_np.__version__}")
    print(f"  scipy {__import__('scipy').__version__}")
    print(f"  pandas {__import__('pandas').__version__}")
    print(f"  matplotlib {_mpl.__version__}")
    print(f"  Pillow {_Img.__version__}")
    print(f"  opencv {_cv2.__version__}")
    try:
        import nilearn as _nl

        print(f"  nilearn {_nl.__version__}")
    except Exception as e:
        print(
            f"  nilearn: NOT IMPORTABLE ({e!s}). "
            "Fix NumPy (see environment.yml / conda reinstall). "
            "Surface plots (Part 8B) will fail until fixed."
        )


ensure_packages()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy import stats
from scipy.io import loadmat

import h5py
import nibabel as nib

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
NSD_MANUAL_URL = (
    "https://cvnlab.slite.com/api/s/channel/CPyFRAyDYpxdkPK6YbB5R1/NSD%20Data%20Manual"
)
S3_BUCKET = "s3://natural-scenes-dataset"
ROOT = Path(".").resolve()
PREP = ROOT / "nsd_prepared"
TMP = ROOT / "nsd_tmp"
ERR_LOG = PREP / "errors.log"

# FreeSurfer label lookup on NSD S3 uses *.mgz.ctab (not *.mgz.txt)
LABEL_LOOKUP_SUFFIX = ".mgz.ctab"

ROI_ORDER = ["V1", "V2", "V4", "IT"]
ROI_COLORS_QC = {
    "V1": "#E63946",
    "V2": "#457B9D",
    "V4": "#E9C46A",
    "IT": "#9B5DE5",
}
SURF_ROI_COLORS = {
    0: "#CCCCCC",
    1: "#E63946",
    2: "#457B9D",
    3: "#2A9D8F",
    4: "#E9C46A",
    5: "#9B5DE5",
}


def setup_logging() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(ERR_LOG),
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )
    # Nibabel logs NIfTI header quirks (e.g. pixdim[0]/qfac) at INFO for NSD atlases — harmless.
    logging.getLogger("nibabel").setLevel(logging.WARNING)
    logging.getLogger("nibabel.nifti1").setLevel(logging.WARNING)


def log_error(msg: str, exc: Optional[BaseException] = None) -> None:
    if exc is not None:
        msg = f"{msg}\n{traceback.format_exc()}"
    logging.error(msg)
    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def check_disk_space(
    required_gb: float, path: str = ".", buffer_gb: float = 10.0
) -> None:
    free_gb = shutil.disk_usage(path).free / 1e9
    print(
        f"[DISK] Free: {free_gb:.1f} GB | Required: {required_gb:.1f} GB | "
        f"Buffer: {buffer_gb:.1f} GB"
    )
    if free_gb < required_gb + buffer_gb:
        raise RuntimeError(
            f"Insufficient disk space. Need {required_gb + buffer_gb:.1f} GB free, "
            f"have {free_gb:.1f} GB. Please free up space and re-run."
        )


def print_part_disk(part_name: str) -> None:
    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"[{part_name}] Free disk space: {free_gb:.1f} GB")


def _aws_child_environ() -> dict:
    """Env for all `aws` subprocesses: more internal retries on flaky networks."""
    e = os.environ.copy()
    e.setdefault("AWS_RETRY_MODE", "adaptive")
    e.setdefault("AWS_MAX_ATTEMPTS", "15")
    return e


def aws_run(args: List[str], **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("env", _aws_child_environ())
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        **kwargs,
    )


def aws_s3_cp(src: str, dst: str | Path, recursive: bool = False) -> None:
    dst = Path(dst)
    if recursive:
        dst.mkdir(parents=True, exist_ok=True)
        check_disk_space(0.1, str(dst))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        check_disk_space(0.05, str(dst.parent))
    cmd = ["aws", "s3", "cp", src, str(dst), "--no-sign-request"]
    if recursive:
        cmd.insert(-1, "--recursive")
    aws_run(cmd)


def aws_s3_ls(uri: str) -> str:
    p = aws_run(["aws", "s3", "ls", uri, "--no-sign-request"])
    return p.stdout


def ensure_aws_cli() -> None:
    try:
        r = subprocess.run(
            ["aws", "--version"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError("aws failed")
        print(f"[0.2] AWS CLI: {r.stdout.strip() or r.stderr.strip()}")
    except FileNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "awscli", "--quiet"])
        r = subprocess.run(["aws", "--version"], capture_output=True, text=True)
        if r.returncode != 0:
            print(
                "AWS CLI not found. Install AWS CLI v2 from "
                "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
            )
            sys.exit(1)
        print(f"[0.2] AWS CLI (after pip): {r.stdout.strip()}")


def verify_s3_access() -> None:
    print("[0.3] Testing public S3 access...")
    try:
        aws_s3_ls(f"{S3_BUCKET}/")
        print("[0.3] OK: bucket listable.")
    except subprocess.CalledProcessError as e:
        print(f"[0.3] FAILED: {e.stderr or e.stdout}")
        sys.exit(1)


def gate_data_agreement(non_interactive: bool = False) -> None:
    if non_interactive:
        print(
            "[non-interactive] Skipping agreement prompt. "
            "You must have completed the NSD Data Access Agreement: "
            "https://forms.gle/eT4jHxaWwYUDEf2i9"
        )
        return
    banner = """
╔════════════════════════════════════════════════════════════════╗
║  REQUIRED: NSD Data Access Agreement                          ║
║                                                               ║
║  Before downloading NSD data you must complete the form at:   ║
║  https://forms.gle/eT4jHxaWwYUDEf2i9                         ║
║                                                               ║
║  Type  YES  to confirm you have completed it.                 ║
║  Type  NO   to abort.                                         ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    ans = input("Your response: ").strip().upper()
    if ans != "YES":
        print("Aborting (agreement not confirmed).")
        sys.exit(1)


def gate_subjects(non_interactive: bool = False) -> None:
    if non_interactive:
        print(
            "[non-interactive] Processing all 8 subjects (subj01–subj08) without prompt."
        )
        return
    print(
        'This pipeline will process all 8 NSD subjects (subj01–subj08) '
        "for cross-subject overlap analysis. Subjects with incomplete "
        "sessions will be flagged but not skipped.\nConfirm? (YES / NO)"
    )
    ans = input("Your response: ").strip().upper()
    if ans != "YES":
        print("Aborting.")
        sys.exit(1)


def create_directories() -> None:
    for p in [
        PREP,
        PREP / "qc",
        PREP / "surfaces",
        *[PREP / f"subj{aa:02d}" for aa in range(1, 9)],
        TMP,
        TMP / "atlases" / "labels",
        *[TMP / "atlases" / f"subj{aa:02d}" for aa in range(1, 9)],
        TMP / "surfaces" / "fsaverage" / "labels",
        TMP / "surfaces" / "subj01",
        TMP / "ncsnr",
    ]:
        p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Part 2 — Experimental design
# =============================================================================
def load_expdesign(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = loadmat(str(mat_path))
    subjectim = np.asarray(mat["subjectim"], dtype=np.int64)
    masterordering = np.asarray(mat["masterordering"]).ravel()
    sharedix = np.asarray(mat["sharedix"]).ravel()
    print("subjectim shape:", subjectim.shape, "first 5:", subjectim[0, :5])
    print(
        "masterordering shape:",
        masterordering.shape,
        "first 5:",
        masterordering[:5],
    )
    print("sharedix shape:", sharedix.shape, "first 5:", sharedix[:5])
    return subjectim, masterordering, sharedix


def load_stim_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("nsd_stim_info_merged shape:", df.shape)
    print("columns:", list(df.columns))
    return df


def build_mappings(
    subjectim: np.ndarray, masterordering: np.ndarray
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]], Dict[int, List[int]]]:
    subject_73k_ids: Dict[int, np.ndarray] = {}
    global_to_local: Dict[int, Dict[int, int]] = {}
    trial_local_idx: Dict[int, List[int]] = {aa: [] for aa in range(1, 9)}
    mo = masterordering.astype(np.int64)
    for aa in range(1, 9):
        sid = subjectim[aa - 1, :].astype(np.int64) - 1
        subject_73k_ids[aa] = sid
        global_to_local[aa] = {int(v): i for i, v in enumerate(sid)}
    for t in range(30000):
        local_10k_idx = int(mo[t]) - 1
        for aa in range(1, 9):
            trial_local_idx[aa].append(local_10k_idx)
    return subject_73k_ids, global_to_local, trial_local_idx


# =============================================================================
# Part 3 — Repetition & overlap
# =============================================================================
def count_reps(
    trial_local_idx: Dict[int, List[int]],
) -> Dict[int, np.ndarray]:
    rep_count: Dict[int, np.ndarray] = {}
    for aa in range(1, 9):
        arr = np.zeros(10000, dtype=np.int32)
        for li in trial_local_idx[aa]:
            arr[li] += 1
        rep_count[aa] = arr
    return rep_count


def reps_from_csv_row(row: pd.Series, subject_idx: int) -> int:
    cols = [f"subject{subject_idx}_rep{c}" for c in range(3)]
    return sum(1 for c in cols if c in row and row[c] > 0)


def assert_csv_agreement(
    rep_count: Dict[int, np.ndarray],
    stim_df: pd.DataFrame,
    subject_73k_ids: Dict[int, np.ndarray],
    rng: np.random.Generator,
) -> None:
    for _ in range(100):
        aa = int(rng.integers(1, 9))
        i = int(rng.integers(0, 10000))
        g73k = int(subject_73k_ids[aa][i])
        rows = stim_df.loc[stim_df["nsdId"] == g73k]
        if len(rows) != 1:
            raise AssertionError(f"nsdId {g73k} rows {len(rows)}")
        row = rows.iloc[0]
        r_csv = reps_from_csv_row(row, aa)
        assert int(rep_count[aa][i]) == r_csv, (aa, i, rep_count[aa][i], r_csv)
    print("[3.1] Random sample of 100: rep_count matches CSV.")


def overlap_analysis(
    rep_count: Dict[int, np.ndarray], subject_73k_ids: Dict[int, np.ndarray]
) -> Tuple[Dict[int, int], Dict[int, set]]:
    n_overlap: Dict[int, int] = {}
    overlap_sets: Dict[int, set] = {}
    for T in (1, 2, 3):
        subj_sets = []
        for aa in range(1, 9):
            qual_local = np.where(rep_count[aa] >= T)[0]
            qual_73k = set(int(subject_73k_ids[aa][j]) for j in qual_local)
            subj_sets.append(qual_73k)
        inter = set.intersection(*subj_sets)
        overlap_sets[T] = inter
        n_overlap[T] = len(inter)
    return n_overlap, overlap_sets


def print_overlap_table(
    rep_count: Dict[int, np.ndarray],
    subject_73k_ids: Dict[int, np.ndarray],
    n_overlap: Dict[int, int],
) -> None:
    rows = []
    for T in (1, 2, 3):
        row = [f">= {T}"]
        for aa in range(1, 9):
            n = int(np.sum(rep_count[aa] >= T))
            row.append(n)
        row.append(n_overlap[T])
        rows.append(row)
    header = (
        "┌──────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────────────────┐\n"
        "│  Min repeats │ Subj01 │ Subj02 │ Subj03 │ Subj04 │ Subj05 │ Subj06 │ Subj07 │ Subj08 │ All-subject overlap│\n"
        "├──────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────────────────┤"
    )
    print(header)
    for r in rows:
        print(
            f"│    {r[0]:4s}      │ {r[1]:6d} │ {r[2]:6d} │ {r[3]:6d} │ {r[4]:6d} │ "
            f"{r[5]:6d} │ {r[6]:6d} │ {r[7]:6d} │ {r[8]:6d} │ {r[9]:18d} │"
        )
    print(
        "└──────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────────────────┘"
    )


def sessions_from_stim(stim_df: pd.DataFrame, aa: int) -> int:
    cols = [f"subject{aa}_rep{c}" for c in range(3)]
    mx = 0
    for c in cols:
        if c in stim_df.columns:
            mx = max(mx, int(stim_df[c].max()))
    if mx <= 0:
        return 0
    return int(np.ceil(mx / 750.0))


def plot_rep_distribution(
    rep_count: Dict[int, np.ndarray], stim_df: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    colors = ["#888888", "#A8D8EA", "#4A90D9", "#1E3A5F"]
    for idx, aa in enumerate(range(1, 9)):
        ax = axes[idx // 4, idx % 4]
        rc = rep_count[aa]
        counts = [int(np.sum(rc == k)) for k in range(4)]
        x = np.arange(4)
        bars = ax.bar(x, counts, color=colors)
        for xi, b in zip(x, bars):
            ax.text(
                xi,
                b.get_height(),
                str(counts[xi]),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ns = sessions_from_stim(stim_df, aa)
        ax.set_title(f"subj{aa:02d}  ({ns} sessions completed)")
        ax.set_xlabel("Number of repetitions")
        ax.set_ylabel("Number of images")
        ax.set_xticks(x)
    plt.suptitle("Repetition distribution by subject")
    plt.tight_layout()
    out = PREP / "qc" / "repetition_distribution_by_subject.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_cross_subject_overlap_bars(
    rep_count: Dict[int, np.ndarray],
    subject_73k_ids: Dict[int, np.ndarray],
    n_overlap: Dict[int, int],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    n_groups = 9
    x = np.arange(n_groups, dtype=float)
    width = 0.25
    blues = ["#A8D8EA", "#4A90D9", "#1E3A5F"]
    labels_leg = [">=1 rep", ">=2 reps", ">=3 reps"]
    for ti, T in enumerate((1, 2, 3)):
        vals = []
        for aa in range(1, 9):
            vals.append(int(np.sum(rep_count[aa] >= T)))
        vals.append(int(n_overlap[T]))
        ax.bar(
            x + (ti - 1) * width,
            vals,
            width,
            color=blues[ti],
            label=labels_leg[ti],
        )
    for T in (1, 2, 3):
        ax.axhline(
            n_overlap[T],
            color="k",
            linestyle="-",
            linewidth=2.5,
            alpha=0.35,
            zorder=0,
        )
    labels = [f"subj{aa:02d}" for aa in range(1, 9)] + ["All Subjects"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Cross-subject overlap (last group: all-subject 73k intersection)")
    plt.tight_layout()
    out = PREP / "qc" / "cross_subject_overlap.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def gate_min_reps(
    n_overlap: Dict[int, int],
    print_table_fn: Callable[[], None],
    non_interactive: bool = False,
) -> int:
    if non_interactive:
        print("[non-interactive] Using MIN_REPS=3 (recommended).")
        return 3
    print(
        """
╔════════════════════════════════════════════════════════════════╗
║  REPETITION THRESHOLD SELECTION                               ║
║                                                               ║
║  Summary table (repeated from 3.2 above):                    ║
╚════════════════════════════════════════════════════════════════╝
"""
    )
    print_table_fn()
    print(
        """
║  Recommendation: >= 3 repetitions                            ║
║  (maximises signal quality; use averaged repeats to reduce    ║
║   noise — see noise ceiling formula in Part 8B)               ║
║                                                               ║
║  Enter minimum repetition threshold (integer >= 1):           ║
"""
    )
    s = input("MIN_REPS: ").strip()
    try:
        m = int(s)
    except ValueError:
        m = 3
    if m > 3:
        print("Warning: no image exceeds 3 reps in NSD; capping at 3.")
    if m < 1:
        print("Invalid MIN_REPS; defaulting to 3.")
        m = 3
    m = max(1, min(3, m))
    return m


def finalize_image_set(
    rep_count: Dict[int, np.ndarray],
    subject_73k_ids: Dict[int, np.ndarray],
    global_to_local: Dict[int, Dict[int, int]],
    min_reps: int,
) -> Tuple[List[int], Dict[int, List[int]], Dict[int, int]]:
    # Images in ALL 8 subjects' 10k with rep >= min_reps
    cand: Optional[set] = None
    for aa in range(1, 9):
        loc = np.where(rep_count[aa] >= min_reps)[0]
        s73 = set(int(subject_73k_ids[aa][j]) for j in loc)
        cand = s73 if cand is None else cand & s73
    assert cand is not None
    final_list = sorted(cand)
    final_set_pos = {g: p for p, g in enumerate(final_list)}
    final_local_idx: Dict[int, List[int]] = {}
    for aa in range(1, 9):
        g2l = global_to_local[aa]
        final_local_idx[aa] = [g2l[g] for g in final_list]
    n_final = len(final_list)
    print(
        f"Final image set: {n_final} images (0-based 73k IDs) shared across all 8 "
        f"subjects with >= {min_reps} repetitions."
    )
    np.savez(
        PREP / "final_image_set.npz",
        final_image_set_73k=np.array(final_list, dtype=np.int64),
        min_reps=np.int32(min_reps),
        **{
            f"final_local_idx_subj{aa:02d}": np.array(
                final_local_idx[aa], dtype=np.int64
            )
            for aa in range(1, 9)
        },
    )
    return final_list, final_local_idx, final_set_pos


# =============================================================================
# Part 4 — Atlases & ROI
# =============================================================================
ATLAS_CANDIDATES = [
    "prf-visualrois",
    "prf-eccrois",
    "Kastner2015",
    "HCP_MMP1",
    "streams",
    "nsdgeneral",
    "floc-faces",
    "floc-places",
    "floc-bodies",
]


def download_subj01_atlases() -> None:
    base = f"{S3_BUCKET}/nsddata/ppdata/subj01/func1pt8mm/roi/"
    outd = TMP / "atlases" / "subj01"
    for name in ATLAS_CANDIDATES:
        aws_s3_cp(f"{base}{name}.nii.gz", outd / f"{name}.nii.gz")
    lbl_base = f"{S3_BUCKET}/nsddata/freesurfer/subj01/label/"
    for stem in [
        "prf-visualrois",
        "prf-eccrois",
        "Kastner2015",
        "HCP_MMP1",
        "streams",
    ]:
        aws_s3_cp(
            f"{lbl_base}{stem}{LABEL_LOOKUP_SUFFIX}",
            TMP / "atlases" / "labels" / f"{stem}{LABEL_LOOKUP_SUFFIX}",
        )


def parse_label_txt(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s\t]+", line, maxsplit=2)
        if len(parts) >= 2 and parts[0].isdigit():
            idx = int(parts[0])
            name = parts[1] if len(parts) == 2 else parts[2]
            out[idx] = name.strip()
    return out


def print_atlas_label_table(atlas_stem: str, nii_path: Path, label_map: Dict[int, str]) -> None:
    vol = nib.load(str(nii_path)).get_fdata().astype(np.int32)
    flat = vol.ravel()
    shape = vol.shape
    print("══════════════════════════════════════════════════════")
    print(f"ATLAS: {atlas_stem}  |  Volume shape: {shape}")
    print("══════════════════════════════════════════════════════")
    print("Label │ Name              │ Voxel count")
    print("──────┼───────────────────┼────────────")
    all_labels = sorted(set(flat.tolist()) | set(label_map.keys()))
    for lab in all_labels:
        cnt = int(np.sum(flat == lab))
        name = label_map.get(lab, "(unknown)")
        if lab == 0:
            name = "(unlabeled)"
        print(f"{lab:5d} │ {name[:17]:17s} │ {cnt:10d}")


def infer_default_it_streams(labels_path: Path) -> Tuple[str, List[int], List[str]]:
    """Prefer ventral / temporal stream labels from streams.mgz.ctab; fallback Kastner VO/PHC names."""
    lm = parse_label_txt(labels_path)
    picked: List[int] = []
    names: List[str] = []
    for k, name in sorted(lm.items()):
        nl = name.lower()
        if any(
            x in nl
            for x in ("ventral", "ventralstream", "ventral stream", "ventral_visual")
        ):
            picked.append(k)
            names.append(name)
    if not picked:
        for k, name in sorted(lm.items()):
            nl = name.lower()
            if "temporal" in nl and "dorsal" not in nl:
                picked.append(k)
                names.append(name)
    if picked:
        return "streams", picked, names
    # Fallback: Kastner2015 VO/PHC
    kpath = TMP / "atlases" / "labels" / f"Kastner2015{LABEL_LOOKUP_SUFFIX}"
    km = parse_label_txt(kpath)
    for key in ("VO1", "VO2", "PHC1", "PHC2"):
        for k, name in km.items():
            if key.lower() in name.lower().replace(" ", ""):
                picked.append(k)
                names.append(name)
                break
    if picked:
        return "Kastner2015", picked, names
    return "streams", [3], ["fallback_label_3"]


def default_roi_config(it_atlas: str, it_labels: List[int], it_names: List[str]) -> Dict[str, Any]:
    return {
        "V1": {
            "atlas": "prf-visualrois",
            "labels": [1, 2],
            "label_names": ["V1v", "V1d"],
        },
        "V2": {
            "atlas": "prf-visualrois",
            "labels": [3, 4],
            "label_names": ["V2v", "V2d"],
        },
        "V4": {
            "atlas": "prf-visualrois",
            "labels": [7],
            "label_names": ["hV4"],
        },
        "IT": {
            "atlas": it_atlas,
            "labels": it_labels,
            "label_names": it_names,
        },
    }


def gate_roi_config(
    default_cfg: Dict[str, Any], non_interactive: bool = False
) -> Dict[str, Any]:
    if non_interactive:
        cfg = json.loads(json.dumps(default_cfg))
        print("[non-interactive] Using default ROI_CONFIG (no prompts):")
        print(json.dumps(cfg, indent=2))
        return cfg
    print(
        """
╔═══════╦══════════════════╦══════════════════════════════════════════════╗
║  ROI  ║  Atlas           ║  Label integers  (names)                    ║
╠═══════╬══════════════════╬══════════════════════════════════════════════╣
║  V1   ║  prf-visualrois  ║  1 (V1v)  + 2 (V1d)      [DEFAULT]         ║
║  V2   ║  prf-visualrois  ║  3 (V2v)  + 4 (V2d)      [DEFAULT]         ║
║  V4   ║  prf-visualrois  ║  7 (hV4)                  [DEFAULT]         ║
║  IT   ║  (see below)     ║  (confirmed interactively)                  ║
╚═══════╩══════════════════╩══════════════════════════════════════════════╝
"""
    )
    cfg = json.loads(json.dumps(default_cfg))
    print("Confirm or modify the ROI mappings above.\nFor each ROI, press ENTER to accept the default, or type:\n  <ROI> <atlas_filename_without_.nii.gz> <label_int,...>\nExample:  IT streams 3,4\nType DONE when finished with all ROIs.")
    while True:
        line = input("ROI> ").strip()
        if line.upper() == "DONE":
            break
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            print("Expected: ROI atlas l1,l2,...")
            continue
        roi = parts[0].upper()
        if roi not in cfg:
            print("Unknown ROI", roi)
            continue
        atlas = parts[1]
        labs = [int(x) for x in ",".join(parts[2:]).split(",") if x.strip()]
        lbl_path = TMP / "atlases" / "labels" / f"{atlas}{LABEL_LOOKUP_SUFFIX}"
        lmap = parse_label_txt(lbl_path) if lbl_path.exists() else {}
        names = [lmap.get(li, f"L{li}") for li in labs]
        cfg[roi] = {"atlas": atlas, "labels": labs, "label_names": names}
    print("Final ROI_CONFIG:", json.dumps(cfg, indent=2))
    return cfg


# =============================================================================
# Part 5 — Stimuli
# =============================================================================
# Confirmed size on S3 (aws s3 ls .../nsd_stimuli.hdf5); used to skip re-download.
_NSD_STIMULI_HDF5_BYTES = 39556877048


def _aws_s3_download_bytes_on_disk(dst: Path) -> int:
    """AWS CLI writes to `dst.<random>` then renames to `dst`; count both."""
    total = 0
    if dst.exists():
        total += dst.stat().st_size
    for p in dst.parent.glob(f"{dst.name}.*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _is_complete_stimulus_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return path.stat().st_size >= _NSD_STIMULI_HDF5_BYTES * 0.999


def _recover_complete_stimulus_file(dst: Path) -> bool:
    """
    If aws left a fully downloaded temp file like `dst.<random>`, recover it.
    Returns True when a complete file is available at `dst` after this call.
    """
    if _is_complete_stimulus_file(dst):
        return True
    for p in sorted(dst.parent.glob(f"{dst.name}.*")):
        if p.is_file() and _is_complete_stimulus_file(p):
            print(f"[5.3] Found complete download in temp file {p.name}; moving to {dst.name}")
            if dst.exists():
                dst.unlink()
            p.rename(dst)
            return True
    return False


def download_stimuli_progress(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _recover_complete_stimulus_file(dst):
        got = dst.stat().st_size
        print(f"[5.3] Reusing complete local stimulus file ({got / 1e9:.2f} GB).")
        return
    # Resume / cleanup after Ctrl+C or failed run: remove AWS partial `dst.*` files
    # (a new `aws s3 cp` does not resume into them; leaving them wastes disk).
    for p in sorted(dst.parent.glob(f"{dst.name}.*")):
        if p.is_file():
            print(f"[5.3] Removing incomplete partial download: {p.name}")
            p.unlink()
    if dst.exists():
        got = dst.stat().st_size
        if got >= _NSD_STIMULI_HDF5_BYTES * 0.999:
            print(
                f"[5.3] Stimulus already on disk ({got / 1e9:.2f} GB); skipping download."
            )
            return
        if got > 0:
            print(
                f"[5.3] Removing incomplete file {dst.name} ({got / 1e9:.2f} GB); "
                "re-downloading from scratch."
            )
            dst.unlink()
    check_disk_space(45.0)
    err_log = dst.parent / "aws_s3cp_stimuli.log"
    # Full-file retries: S3 cp does not resume a partial file; each attempt restarts from 0.
    max_attempts = 8
    print(
        "[5.3] Unstable internet: use Ethernet if you can; run inside tmux/screen so "
        "closing the laptop does not kill Python. Up to "
        f"{max_attempts} full retries with backoff; AWS CLI also uses adaptive retries."
    )
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            wait = min(30 * (2 ** (attempt - 2)), 600)
            print(
                f"[5.3] Retry {attempt}/{max_attempts} after failure "
                f"(waiting {wait}s, see {err_log.name})…"
            )
            time.sleep(wait)
            for p in sorted(dst.parent.glob(f"{dst.name}.*")):
                if p.is_file():
                    p.unlink()
            if dst.exists():
                dst.unlink()
        print(
            f"[5.3] Starting download (~40 GB). While in progress, AWS may write "
            f"`{dst.name}.<random>`; then rename to `{dst.name}`.\n"
            f"      Watch: ls -lh {dst.parent}/{dst.name}*\n"
            f"      If this fails mid-way, see: {err_log}"
        )
        # --cli-read-timeout 0: no stall limit between chunks (default ~60s can kill 40GB pulls).
        aws_cmd = [
            "aws",
            "s3",
            "cp",
            src,
            str(dst),
            "--no-sign-request",
            "--cli-read-timeout",
            "0",
            "--cli-connect-timeout",
            "60",
        ]
        with open(err_log, "w", encoding="utf-8") as errf:
            proc = subprocess.Popen(
                aws_cmd,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                env=_aws_child_environ(),
            )
        t0 = time.time()
        last = t0
        while proc.poll() is None:
            time.sleep(10)
            now = time.time()
            if now - last >= 300:
                sz = _aws_s3_download_bytes_on_disk(dst) / 1e9
                hint = ""
                if sz < 1e-6 and now - t0 > 120:
                    hint = (
                        " (still 0 B — check network/VPN, or run: "
                        f"aws s3 ls {S3_BUCKET}/nsddata_stimuli/stimuli/nsd/ --no-sign-request)"
                    )
                print(
                    f"[5min progress] {dst.name}: {sz:.2f} GB on disk | "
                    f"elapsed {now - t0:.0f}s{hint}"
                )
                last = now
        rc = proc.wait()
        if rc == 0:
            break
        tail = ""
        try:
            txt = err_log.read_text(encoding="utf-8", errors="replace")
            tail = txt.strip()[-4000:]
        except OSError:
            pass
        log_error(f"aws s3 cp stimuli failed (attempt {attempt}, exit {rc})\n{tail}")
        print(
            f"[5.3] aws s3 cp exited with code {rc}. Last stderr lines:\n{tail}\n"
        )
        if attempt == max_attempts:
            raise RuntimeError(
                f"aws s3 cp stimuli failed after {max_attempts} attempts (exit {rc}). "
                f"Full stderr: {err_log}"
            )
    sz_gb = dst.stat().st_size / 1e9
    print(f"Download complete. Size: {sz_gb:.2f} GB")


def extract_stimuli_224(
    final_list: List[int],
    min_reps: int,
    n_final: int,
) -> None:
    out_path = PREP / "nsd_stimuli_224.hdf5"
    raw_path = TMP / "nsd_stimuli_raw.hdf5"
    if out_path.exists():
        with h5py.File(out_path, "r") as f:
            if f["/images"].shape[0] == n_final:
                print(f"[5.4] Using existing {out_path}")
                return
    with h5py.File(raw_path, "r") as f_in:
        brick = f_in["/imgBrick"]
        with h5py.File(out_path, "w") as f_out:
            d = f_out.create_dataset(
                "/images",
                shape=(n_final, 224, 224, 3),
                dtype=np.uint8,
                chunks=(1, 224, 224, 3),
                compression="gzip",
                compression_opts=4,
            )
            d.attrs["source"] = "NSD nsd_stimuli.hdf5 /imgBrick"
            d.attrs["resize"] = "224x224"
            d.attrs["color_space"] = "RGB"
            d.attrs["index_base"] = "0-based 73k IDs"
            d.attrs["min_reps"] = min_reps
            d.attrs["n_subjects"] = 8
            d.attrs["n_images"] = n_final
            gi = f_out.create_dataset(
                "/global_image_indices_73k",
                data=np.array(final_list, dtype=np.int32),
            )
            gi.attrs["index_base"] = "0-based"
            for batch_start in range(0, n_final, 500):
                batch = final_list[batch_start : batch_start + 500]
                for pos, g73k in enumerate(batch):
                    raw = brick[:, :, :, g73k]
                    img = np.asarray(raw).transpose(1, 2, 0)
                    im = Image.fromarray(img)
                    im = im.resize((224, 224), Image.LANCZOS)
                    d[batch_start + pos] = np.asarray(im, dtype=np.uint8)
                print(f"Processed {batch_start + len(batch)}/{n_final} images")
    # verify
    with h5py.File(out_path, "r") as f:
        assert f["/images"].shape == (n_final, 224, 224, 3)
        assert f["/images"].dtype == np.uint8
        rng = np.random.default_rng(0)
        for _ in range(10):
            i = int(rng.integers(0, n_final))
            assert f["/images"][i].min() >= 0 and f["/images"][i].max() <= 255
    print(f"nsd_stimuli_224.hdf5 size: {out_path.stat().st_size / 1e9:.2f} GB")
    if raw_path.exists():
        os.remove(raw_path)
        print("Removed raw stimuli HDF5 from nsd_tmp.")


# =============================================================================
# Part 6 — Masks
# =============================================================================
def download_atlases_all_subjects(roi_cfg: Dict[str, Any]) -> None:
    atlases = {roi_cfg[r]["atlas"] for r in ROI_ORDER}
    for aa in range(1, 9):
        sub = f"subj{aa:02d}"
        base = f"{S3_BUCKET}/nsddata/ppdata/{sub}/func1pt8mm/roi/"
        outd = TMP / "atlases" / sub
        for at in sorted(atlases):
            aws_s3_cp(f"{base}{at}.nii.gz", outd / f"{at}.nii.gz")
        # prf-visualrois for V3 viz
        if "prf-visualrois" not in atlases:
            aws_s3_cp(f"{base}prf-visualrois.nii.gz", outd / "prf-visualrois.nii.gz")


def build_masks(
    roi_cfg: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, int]], Dict[int, np.ndarray]]:
    mask_3d: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
    mask_flat: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
    voxel_count: Dict[int, Dict[str, int]] = defaultdict(dict)
    v3_mask: Dict[int, np.ndarray] = {}
    for aa in range(1, 9):
        sub = f"subj{aa:02d}"
        for roi in ROI_ORDER:
            atlas = roi_cfg[roi]["atlas"]
            path = TMP / "atlases" / sub / f"{atlas}.nii.gz"
            vol = nib.load(str(path)).get_fdata().astype(np.int32)
            m = np.isin(vol, roi_cfg[roi]["labels"])
            mask_3d[aa][roi] = m
            mask_flat[aa][roi] = m.ravel()
            c = int(m.sum())
            voxel_count[aa][roi] = c
            if c == 0:
                print(f"WARNING: {sub} {roi} has 0 voxels.")
        ppath = TMP / "atlases" / sub / "prf-visualrois.nii.gz"
        pv = nib.load(str(ppath)).get_fdata().astype(np.int32)
        v3_mask[aa] = np.isin(pv, [5, 6])
    return mask_3d, mask_flat, voxel_count, v3_mask


def print_voxel_table(voxel_count: Dict[int, Dict[str, int]]) -> None:
    print(
        "┌─────────┬───────────┬───────────┬───────────┬───────────┐\n"
        "│ Subject │ V1 voxels │ V2 voxels │ V4 voxels │ IT voxels │\n"
        "├─────────┼───────────┼───────────┼───────────┼───────────┤"
    )
    means = {r: [] for r in ROI_ORDER}
    for aa in range(1, 9):
        row = [f"subj{aa:02d}"]
        for roi in ROI_ORDER:
            v = voxel_count[aa][roi]
            means[roi].append(v)
            row.append(f"{v:8d}")
        print("│ " + " │ ".join(row) + " │")
    print("├─────────┼───────────┼───────────┼───────────┼───────────┤")
    mr = ["Mean   "]
    for roi in ROI_ORDER:
        mr.append(f"{int(np.mean(means[roi])):8d}")
    print("│ " + " │ ".join(mr) + " │")
    print(
        "└─────────┴───────────┴───────────┴───────────┴───────────┘"
    )


# =============================================================================
# Part 7 — Neural
# =============================================================================
def list_beta_sessions(aa: int) -> List[int]:
    sub = f"subj{aa:02d}"
    uri = f"{S3_BUCKET}/nsddata_betas/ppdata/{sub}/func1pt8mm/betas_fithrf/"
    out = aws_s3_ls(uri)
    sessions: List[int] = []
    for line in out.splitlines():
        m = re.search(r"betas_session(\d+)\.hdf5", line)
        if m:
            sessions.append(int(m.group(1)))
    sessions.sort()
    return sessions


def create_neural_h5(
    aa: int,
    roi_cfg: Dict[str, Any],
    voxel_count: Dict[int, Dict[str, int]],
    n_final: int,
    min_reps: int,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    sub = f"subj{aa:02d}"
    odir = PREP / sub
    odir.mkdir(parents=True, exist_ok=True)
    for roi in ROI_ORDER:
        p = odir / f"nsd_neural_{roi}.hdf5"
        nv = voxel_count[aa][roi]
        if p.exists():
            with h5py.File(p, "r") as f:
                if f["/betas"].shape == (n_final, min_reps, nv):
                    out[roi] = p
                    continue
            p.unlink()
        with h5py.File(p, "w") as f:
            d = f.create_dataset(
                "/betas",
                shape=(n_final, min_reps, nv),
                dtype=np.float32,
                chunks=(1, min_reps, nv),
                fillvalue=np.nan,
            )
            d.attrs["subject"] = sub
            d.attrs["roi"] = roi
            d.attrs["atlas"] = roi_cfg[roi]["atlas"]
            d.attrs["label_values"] = str(roi_cfg[roi]["labels"])
            d.attrs["label_names"] = str(roi_cfg[roi]["label_names"])
            d.attrs["beta_version"] = "betas_fithrf"
            d.attrs["space"] = "func1pt8mm"
            d.attrs["units"] = "percent_signal_change"
            d.attrs["n_images"] = n_final
            d.attrs["n_repeats"] = min_reps
            d.attrs["missing"] = "NaN"
        out[roi] = p
    return out


def load_repeat_counter_from_h5(
    paths: Dict[str, Path], n_final: int, min_reps: int
) -> np.ndarray:
    """How many repeat slots are filled per final image (from first ROI file)."""
    roi = ROI_ORDER[0]
    with h5py.File(paths[roi], "r") as f:
        b = f["/betas"][:]
    ctr = np.zeros(n_final, dtype=np.int32)
    for i in range(n_final):
        filled = 0
        for r in range(min_reps):
            if np.all(np.isnan(b[i, r, :])):
                break
            filled += 1
        ctr[i] = filled
    return ctr


def process_subject_sessions(
    aa: int,
    trial_local_idx: Dict[int, List[int]],
    subject_73k_ids: Dict[int, np.ndarray],
    final_set_pos: Dict[int, int],
    mask_3d: Dict[int, Dict[str, np.ndarray]],
    out_paths: Dict[str, Path],
    n_final: int,
    min_reps: int,
    n_sessions: int,
    session_list: List[int],
) -> None:
    repeat_counter = np.zeros(n_final, dtype=np.int32)
    if all(p.exists() for p in out_paths.values()):
        repeat_counter = load_repeat_counter_from_h5(out_paths, n_final, min_reps)
        print(
            f"[subj{aa:02d}] Resuming: "
            f"{int(np.sum(repeat_counter >= min_reps))}/{n_final} images complete."
        )

    failed: List[int] = []

    def run_session(BB: int) -> None:
        nonlocal repeat_counter
        if int(np.sum(repeat_counter >= min_reps)) == n_final:
            print(f"[subj{aa:02d}] All images complete. Skipping session {BB}.")
            return
        check_disk_space(4.0)
        src = (
            f"{S3_BUCKET}/nsddata_betas/ppdata/subj{aa:02d}/"
            f"func1pt8mm/betas_fithrf/betas_session{BB:02d}.hdf5"
        )
        tmp = TMP / "betas_tmp.hdf5"
        try:
            aws_s3_cp(src, tmp)
        except Exception as e:
            log_error(f"subj{aa:02d} session {BB} download", e)
            failed.append(BB)
            return
        with h5py.File(tmp, "r") as f:
            betas_int16 = f["/betas"][:]
        betas = betas_int16.astype(np.float32) / 300.0
        del betas_int16
        gt0 = (BB - 1) * 750
        sess_idx = trial_local_idx[aa][gt0 : gt0 + 750]
        fds = {
            roi: h5py.File(out_paths[roi], "r+") for roi in ROI_ORDER
        }
        try:
            for t in range(750):
                local_10k = sess_idx[t]
                g73k = int(subject_73k_ids[aa][local_10k])
                if g73k not in final_set_pos:
                    continue
                fp = final_set_pos[g73k]
                rs = int(repeat_counter[fp])
                if rs >= min_reps:
                    continue
                for roi in ROI_ORDER:
                    m = mask_3d[aa][roi]
                    vec = betas[m, t]
                    fds[roi]["/betas"][fp, rs, :] = vec
                repeat_counter[fp] += 1
        finally:
            for f in fds.values():
                f.close()
        if tmp.exists():
            os.remove(tmp)
        n_done = int(np.sum(repeat_counter >= min_reps))
        free = shutil.disk_usage(".").free / 1e9
        print(
            f"[subj{aa:02d}] Session {BB:02d}/{n_sessions:02d} done | "
            f"Images fully filled: {n_done}/{n_final} | Free disk: {free:.1f} GB"
        )

    for BB in session_list:
        try:
            run_session(BB)
        except Exception as e:
            log_error(f"subj{aa:02d} session {BB} process", e)
            failed.append(BB)
    for BB in list(failed):
        try:
            run_session(BB)
        except Exception as e:
            log_error(f"subj{aa:02d} session {BB} retry", e)


def verify_neural(
    aa: int, out_paths: Dict[str, Path], n_final: int, min_reps: int
) -> None:
    for roi in ROI_ORDER:
        with h5py.File(out_paths[roi], "r") as f:
            b = f["/betas"][:]
        n_complete = int(np.sum(~np.isnan(b[:, -1, 0])))
        n_any_nan = int(np.sum(np.any(np.isnan(b), axis=(1, 2))))
        nan_frac = float(np.isnan(b).mean())
        print(
            f"  subj{aa:02d} {roi}: complete(last rep)={n_complete}/{n_final} "
            f"any_nan_rows={n_any_nan} nan_frac={nan_frac:.4f}"
        )
        if nan_frac > 0.05:
            print(f"WARNING: nan_frac > 0.05 for subj{aa:02d} {roi}")


# =============================================================================
# Part 8 — QC figures
# =============================================================================
def qc_example_stimuli(
    qc_indices_in_final: np.ndarray,
    qc_73k_ids: List[int],
) -> None:
    path = PREP / "nsd_stimuli_224.hdf5"
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    with h5py.File(path, "r") as f:
        imgs = f["/images"]
        for i in range(10):
            ax = axes[i // 5, i % 5]
            ax.imshow(imgs[qc_indices_in_final[i]])
            ax.set_title(
                f"73k-ID={qc_73k_ids[i]}\nfinal_pos={qc_indices_in_final[i]}",
                fontsize=8,
            )
            ax.axis("off")
    plt.suptitle("10 QC Example Images (224×224, final image set)")
    plt.tight_layout()
    out = PREP / "qc" / "example_images_stimuli.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def qc_example_responses(
    qc_indices_in_final: np.ndarray,
    min_reps: int,
) -> None:
    pdir = PREP / "subj01"
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    stim = h5py.File(PREP / "nsd_stimuli_224.hdf5", "r")
    roi_files = {r: h5py.File(pdir / f"nsd_neural_{r}.hdf5", "r") for r in ROI_ORDER}
    try:
        for i in range(10):
            ax = axes[i // 5, i % 5]
            fi = int(qc_indices_in_final[i])
            ax.imshow(stim["/images"][fi])
            lines = []
            for roi in ROI_ORDER:
                b = roi_files[roi]["/betas"][fi, :min_reps, :]
                means = np.nanmean(b, axis=1)
                m = float(np.nanmean(means))
                s = float(np.nanstd(means)) if min_reps > 1 else 0.0
                lines.append(f"{roi}: {m:.2f}±{s:.2f}%")
            txt = "\n".join(lines)
            ax.text(
                0.02,
                0.98,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.55, pad=2),
            )
            ax.axis("off")
        plt.suptitle("QC: 10 Example Images with Mean Beta per ROI (subj01)")
        plt.tight_layout()
        out = PREP / "qc" / "example_images_with_responses.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")
    finally:
        stim.close()
        for f in roi_files.values():
            f.close()


def qc_inter_repeat(
    final_list: List[int], min_reps: int, _final_set_pos: Dict[int, int]
) -> None:
    rng = np.random.default_rng(99)
    pick = rng.choice(len(final_list), size=min(5, len(final_list)), replace=False)
    path = PREP / "subj01"
    for pi in pick:
        g73k = int(final_list[pi])
        print(f"\n--- Image 73k-ID {g73k} (final pos {pi}) ---")
        for roi in ROI_ORDER:
            with h5py.File(path / f"nsd_neural_{roi}.hdf5", "r") as f:
                b = np.asarray(f["/betas"][pi, :min_reps, :])
            vecs = [b[r].ravel() for r in range(min_reps) if not np.all(np.isnan(b[r]))]
            pairs = [(0, 1), (0, 2), (1, 2)]
            rs: List[float] = []
            for a, c in pairs:
                if a < len(vecs) and c < len(vecs):
                    r, _ = stats.pearsonr(vecs[a], vecs[c])
                    rs.append(float(r))
                else:
                    rs.append(float("nan"))
            flat = b[~np.isnan(b)]
            mb = float(np.nanmean(flat)) if flat.size else float("nan")
            sb = float(np.nanstd(flat)) if flat.size else float("nan")
            print(
                f"  {roi}: r12={rs[0]:.3f} r13={rs[1]:.3f} r23={rs[2]:.3f} "
                f"mean_beta={mb:.3f} std_beta={sb:.3f}"
            )


def qc_final_rep_distribution(
    min_reps: int, n_final: int, all_out: Dict[int, Dict[str, Path]]
) -> None:
    fig, axes = plt.subplots(4, 8, figsize=(22, 14))
    for ri, roi in enumerate(ROI_ORDER):
        for sj in range(1, 9):
            ax = axes[ri, sj - 1]
            p = all_out[sj][roi]
            with h5py.File(p, "r") as f:
                b = f["/betas"][:]
            counts = np.zeros(min_reps + 1, dtype=np.int32)
            for i in range(n_final):
                nfill = 0
                for r in range(min_reps):
                    if np.all(np.isnan(b[i, r, :])):
                        break
                    nfill += 1
                counts[nfill] += 1
            x = np.arange(min_reps + 1)
            ax.bar(x, counts, color=ROI_COLORS_QC[roi], alpha=0.75)
            frac = counts[min_reps] / max(1, n_final)
            ax.set_title(f"{roi} subj{sj:02d}\ncomplete={frac:.2f}", fontsize=7)
            if ri == 3:
                ax.set_xlabel("repeats filled")
            if sj == 1:
                ax.set_ylabel("n images")
    plt.suptitle("Final repetition distribution (actual HDF5 fills)")
    plt.tight_layout()
    out = PREP / "qc" / "final_repetition_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# Part 8B — Surfaces & noise ceiling
# =============================================================================
def download_surface_assets(it_atlas: str) -> None:
    fs = TMP / "surfaces" / "fsaverage"
    s01 = TMP / "surfaces" / "subj01"
    for name in [
        "lh.inflated",
        "rh.inflated",
        "lh.sulc",
        "rh.sulc",
        "lh.curv",
        "rh.curv",
    ]:
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata/freesurfer/fsaverage/surf/{name}",
            fs / name,
        )
    labd = fs / "labels"
    for hemi in ("lh", "rh"):
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata/freesurfer/fsaverage/label/{hemi}.prf-visualrois.mgz",
            labd / f"{hemi}.prf-visualrois.mgz",
        )
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata/freesurfer/fsaverage/label/{hemi}.{it_atlas}.mgz",
            labd / f"{hemi}.{it_atlas}.mgz",
        )
    for name in ["lh.inflated", "rh.inflated", "lh.sulc", "rh.sulc"]:
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata/freesurfer/subj01/surf/{name}",
            s01 / name,
        )
    for hemi in ("lh", "rh"):
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata/freesurfer/subj01/label/{hemi}.prf-visualrois.mgz",
            s01 / f"{hemi}.prf-visualrois.mgz",
        )
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata/freesurfer/subj01/label/{hemi}.{it_atlas}.mgz",
            s01 / f"{hemi}.{it_atlas}.mgz",
        )
    check_disk_space(0.1)
    aws_s3_cp(
        f"{S3_BUCKET}/nsddata/inspections/rois/",
        PREP / "qc" / "reference_roi_images",
        recursive=True,
    )


def _nilearn_plotting():
    """Lazy import so a broken NumPy install does not block Parts 0–8."""
    from nilearn import plotting as nl_plotting

    return nl_plotting


def _build_surface_texture(
    hemi: str,
    roi_cfg: Dict[str, Any],
    base: Path,
) -> np.ndarray:
    prf = nib.load(str(base / f"{hemi}.prf-visualrois.mgz")).get_fdata().squeeze()
    it_name = roi_cfg["IT"]["atlas"]
    it = nib.load(str(base / f"{hemi}.{it_name}.mgz")).get_fdata().squeeze()
    tex = np.zeros(len(prf), dtype=np.int32)
    tex[np.isin(prf, [1, 2])] = 1
    tex[np.isin(prf, [3, 4])] = 2
    tex[np.isin(prf, [5, 6])] = 3
    tex[np.isin(prf, [7])] = 4
    tex[np.isin(it, roi_cfg["IT"]["labels"])] = 5
    return tex


def plot_roi_surfaces(roi_cfg: Dict[str, Any], native: bool) -> str:
    from matplotlib.colors import ListedColormap

    plotting = _nilearn_plotting()
    subdir = "subj01" if native else "fsaverage"
    base = TMP / "surfaces" / subdir
    lbl = base if native else (TMP / "surfaces" / "fsaverage" / "labels")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), subplot_kw={"projection": "3d"})
    views = [
        ("left", "lateral", axes[0, 0], "LH Lateral"),
        ("left", "medial", axes[0, 1], "LH Medial"),
        ("right", "lateral", axes[1, 0], "RH Lateral"),
        ("right", "medial", axes[1, 1], "RH Medial"),
    ]
    cmap = ListedColormap([SURF_ROI_COLORS[k] for k in sorted(SURF_ROI_COLORS)])
    for hemi, view, ax, title in views:
        pref = hemi[0]
        surf_mesh = str(base / f"{pref}h.inflated")
        bg_map = str(base / f"{pref}h.sulc")
        htex = _build_surface_texture(f"{pref}h", roi_cfg, lbl)
        plotting.plot_surf_roi(
            surf_mesh=surf_mesh,
            roi_map=htex,
            hemi=hemi,
            view=view,
            bg_map=bg_map,
            bg_on_data=True,
            darkness=0.6,
            cmap=cmap,
            axes=ax,
            title=title,
        )
    if native:
        st = "NSD Visual ROIs — subj01 Native Inflated Surface"
    else:
        st = (
            f"NSD Visual ROIs on fsaverage Inflated Surface\n"
            f"(prf-visualrois + {roi_cfg['IT']['atlas']})"
        )
    plt.suptitle(st, fontsize=14)
    plt.tight_layout()
    if native:
        outp = PREP / "surfaces" / "roi_surface_subj01_native_inflated.png"
    else:
        outp = PREP / "surfaces" / "roi_surface_fsaverage_inflated.png"
    fig.savefig(outp, dpi=120)
    plt.close(fig)
    return str(outp)


def roi_comparison_figure() -> None:
    ref_dir = PREP / "qc" / "reference_roi_images"
    candidates = list(ref_dir.rglob("*subj01*.png")) + list(ref_dir.rglob("*fsaverage*.png"))
    ref = None
    for c in candidates:
        if "prf" in c.name.lower() or "visual" in c.name.lower():
            ref = c
            break
    if ref is None and candidates:
        ref = candidates[0]
    pipe = PREP / "surfaces" / "roi_surface_fsaverage_inflated.png"
    if ref is None or not pipe.exists():
        print("WARNING: reference PNG not found; skip comparison.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].imshow(plt.imread(ref))
    axes[0].set_title("NSD Official ROI Visualisation")
    axes[0].axis("off")
    axes[1].imshow(plt.imread(pipe))
    axes[1].set_title("Pipeline-Generated ROI Map")
    axes[1].axis("off")
    plt.suptitle("ROI Visualisation Comparison")
    plt.tight_layout()
    out = PREP / "surfaces" / "roi_surface_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def ncsnr_to_nc(ncsnr: np.ndarray, n: int) -> np.ndarray:
    nc = np.full_like(ncsnr, np.nan, dtype=np.float64)
    valid = ncsnr > 0
    nc[valid] = (ncsnr[valid] ** 2 / (ncsnr[valid] ** 2 + 1.0 / n)) * 100.0
    return nc


def compute_nc_stats(
    mask_3d: Dict[int, Dict[str, np.ndarray]],
    min_reps: int,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    stats_out: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for aa in range(1, 9):
        path = TMP / "ncsnr" / f"subj{aa:02d}_ncsnr.nii.gz"
        vol = nib.load(str(path)).get_fdata().astype(np.float32)
        for roi in ROI_ORDER:
            roi_nc = vol[mask_3d[aa][roi]]
            roi_nc = roi_nc[np.isfinite(roi_nc) & (roi_nc > 0)]
            nc_pct = (roi_nc**2 / (roi_nc**2 + 1.0 / min_reps)) * 100.0
            stats_out[aa][roi] = {
                "n_voxels": len(roi_nc),
                "mean": float(np.mean(nc_pct)),
                "median": float(np.median(nc_pct)),
                "p25": float(np.percentile(nc_pct, 25)),
                "p75": float(np.percentile(nc_pct, 75)),
                "p10": float(np.percentile(nc_pct, 10)),
                "p90": float(np.percentile(nc_pct, 90)),
                "nc_pct_arr": nc_pct,
            }
    return stats_out


def print_nc_table(stats_out: Dict[int, Dict[str, Dict[str, Any]]]) -> None:
    print(
        "╔══════════╦═══════╦══════════╦══════════╦══════════╦══════════╦══════════╗\n"
        "║ Subject  ║  ROI  ║ N voxels ║ Mean NC% ║ Med  NC% ║ P25  NC% ║ P75  NC% ║\n"
        "╠══════════╬═══════╬══════════╬══════════╬══════════╬══════════╬══════════╣"
    )
    group: Dict[str, List[float]] = {r: [] for r in ROI_ORDER}
    for aa in range(1, 9):
        for roi in ROI_ORDER:
            s = stats_out[aa][roi]
            group[roi].append(s["mean"])
            print(
                f"║ subj{aa:02d}   ║  {roi:3s}  ║ {s['n_voxels']:8d} ║ "
                f"{s['mean']:8.1f} ║ {s['median']:8.1f} ║ "
                f"{s['p25']:8.1f} ║ {s['p75']:8.1f} ║"
            )
    print(
        "╠══════════╬═══════╬══════════╬══════════╬══════════╬══════════╬══════════╣"
    )
    for roi in ROI_ORDER:
        gm = float(np.mean(group[roi]))
        gmed = float(np.median([stats_out[aa][roi]["median"] for aa in range(1, 9)]))
        gp25 = float(np.mean([stats_out[aa][roi]["p25"] for aa in range(1, 9)]))
        gp75 = float(np.mean([stats_out[aa][roi]["p75"] for aa in range(1, 9)]))
        print(
            f"║ GROUP    ║  {roi:3s}  ║    ---   ║ {gm:8.1f} ║ {gmed:8.1f} ║ "
            f"{gp25:8.1f} ║ {gp75:8.1f} ║"
        )
    print(
        "╚══════════╩═══════╩══════════╩══════════╩══════════╩══════════╩══════════╝"
    )
    print(
        "Expected (reference): V1 ≈ 50–70%, V2 ≈ 45–65%, V4 ≈ 30–50%, IT ≈ 20–45%."
    )
    for aa in range(1, 9):
        for roi in ROI_ORDER:
            m = stats_out[aa][roi]["mean"]
            if m < 10:
                print(
                    f"WARNING: subj{aa:02d} {roi}: mean NC {m:.1f}% < 10% — "
                    "unusually low. Check ROI mask and session completeness."
                )
            if m > 90:
                print("WARNING: Suspiciously high NC — possible mask error.")


def plot_nc_distributions(
    stats_out: Dict[int, Dict[str, Dict[str, Any]]], min_reps: int
) -> None:
    fig, axes = plt.subplots(4, 8, figsize=(28, 16))
    for ri, roi in enumerate(ROI_ORDER):
        col = ROI_COLORS_QC[roi]
        for sj in range(1, 9):
            ax = axes[ri, sj - 1]
            s = stats_out[sj][roi]
            arr = s["nc_pct_arr"]
            ax.hist(arr, bins=30, range=(0, 100), color=col, alpha=0.7)
            ax.axvline(s["mean"], color="blue", linestyle="--")
            ax.axvline(s["median"], color="orange", linestyle="--")
            ax.set_title(
                f"subj{sj:02d} {roi}\nμ={s['mean']:.1f}% med={s['median']:.1f}%",
                fontsize=8,
            )
            if ri == 3:
                ax.set_xlabel("Noise Ceiling (%)")
            if sj == 1:
                ax.set_ylabel("Voxel count")
    plt.suptitle(
        f"Noise Ceiling Distributions by ROI and Subject\n"
        f"(n={min_reps} repeats, betas_fithrf, func1pt8mm)",
        fontsize=14,
    )
    plt.tight_layout()
    out = PREP / "qc" / "noise_ceiling_distributions.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def boundary_vertices(
    surf_path: Path, texture: np.ndarray, n_rois: int = 5
) -> np.ndarray:
    from nibabel.freesurfer.io import read_geometry

    coords, faces = read_geometry(str(surf_path))
    n_v = coords.shape[0]
    bound = np.zeros(n_v, dtype=bool)
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if 0 <= texture[u] <= n_rois and 0 <= texture[v] <= n_rois:
                if texture[u] != texture[v]:
                    bound[u] = True
                    bound[v] = True
    return bound


def plot_nc_surface(
    roi_cfg: Dict[str, Any], min_reps: int, it_atlas: str
) -> None:
    fs = TMP / "surfaces" / "fsaverage"
    lh_surf = fs / "lh.inflated"
    rh_surf = fs / "rh.inflated"
    lbl = fs / "labels"
    lh_tex = _build_surface_texture("lh", roi_cfg, lbl)
    rh_tex = _build_surface_texture("rh", roi_cfg, lbl)
    _ = boundary_vertices(lh_surf, lh_tex)
    _ = boundary_vertices(rh_surf, rh_tex)

    ncl = TMP / "ncsnr" / "lh.ncsnr.mgh"
    ncr = TMP / "ncsnr" / "rh.ncsnr.mgh"
    aws_s3_cp(
        f"{S3_BUCKET}/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf/lh.ncsnr.mgh",
        ncl,
    )
    aws_s3_cp(
        f"{S3_BUCKET}/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf/rh.ncsnr.mgh",
        ncr,
    )
    lh_ncsnr = nib.load(str(ncl)).get_fdata().squeeze()
    rh_ncsnr = nib.load(str(ncr)).get_fdata().squeeze()
    lh_nc = ncsnr_to_nc(lh_ncsnr.astype(np.float64), min_reps)
    rh_nc = ncsnr_to_nc(rh_ncsnr.astype(np.float64), min_reps)

    plotting = _nilearn_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw={"projection": "3d"})
    views = [
        ("left", "lateral", axes[0, 0]),
        ("left", "medial", axes[0, 1]),
        ("right", "lateral", axes[1, 0]),
        ("right", "medial", axes[1, 1]),
    ]
    for hemi, view, ax in views:
        pref = hemi[0]
        smesh = str(fs / f"{pref}h.inflated")
        bg = str(fs / f"{pref}h.sulc")
        stat = lh_nc if hemi == "left" else rh_nc
        plotting.plot_surf_stat_map(
            surf_mesh=smesh,
            stat_map=stat,
            hemi=hemi,
            view=view,
            bg_map=bg,
            bg_on_data=True,
            cmap="hot",
            vmin=0,
            vmax=80,
            colorbar=True,
            title=f"{hemi.capitalize()} {view.capitalize()}",
            axes=ax,
        )
    plt.suptitle(
        f"Noise Ceiling Map — subj01 on fsaverage\n"
        f"(n={min_reps} repeats, betas_fithrf)\n"
        f"Colormap: 0–80% NC; ROI mesh boundaries computed for QC (see code)",
        fontsize=11,
    )
    plt.tight_layout()
    out = PREP / "surfaces" / "noise_ceiling_surface_subj01.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved {out}")


def plot_nc_hierarchy(
    stats_out: Dict[int, Dict[str, Dict[str, Any]]], min_reps: int
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [1, 2, 3, 4]
    for aa in range(1, 9):
        means = [stats_out[aa][r]["mean"] for r in ROI_ORDER]
        ax.plot(
            xs,
            means,
            "-o",
            alpha=0.4,
            linewidth=1,
            markersize=4,
            label=f"subj{aa:02d}",
        )
    group_means = [
        float(np.mean([stats_out[aa][r]["mean"] for aa in range(1, 9)]))
        for r in ROI_ORDER
    ]
    group_sems = [
        float(
            np.std([stats_out[aa][r]["mean"] for aa in range(1, 9)]) / np.sqrt(8)
        )
        for r in ROI_ORDER
    ]
    ax.errorbar(
        xs,
        group_means,
        yerr=group_sems,
        fmt="-o",
        color="black",
        linewidth=3,
        markersize=8,
        capsize=5,
        label="Group mean ± SEM",
        zorder=10,
    )
    ax.set_xlabel("Visual Hierarchy Level", fontsize=12)
    ax.set_ylabel("Mean Noise Ceiling (%)", fontsize=12)
    ax.set_title(
        f"Noise Ceiling Across ROI Hierarchy\n(n={min_reps} repeats, betas_fithrf)",
        fontsize=12,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(ROI_ORDER)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PREP / "qc" / "noise_ceiling_by_roi_hierarchy.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")
    for aa in range(1, 9):
        ms = [stats_out[aa][r]["mean"] for r in ROI_ORDER]
        if not all(ms[i] >= ms[i + 1] for i in range(3)):
            print(
                f"Note: subj{aa:02d} mean NC% is not monotonic V1→V2→V4→IT: {ms}"
            )


def save_nc_npz(stats_out: Dict[int, Dict[str, Dict[str, Any]]]) -> None:
    kw: Dict[str, Any] = {}
    for aa in range(1, 9):
        for roi in ROI_ORDER:
            kw[f"subj{aa:02d}_{roi}_ncsnr"] = stats_out[aa][roi]["nc_pct_arr"]
            kw[f"subj{aa:02d}_{roi}_mean"] = np.float32(stats_out[aa][roi]["mean"])
    np.savez(PREP / "noise_ceiling_data.npz", **kw)


# =============================================================================
# Part 9 — Summary & metadata
# =============================================================================
def validate_outputs(
    n_final: int,
    min_reps: int,
    voxel_count: Dict[int, Dict[str, int]],
    roi_cfg: Dict[str, Any],
) -> bool:
    ok = True
    stim = PREP / "nsd_stimuli_224.hdf5"
    if stim.exists():
        with h5py.File(stim, "r") as f:
            if f["/images"].shape[0] != n_final:
                print("WARNING: stimuli shape mismatch")
                ok = False
    for aa in range(1, 9):
        for roi in ROI_ORDER:
            p = PREP / f"subj{aa:02d}" / f"nsd_neural_{roi}.hdf5"
            nv = voxel_count[aa][roi]
            exp = (n_final, min_reps, nv)
            with h5py.File(p, "r") as f:
                b = f["/betas"]
                if b.shape != exp:
                    print(f"WARNING: shape {p} {b.shape} != {exp}")
                    ok = False
                if b.dtype != np.float32:
                    ok = False
                if np.all(np.isnan(b[:])):
                    print(f"WARNING: all NaN {p}")
                    ok = False
                if len(f["/betas"].attrs.keys()) < 3:
                    ok = False
    return ok


def write_metadata(
    min_reps: int,
    n_final: int,
    roi_cfg: Dict[str, Any],
    voxel_count: Dict[int, Dict[str, int]],
    n_sessions: Dict[str, int],
    stats_out: Dict[int, Dict[str, Dict[str, Any]]],
    it_proxy_note: str,
) -> None:
    gm = {
        r: float(np.mean([stats_out[aa][r]["mean"] for aa in range(1, 9)]))
        for r in ROI_ORDER
    }
    meta = {
        "nsd_manual_url": NSD_MANUAL_URL,
        "creation_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "pipeline_version": "1.0",
        "subjects": [f"subj{aa:02d}" for aa in range(1, 9)],
        "n_sessions_per_subject": n_sessions,
        "image_set": {
            "n_final_images": n_final,
            "min_repetitions": min_reps,
            "index_convention": "0-based 73k IDs (Python)",
            "source_file": "nsd_stimuli.hdf5 /imgBrick (3,425,425,73000)",
            "resize": "224x224 PIL.Image.LANCZOS",
            "color_space": "RGB uint8",
        },
        "roi_config": roi_cfg,
        "it_proxy_note": it_proxy_note,
        "betas": {
            "version": "betas_fithrf (b2)",
            "space": "func1pt8mm",
            "units": "percent_signal_change (int16 / 300.0)",
            "missing": "NaN",
        },
        "per_subject_voxel_counts": {
            f"subj{aa:02d}": {r: voxel_count[aa][r] for r in ROI_ORDER}
            for aa in range(1, 9)
        },
        "noise_ceiling": {
            "formula": "NC(%) = ncsnr^2 / (ncsnr^2 + 1/n) * 100",
            "n_repeats": min_reps,
            "source_files": "nsddata_betas/.../betas_fithrf/ncsnr.nii.gz",
            "group_mean_nc_percent": gm,
        },
        "output_files": {
            "stimuli": "nsd_prepared/nsd_stimuli_224.hdf5",
            "neural_arrays": "nsd_prepared/subj{AA}/nsd_neural_{ROI}.hdf5",
            "final_image_set": "nsd_prepared/final_image_set.npz",
            "noise_ceiling": "nsd_prepared/noise_ceiling_data.npz",
            "metadata": "nsd_prepared/metadata.json",
        },
        "qc_figures": {
            "repetition_dist_by_subject": "qc/repetition_distribution_by_subject.png",
            "cross_subject_overlap": "qc/cross_subject_overlap.png",
            "example_images_stimuli": "qc/example_images_stimuli.png",
            "example_images_responses": "qc/example_images_with_responses.png",
            "final_rep_distribution": "qc/final_repetition_distribution.png",
            "nc_distributions": "qc/noise_ceiling_distributions.png",
            "nc_hierarchy": "qc/noise_ceiling_by_roi_hierarchy.png",
        },
        "surface_figures": {
            "roi_fsaverage_inflated": "surfaces/roi_surface_fsaverage_inflated.png",
            "roi_subj01_native": "surfaces/roi_surface_subj01_native_inflated.png",
            "roi_comparison": "surfaces/roi_surface_comparison.png",
            "nc_surface_subj01": "surfaces/noise_ceiling_surface_subj01.png",
        },
    }
    (PREP / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def print_final_summary(
    n_final: int,
    min_reps: int,
    voxel_count: Dict[int, Dict[str, int]],
) -> None:
    print(
        "┌─────────┬──────────┬──────────┬───────────┬───────────┬───────────┬───────────┐\n"
        "│ Subject │ N images │ Min reps │ V1 voxels │ V2 voxels │ V4 voxels │ IT voxels │\n"
        "├─────────┼──────────┼──────────┼───────────┼───────────┼───────────┼───────────┤"
    )
    for aa in range(1, 9):
        print(
            f"│ subj{aa:02d}  │ {n_final:8d} │ {min_reps:8d} │ "
            f"{voxel_count[aa]['V1']:9d} │ {voxel_count[aa]['V2']:9d} │ "
            f"{voxel_count[aa]['V4']:9d} │ {voxel_count[aa]['IT']:9d} │"
        )
    print(
        "└─────────┴──────────┴──────────┴───────────┴───────────┴───────────┴───────────┘"
    )
    stim = PREP / "nsd_stimuli_224.hdf5"
    if stim.exists():
        print(f"Stimulus file: nsd_stimuli_224.hdf5 size: {stim.stat().st_size / 1e9:.2f} GB")
    total_out = sum(f.stat().st_size for f in PREP.rglob("*") if f.is_file())
    print(f"Total output under nsd_prepared: {total_out / 1e9:.2f} GB")
    qcs = sorted((PREP / "qc").glob("*.png"))
    print("QC figures:", [q.name for q in qcs])
    surfs = sorted((PREP / "surfaces").glob("*.png"))
    print("Surface figures:", [s.name for s in surfs])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and prepare NSD (Natural Scenes Dataset) for 8 subjects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--non-interactive",
        "-y",
        action="store_true",
        help="Skip agreement/subject/MIN_REPS/ROI prompts; use MIN_REPS=3 and default ROIs. "
        "You must still comply with the NSD Data Access Agreement.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ni = bool(args.non_interactive)
    setup_logging()
    print_part_disk("Part 0")
    ensure_aws_cli()
    verify_s3_access()
    gate_data_agreement(non_interactive=ni)
    gate_subjects(non_interactive=ni)
    create_directories()
    print_part_disk("Part 1")

    # Part 2
    print_part_disk("Part 2")
    check_disk_space(0.5)
    exp = TMP / "nsd_expdesign.mat"
    stim_csv = TMP / "nsd_stim_info_merged.csv"
    if not exp.exists():
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata/experiments/nsd/nsd_expdesign.mat",
            exp,
        )
    if not stim_csv.exists():
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata/experiments/nsd/nsd_stim_info_merged.csv",
            stim_csv,
        )
    subjectim, masterordering, sharedix = load_expdesign(exp)
    stim_df = load_stim_csv(stim_csv)
    subject_73k_ids, global_to_local, trial_local_idx = build_mappings(
        subjectim, masterordering
    )

    # Part 3
    print_part_disk("Part 3")
    rep_count = count_reps(trial_local_idx)
    rng = np.random.default_rng(42)
    assert_csv_agreement(rep_count, stim_df, subject_73k_ids, rng)
    n_overlap, overlap_sets = overlap_analysis(rep_count, subject_73k_ids)
    print_overlap_table(rep_count, subject_73k_ids, n_overlap)
    shared_73k = set(int(x - 1) for x in sharedix.tolist())
    in_o3 = len(shared_73k & overlap_sets[3])
    print(
        f"Shared-1000 images (from sharedix): verified in overlap at >= 3 reps: {in_o3}"
    )
    plot_rep_distribution(rep_count, stim_df)
    plot_cross_subject_overlap_bars(rep_count, subject_73k_ids, n_overlap)

    def print_table_again() -> None:
        print_overlap_table(rep_count, subject_73k_ids, n_overlap)

    min_reps = gate_min_reps(n_overlap, print_table_again, non_interactive=ni)
    final_list, final_local_idx, final_set_pos = finalize_image_set(
        rep_count, subject_73k_ids, global_to_local, min_reps
    )
    n_final = len(final_list)

    rng2 = np.random.default_rng(42)
    qc_indices_in_final = rng2.choice(n_final, size=10, replace=False)
    qc_73k_ids = [final_list[i] for i in qc_indices_in_final]

    # Part 4
    print_part_disk("Part 4")
    download_subj01_atlases()
    for stem in ATLAS_CANDIDATES:
        p = TMP / "atlases" / "subj01" / f"{stem}.nii.gz"
        lp = TMP / "atlases" / "labels" / f"{stem}{LABEL_LOOKUP_SUFFIX}"
        if lp.exists():
            lm = parse_label_txt(lp)
        else:
            lm = {}
        print_atlas_label_table(stem, p, lm)
    streams_txt = TMP / "atlases" / "labels" / f"streams{LABEL_LOOKUP_SUFFIX}"
    it_atlas, it_labels, it_names = infer_default_it_streams(streams_txt)
    print(
        f"Default IT inference: atlas={it_atlas} labels={it_labels} names={it_names}"
    )
    default_cfg = default_roi_config(it_atlas, it_labels, it_names)
    def_it_atlas, def_it_labels, def_it_names = it_atlas, list(it_labels), list(it_names)
    roi_cfg = gate_roi_config(default_cfg, non_interactive=ni)
    it_atlas = roi_cfg["IT"]["atlas"]
    it_proxy_note = (
        f"IT ROI (confirmed): atlas={roi_cfg['IT']['atlas']}, "
        f"labels={roi_cfg['IT']['labels']}, names={roi_cfg['IT']['label_names']}. "
        f"Heuristic default was atlas={def_it_atlas}, labels={def_it_labels}, "
        f"names={def_it_names}. "
        "Ventral/temporal stream labels preferred from streams.mgz.ctab when used; "
        "else Kastner2015 VO/PHC; else fallback label 3."
    )

    # Part 5
    print_part_disk("Part 5")
    stim_raw = TMP / "nsd_stimuli_raw.hdf5"
    stim_out = PREP / "nsd_stimuli_224.hdf5"
    need_stim = True
    if stim_out.exists():
        with h5py.File(stim_out, "r") as sf:
            need_stim = sf["/images"].shape[0] != n_final
    if need_stim:
        if not _recover_complete_stimulus_file(stim_raw):
            download_stimuli_progress(
                f"{S3_BUCKET}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
                stim_raw,
            )
        else:
            print(
                f"[5.3] Found existing complete stimulus file at {stim_raw}; skipping download."
            )
        extract_stimuli_224(final_list, min_reps, n_final)

    # Part 6
    print_part_disk("Part 6")
    download_atlases_all_subjects(roi_cfg)
    mask_3d, mask_flat, voxel_count, _v3 = build_masks(roi_cfg)
    print_voxel_table(voxel_count)

    # Part 7
    print_part_disk("Part 7")
    n_sessions: Dict[str, int] = {}
    session_lists: Dict[int, List[int]] = {}
    all_out: Dict[int, Dict[str, Path]] = {}
    for aa in range(1, 9):
        sl = list_beta_sessions(aa)
        session_lists[aa] = sl
        n_sessions[f"subj{aa:02d}"] = len(sl)
        print(f"subj{aa:02d}: {len(sl)} beta sessions on S3")
        paths = create_neural_h5(aa, roi_cfg, voxel_count, n_final, min_reps)
        all_out[aa] = paths
        try:
            process_subject_sessions(
                aa,
                trial_local_idx,
                subject_73k_ids,
                final_set_pos,
                mask_3d,
                paths,
                n_final,
                min_reps,
                len(sl),
                sl,
            )
        except Exception as e:
            log_error(f"subj{aa:02d} session loop", e)
        verify_neural(aa, paths, n_final, min_reps)

    # Part 8
    print_part_disk("Part 8")
    qc_example_stimuli(qc_indices_in_final, qc_73k_ids)
    qc_example_responses(qc_indices_in_final, min_reps)
    qc_inter_repeat(final_list, min_reps, final_set_pos)
    qc_final_rep_distribution(min_reps, n_final, all_out)

    # Part 8B
    print_part_disk("Part 8B")
    try:
        download_surface_assets(it_atlas)
    except Exception as e:
        log_error("download_surface_assets", e)
    try:
        plot_roi_surfaces(roi_cfg, native=False)
        plot_roi_surfaces(roi_cfg, native=True)
        roi_comparison_figure()
    except Exception as e:
        log_error("surface plots", e)
    for aa in range(1, 9):
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata_betas/ppdata/subj{aa:02d}/func1pt8mm/betas_fithrf/ncsnr.nii.gz",
            TMP / "ncsnr" / f"subj{aa:02d}_ncsnr.nii.gz",
        )
    stats_out = compute_nc_stats(mask_3d, min_reps)
    print_nc_table(stats_out)
    plot_nc_distributions(stats_out, min_reps)
    try:
        plot_nc_surface(roi_cfg, min_reps, it_atlas)
    except Exception as e:
        log_error("plot_nc_surface", e)
    plot_nc_hierarchy(stats_out, min_reps)
    save_nc_npz(stats_out)

    # Part 9
    print_part_disk("Part 9")
    print_final_summary(n_final, min_reps, voxel_count)
    ok = validate_outputs(n_final, min_reps, voxel_count, roi_cfg)
    if ok and TMP.exists():
        shutil.rmtree(TMP)
        print("Removed ./nsd_tmp/ after successful validation.")
    else:
        print(
            "Validation had warnings or failed — keeping ./nsd_tmp/. "
            "Inspect messages and re-run."
        )
    write_metadata(
        min_reps,
        n_final,
        roi_cfg,
        voxel_count,
        n_sessions,
        stats_out,
        it_proxy_note,
    )
    print(
        f"""
╔════════════════════════════════════════════════════════════════╗
║  NSD DATASET PREPARATION COMPLETE                             ║
║                                                               ║
║  Images:   {n_final} × 224×224×3 (uint8)                       ║
║  Subjects: 8  |  ROIs: V1, V2, V4, IT  |  Min reps: {min_reps} ║
║  All outputs in: ./nsd_prepared/                              ║
║  See metadata.json for full provenance.                       ║
╚════════════════════════════════════════════════════════════════╝
"""
    )


if __name__ == "__main__":
    main()

