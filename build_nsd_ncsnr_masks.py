#!/usr/bin/env python3
"""
Build per-subject, per-ROI voxel masks from OFFICIAL NSD ncsnr maps.

Inputs:
- nsd_prepared/metadata.json (for ROI atlas + labels)
- nsd_prepared/subjXX/nsd_neural_{ROI}.hdf5 (for expected n_vox per ROI)

Downloads:
- NSD official ncsnr.nii.gz per subject
- ROI atlas volumes per subject

Outputs:
- <outdir>/subjXX/subjXX_{ROI}_ncsnr.npy         (float32, length n_vox)
- <outdir>/subjXX/subjXX_{ROI}_nc_pct.npy        (float32, length n_vox; optional)
- <outdir>/subjXX/subjXX_{ROI}_mask_*.npy        (bool, length n_vox)

Example:
python build_nsd_ncsnr_masks.py \
  --prepared-dir nsd_prepared \
  --subjects subj01,subj02 \
  --outdir nsd_ncsnr_masks \
  --threshold-nc-pct 20 \
  --min-reps 3 \
  --perm 0,1,2
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import nibabel as nib
import numpy as np

S3_BUCKET = "s3://natural-scenes-dataset"
ROI_ORDER = ["V1", "V2", "V4", "IT"]


def parse_subjects(raw: str) -> List[int]:
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(1, 9))
    toks = [t.strip().lower() for t in raw.split(",") if t.strip()]
    out: List[int] = []
    for tok in toks:
        m = re.fullmatch(r"(?:subj)?0*([1-8])", tok)
        if not m:
            raise ValueError(f"Invalid subject token: {tok}")
        aa = int(m.group(1))
        if aa not in out:
            out.append(aa)
    out.sort()
    return out


def parse_perm(raw: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in raw.split(",")]
    if sorted(parts) != [0, 1, 2]:
        raise ValueError("--perm must be a permutation of 0,1,2 (e.g., 0,1,2 or 2,1,0)")
    return tuple(parts)  # type: ignore


def aws_cp_public(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[CACHE] {dst}")
        return
    cmd = ["aws", "s3", "cp", src, str(dst), "--no-sign-request"]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_roi_cfg(prepared_dir: Path) -> Dict[str, Dict]:
    meta_path = prepared_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {meta_path}. "
            "Need this to recover ROI atlas/labels used by your pipeline."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    roi_cfg = meta.get("roi_config")
    if not isinstance(roi_cfg, dict):
        raise RuntimeError("metadata.json missing roi_config")
    for roi in ROI_ORDER:
        if roi not in roi_cfg:
            raise RuntimeError(f"roi_config missing {roi}")
    return roi_cfg


def expected_nvox_from_neural(prepared_dir: Path, subj: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    sub = f"subj{subj:02d}"
    for roi in ROI_ORDER:
        p = prepared_dir / sub / f"nsd_neural_{roi}.hdf5"
        if not p.exists():
            raise FileNotFoundError(f"Missing neural file: {p}")
        with h5py.File(p, "r") as f:
            out[roi] = int(f["/betas"].shape[2])
    return out


def ncsnr_to_nc_pct(ncsnr_vec: np.ndarray, n_repeats: int) -> np.ndarray:
    out = np.full_like(ncsnr_vec, np.nan, dtype=np.float32)
    valid = np.isfinite(ncsnr_vec) & (ncsnr_vec > 0)
    x2 = ncsnr_vec[valid] ** 2
    out[valid] = (x2 / (x2 + 1.0 / float(n_repeats))) * 100.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=Path("nsd_prepared"))
    ap.add_argument("--subjects", type=str, default="subj01")
    ap.add_argument("--outdir", type=Path, default=Path("nsd_ncsnr_masks"))
    ap.add_argument("--cache-dir", type=Path, default=Path("nsd_tmp_ncsnr_cache"))
    ap.add_argument(
        "--perm",
        type=str,
        default="0,1,2",
        help="Axis permutation applied to volume+mask before flattening.",
    )
    ap.add_argument("--min-reps", type=int, default=3)
    ap.add_argument(
        "--threshold-ncsnr",
        type=float,
        default=None,
        help="If set, save mask where ncsnr >= threshold.",
    )
    ap.add_argument(
        "--threshold-nc-pct",
        type=float,
        default=None,
        help="If set, save mask where NC%% >= threshold.",
    )
    args = ap.parse_args()

    prepared_dir = args.prepared_dir.resolve()
    outdir = args.outdir.resolve()
    cache_dir = args.cache_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    perm = parse_perm(args.perm)
    subjects = parse_subjects(args.subjects)
    roi_cfg = load_roi_cfg(prepared_dir)

    for aa in subjects:
        sub = f"subj{aa:02d}"
        print(f"\n=== {sub} ===")

        nvox_expected = expected_nvox_from_neural(prepared_dir, aa)

        # Download official ncsnr volume
        ncsnr_path = cache_dir / f"{sub}_ncsnr.nii.gz"
        aws_cp_public(
            f"{S3_BUCKET}/nsddata_betas/ppdata/{sub}/func1pt8mm/betas_fithrf/ncsnr.nii.gz",
            ncsnr_path,
        )

        # Load and permute ncsnr volume
        ncsnr_vol = nib.load(str(ncsnr_path)).get_fdata().astype(np.float32)
        ncsnr_vol = np.transpose(ncsnr_vol, perm)
        ncsnr_flat = ncsnr_vol.ravel(order="C")

        sub_out = outdir / sub
        sub_out.mkdir(parents=True, exist_ok=True)

        # Build ROI-aligned vectors
        for roi in ROI_ORDER:
            atlas_name = str(roi_cfg[roi]["atlas"])
            labels = [int(x) for x in roi_cfg[roi]["labels"]]

            atlas_path = cache_dir / sub / f"{atlas_name}.nii.gz"
            aws_cp_public(
                f"{S3_BUCKET}/nsddata/ppdata/{sub}/func1pt8mm/roi/{atlas_name}.nii.gz",
                atlas_path,
            )

            atlas_vol = nib.load(str(atlas_path)).get_fdata().astype(np.int32)
            atlas_vol = np.transpose(atlas_vol, perm)
            mask = np.isin(atlas_vol, labels)
            lin_idx = np.flatnonzero(mask.ravel(order="C"))

            vec = ncsnr_flat[lin_idx].astype(np.float32, copy=False)

            # sanity against your saved neural voxel count
            if vec.shape[0] != nvox_expected[roi]:
                raise RuntimeError(
                    f"{sub} {roi}: ncsnr vec len {vec.shape[0]} != neural axis2 {nvox_expected[roi]}.\n"
                    f"Try a different --perm (e.g., 2,1,0)."
                )

            np.save(sub_out / f"{sub}_{roi}_ncsnr.npy", vec)
            print(f"[SAVE] {sub}_{roi}_ncsnr.npy  shape={vec.shape}")

            if args.threshold_ncsnr is not None:
                m = np.isfinite(vec) & (vec >= float(args.threshold_ncsnr))
                np.save(sub_out / f"{sub}_{roi}_mask_ncsnr_ge_{args.threshold_ncsnr:g}.npy", m)
                print(
                    f"[SAVE] {sub}_{roi}_mask_ncsnr_ge_{args.threshold_ncsnr:g}.npy true={int(m.sum())}/{m.size}"
                )

            nc_pct = ncsnr_to_nc_pct(vec, args.min_reps)
            np.save(sub_out / f"{sub}_{roi}_nc_pct.npy", nc_pct)
            print(f"[SAVE] {sub}_{roi}_nc_pct.npy shape={nc_pct.shape}")

            if args.threshold_nc_pct is not None:
                m2 = np.isfinite(nc_pct) & (nc_pct >= float(args.threshold_nc_pct))
                np.save(sub_out / f"{sub}_{roi}_mask_nc_pct_ge_{args.threshold_nc_pct:g}.npy", m2)
                print(
                    f"[SAVE] {sub}_{roi}_mask_nc_pct_ge_{args.threshold_nc_pct:g}.npy true={int(m2.sum())}/{m2.size}"
                )

    print("\nDone.")
    print(f"Outputs: {outdir}")


if __name__ == "__main__":
    main()
