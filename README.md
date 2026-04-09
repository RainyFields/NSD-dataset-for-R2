# NSD data preparation pipeline

A single Python script that downloads and prepares **[Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/)** data from the **public AWS bucket** (`--no-sign-request`, no AWS account required) for **cross-subject** analysis across **8 subjects** (`subj01`–`subj08`).

**Official reference:** [NSD Data Manual](https://cvnlab.slite.com/api/s/channel/CPyFRAyDYpxdkPK6YbB5R1/NSD%20Data%20Manual)

---

## What this repository does

1. **Design & image selection (no large neural downloads)**  
   Loads `nsd_expdesign.mat` and `nsd_stim_info_merged.csv`, builds trial→image mappings, counts repetitions per subject, computes **cross-subject overlap** of images at repeat thresholds 1/2/3, and lets you choose a minimum repetition threshold **`MIN_REPS`** (typically **3**).  
   Produces a **final set of 0-based 73k image IDs** shared by all subjects with at least **`MIN_REPS`** repeats.

2. **ROIs**  
   Uses NSD **func1pt8mm** ROI atlases (default: **V1/V2/V4** from `prf-visualrois`; **IT** inferred from `streams.mgz.ctab` labels, with fallbacks). You can confirm or override mappings interactively, or use **`--non-interactive`** to keep defaults.

3. **Stimuli**  
   Downloads **`nsd_stimuli.hdf5`** (~40 GB), extracts **`/imgBrick`** for the **final image set only**, resizes to **224×224 RGB** (`uint8`), writes **`nsd_prepared/nsd_stimuli_224.hdf5`**, then deletes the raw full brick when extraction succeeds.

4. **Neural betas**  
   For each subject, downloads **`betas_fithrf`** session HDF5s (`betas_sessionXX.hdf5`), converts **`int16 / 300`** → **float32** (percent signal change), and fills **per-ROI** arrays **`(N_FINAL, MIN_REPS, n_voxels)`** with **NaN** for missing slots. Sessions are processed **sequentially**; temporary session files are deleted after each session.

5. **QC & surfaces**  
   Writes figures under **`nsd_prepared/qc/`** and **`nsd_prepared/surfaces/`** (repetition plots, example images, noise-ceiling summaries, optional surface maps via **nilearn**).

6. **Metadata**  
   Writes **`nsd_prepared/metadata.json`** with provenance, ROI config, session counts, and paths to outputs.

---

## What gets generated (under `./nsd_prepared/`)

After a full successful run (exact sizes depend on **`N_FINAL`** and ROI voxel counts):

| Path | Description |
|------|-------------|
| `final_image_set.npz` | Final 73k IDs, `MIN_REPS`, per-subject local indices into each 10k set |
| `nsd_stimuli_224.hdf5` | `/images` `(N_FINAL, 224, 224, 3)` uint8; `/global_image_indices_73k` |
| `subjXX/nsd_neural_V1.hdf5` (and V2, V4, IT) | `/betas` float32, shape `(N_FINAL, MIN_REPS, n_voxels)` |
| `noise_ceiling_data.npz` | Per-subject/ROI noise-ceiling distributions and means |
| `metadata.json` | Full provenance and settings |
| `errors.log` | Logged failures (e.g. failed session downloads) |
| `qc/*.png` | Quality-control figures |
| `surfaces/*.png` | ROI / noise-ceiling surface figures (if dependencies and downloads succeed) |

Temporary downloads use **`./nsd_tmp/`** and are removed after **successful** end-to-end validation (otherwise **`nsd_tmp/`** is kept for inspection).

---

## Requirements

- **Python 3.8+** (3.9+ recommended; tested with conda).
- **AWS CLI v2** on `PATH` (the script can `pip install awscli` if missing).
- **Network** access to `s3://natural-scenes-dataset` (public).
- **Disk space:** plan **~55 GB free** before the **full stimulus** download; **~150+ GB** for a comfortable full run including neural outputs and margin. Beta sessions are downloaded **one at a time** (~1 GB each temp file).
- **NSD Data Access Agreement:** you must complete the form linked in the script before downloading (same as NSD policy).

---

## Quick start (conda)

```bash
git clone <your-repo-url>
cd <repo-directory>

# Create/update environment (see environment.yml)
conda env create -f environment.yml   # first time
conda activate fmri

# Optional: verify S3 access
aws s3 ls s3://natural-scenes-dataset/ --no-sign-request

# Run (interactive prompts for agreement, subjects, MIN_REPS, ROIs)
python nsd_prepare_pipeline.py
```

### Non-interactive mode (defaults only)

Skips prompts and uses **MIN_REPS = 3** and **default ROI** mappings (including inferred IT from `streams`). You are still responsible for having completed the **NSD access agreement**.

```bash
python nsd_prepare_pipeline.py --non-interactive
# short form:
python nsd_prepare_pipeline.py -y
```

---

## Resuming and failures

- The script **skips** steps whose outputs already exist with the **expected shapes** (e.g. existing `nsd_stimuli_224.hdf5`, neural HDF5s).
- **Stimulus download:** `aws s3 cp` does **not** resume a partial file; interrupted runs **restart** the ~40 GB download (partial temp files are cleaned). Use a **stable connection**, **`tmux`/`screen`**, and see script options for retries and timeouts.
- **Beta sessions:** failed sessions are **logged** to `nsd_prepared/errors.log` and **retried** at the end of each subject.

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `ModuleNotFoundError` (e.g. PIL) | Use **`conda activate fmri`**; reinstall env from **`environment.yml`**. |
| Broken NumPy / `numpy._core` | Reinstall from conda-forge: `conda install -y -c conda-forge "numpy>=1.26,<2.1" scipy --force-reinstall` — avoid mixing **pip** and **conda** NumPy. |
| Slow or failing ~40 GB download | Ethernet, no VPN if possible, **`--cli-read-timeout 0`** is set in script; check **`nsd_tmp/aws_s3cp_stimuli.log`**. |
| Label files | NSD uses **`*.mgz.ctab`** for label names (not `*.mgz.txt`). |

---

## Citing NSD

If you use this pipeline or NSD data in a publication, cite the NSD papers and follow **[NSD citation guidelines](https://naturalscenesdataset.org/)** and the **data use agreement**.

---

## Push this code to a new GitHub repository

1. Create an **empty** repository on GitHub (no README/license if you already have them locally), e.g. `your-username/nsd-prepare-pipeline`.

2. On your machine:

```bash
cd /path/to/202604_R2_fmri_data   # this project folder

# If you already ran `git init` here, skip init; ensure branch is main:
git branch -M main

git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Use **SSH** if you prefer: `git@github.com:YOUR_USERNAME/YOUR_REPO.git`.

3. **Never commit** `nsd_prepared/` or `nsd_tmp/` — they are listed in **`.gitignore`** and can be **tens to hundreds of GB**.

---

## License

This **repository** (script + docs) is provided as-is for research use. **NSD data** remain subject to the **NSD Data Access Agreement** and NSD’s terms — not replaced by any license in this repo.
