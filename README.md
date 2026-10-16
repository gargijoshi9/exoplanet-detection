# Exoplanet Detection using Machine Learning

This repository implements a pipeline for detecting and classifying exoplanet signals in noisy
**TESS** (Transiting Exoplanet Survey Satellite) light curves.

A light curve is a star's brightness measured over time. When a planet passes in front of its star,
the brightness dips slightly. The aim is to find these periodic dips and to tell real planet transits apart
from other signals, such as eclipsing binary stars and rejected planet candidates.

## Project Structure

```text
exoplanet-detection/
│
├── data/
│   │                         # NOTE: everything inside data/ and results/ is generated locally and is NOT in git
│   ├── raw/                  # temporary FITS downloads (deleted automatically after processing)
│   ├── processed/            # one TIC_<id>.npz file per star (created by preprocess.py)
│   ├── full/                 # full 2-minute CSV for a single star (created by preprocess.py --full)
│   ├── labels.csv            # star list: tic_id, label, source, train/test split (created by build_labels.py)
│   ├── failed.csv            # stars that could not be processed, with the reason
│   └── preprocess.log        # log of the preprocessing run
│
├── notebooks/
│   └── detection.ipynb       # step-by-step walkthrough of the transit detection on TIC 307210830
│
├── src/
│   ├── build_labels.py       # builds data/labels.csv from public catalogs
│   ├── preprocess.py         # downloads and cleans light curves (all stars, or one star with --tic)
│   ├── detect.py             # transit detection: BLS search, vetting, multi-planet masking
│   ├── feature_extraction.py
│   ├── train_classifier.py
│   └── evaluate.py
│
├── models/
├── results/
│   └── detection/            # detection tables and plots (created by detect.py)
├── requirements.txt
└── README.md
```

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

An internet connection is needed for the data steps, because the catalogs and light curves are
downloaded from NASA / MAST.

## Pipeline Status

| Stage | Status |
|---|---|
| 1. Build labelled star list (`build_labels.py`) | Done |
| 2. Download and preprocess light curves (`preprocess.py`) | Done |
| 3. Transit detection (`detect.py`, `notebooks/detection.ipynb`) | Done |
| 4. Transit parameter fitting | Not started |
| 5. Feature extraction (`feature_extraction.py`) | Not started (placeholder) |
| 6. Classifier training and evaluation | Not started |

---

## Getting the Data (read this first)

The data files are **not stored in this repository**. `.gitignore` excludes `data/raw/`, `data/processed/`,
all `*.csv` files, all `*.png` files and `*.log` files, so after you pull you only have the code. You create the
dataset on your own machine by running two scripts, in this order, from the repository root:

```bash
python src/build_labels.py                       # Step 1: makes data/labels.csv   (about 1 minute)
python src/preprocess.py --workers 4             # Step 2: makes data/processed/   (about 20 to 30 minutes)
```

Run them exactly as shown, without changing any options (the defaults are 50 stars per class, sampled with a
fixed seed), so that you get the same stars as everyone else. You need an internet connection. Run Step 1 only
once: running it again overwrites `data/labels.csv` with a new sampling.

### If some stars fail to download (very common)

The NASA / MAST servers sometimes time out. This is normal. **Just run Step 2 again**:

```bash
python src/preprocess.py --workers 4
```

It skips every star that is already done and only retries the ones that failed. Repeat until the log says
that 0 stars are left "to process". If you see "file may be corrupt" errors that keep repeating, delete the
folder `data/raw/mastDownload` and run Step 2 again. If you see other strange download errors, use
`--workers 1`.

Some stars are skipped on purpose and are **not errors**. They are written to `data/failed.csv` and never
retried:

- stars that have no 2-minute light curve (the log reason starts with "no 2-min"), and
- stars whose first 2-minute sector is after sector 26 (the reason starts with "first sector").

### How to check that you got the same data as the team

After Step 1, `data/labels.csv` should have **240 rows (80 per class)**. After Step 2 has finished, the
summary table printed at the end of the run (and in `data/preprocess.log`) should show about:

| Class | Test | Train | Total |
|---|---|---|---|
| Eclipsing Binary | 20 | 58 | 78 |
| Transit | 10 | 46 | 56 |
| Other | 3 | 21 | 24 |
| **All** | 33 | 125 | **158** |

The star list is sampled from the live NASA catalogs (TOI table and TESS-EBs), using a fixed random seed.
These numbers were produced in October 2026. If NASA has changed a star's disposition since then, your counts
can differ by a few stars. That is acceptable, but tell the team if the difference is large.

When loading data in code, always read the stars from `data/labels.csv`, never by listing the files in
`data/processed/`.

---

## Phase 1: Data Pipeline (labels and preprocessing)

### 1. Why a labelled dataset of many stars

A classifier learns from many examples. One star, even with many TESS sectors of data, is a single
example, so the data was built from many stars, one light curve per star, each with a class label.

### 2. Classes and where the labels come from

| Class | Source | What it means |
|---|---|---|
| **Transit** | NASA Exoplanet Archive TOI table, dispositions CP / KP / PC | Confirmed, known or candidate planets |
| **Eclipsing Binary** | TESS-EBs catalog (Prša et al. 2022) | Two stars eclipsing each other |
| **Other** | NASA Exoplanet Archive TOI table, dispositions FP / FA | Looked like a planet candidate but was rejected |

Rules that keep the labels clean:

- A star with a planet-type TOI that is also in the EB catalog is **dropped** (ambiguous).
- A star in the EB catalog is labelled Eclipsing Binary, even if it is also a rejected TOI.
- "Other" never includes a star that has any planet-type TOI.

**Why "Other" = rejected candidates:** these are the hard negatives. They are the signals that looked
like planets and had to be ruled out, which is the case a classifier most needs to handle.

### 3. Why the stars are matched by era (sectors 1 to 26)

The TESS-EBs catalog only covers TESS sectors 1 to 26. When Transit and Other stars were first sampled
without any restriction, most of them came from later sectors. A model could then learn "early sector means
eclipsing binary" from instrument and noise differences, instead of learning the shape of the light curve.
Predicting the class from the sector number alone scored 73% in cross-validation (the biggest class was
37%).

To fix this:

- `build_labels.py` keeps only Transit / Other stars with a TOI number below 2000 (`--max-toi`), which were
  mostly alerted from the early sectors.
- `preprocess.py` then enforces the real rule: it skips any star whose first 2-minute sector is after
  sector 26 (`--max-sector`).

After the fix, the sector number alone scores 39% in cross-validation, below the 49% you get by always
guessing the biggest class. The shortcut is gone.

### 4. Preprocessing steps (per star)

Implemented in `src/preprocess.py` and applied to every star:

1. **Download** the 2-minute SPOC light curve (PDCSAP flux) with `lightkurve`. Only **one sector per star**
   is used (the earliest one), so one star is one sample.
2. **Remove NaNs.** Stars with fewer than 5,000 valid points are rejected.
3. **Remove outliers:** upper 5σ and lower 10σ. The lower limit is loose on purpose, so real transits
   (downward dips) are not clipped.
4. **Flatten** with a Savitzky-Golay filter (window of 301 cadences, about 10 hours) to remove slow stellar
   and instrument trends.
5. **Normalize** so the median flux is 1.
6. **Bin to 10-minute cadence.** This shrinks each file about five times and keeps the later feature
   extraction fast.

Two flux columns are saved:

- `flux`: detrended and normalized. Use this for detection, features and classification.
- `flux_norm`: normalized but not detrended. Kept so that later stages (such as transit fitting) can
  re-flatten with the transit masked, because flattening without a mask slightly reduces the transit depth.

### 5. Train / test split

The split is done **by star**, about 80 / 20, and depends only on the star ID and a fixed seed. This means:

- no star appears in both train and test;
- adding more stars later does not move existing stars between train and test.

### 6. Final dataset

| Class | Test | Train | Total |
|---|---|---|---|
| Eclipsing Binary | 20 | 58 | 78 |
| Transit | 10 | 46 | 56 |
| Other | 3 | 21 | 24 |
| **All** | 33 | 125 | **158** |

240 stars were sampled (80 per class). Of the 82 not in the final set, 19 have no 2-minute data and 63
first appeared after sector 26. All 158 files were checked: no NaNs, increasing timestamps, and at least
2,214 points each.

### 7. Optional: one star at full 2-minute cadence

The detection phase works best on a long, full-cadence light curve of one star. This command downloads
sectors 8 to 12 of TIC 307210830 (the L 98-59 system), keeps the 2-minute cadence and writes a CSV:

```bash
python src/preprocess.py --tic 307210830 --sectors 8 9 10 11 12 --full
```

It writes `data/full/processed_TIC_307210830.csv` with the columns `time, flux, flux_err, flux_norm`
(75,041 points over about 135 days). It takes about 1 to 3 minutes, depending on your internet speed and
how busy MAST is. If it fails with a timeout or a "file may be corrupt" error, delete `data/raw/mastDownload`
and run the same command again.

### 8. Loading the processed data in Python

Always read stars from `labels.csv`, because `data/processed/` may also contain files from earlier runs.

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from preprocess import load_star

labels = pd.read_csv("data/labels.csv")
star = load_star(labels.tic_id.iloc[0])     # DataFrame: time, flux, flux_err, flux_norm
```

---

## Phase 2: Transit Detection

Implemented in `src/detect.py`. The walkthrough with plots is in `notebooks/detection.ipynb`.

**Input:** a preprocessed (detrended, normalized) light curve: `time` [days], `flux`, `flux_err`.
**Output:** a table with one row per periodic signal found (period, mid-transit time, duration, depth, SNR
and vetting results), plus plots.

### How it works

1. **Bin** the light curve to 10 minutes for a fast search. If the data is already binned, this is skipped.
2. **Run Box Least Squares (BLS)** on a log-spaced period grid (0.5 days up to half the data length, capped
   at 40 days). The grid is made fine enough that the phase drift over the whole data set stays below one
   third of the shortest transit duration, so a real transit is not smeared out.
3. **Take the highest peak.** If its signal-to-noise ratio (SNR) is below 7.1, stop. This is the same
   threshold the TESS pipeline uses for a threshold crossing event.
4. **Refine** period, mid-transit time and duration with a narrow BLS search on the full-cadence data.
5. **Vet** the signal (see below).
6. **Mask** the transits of that signal (±1 duration around each one) and go back to step 2 to look for the
   next planet. This repeats up to 4 times (`--max-planets`).

### Vetting checks

| Check | Limit | What it catches |
|---|---|---|
| Number of transits | at least 2 | single events and noise spikes |
| Odd / even depth difference | below 3σ | eclipsing binaries (odd and even eclipses differ) |
| Secondary eclipse | below 3σ | eclipsing binaries (a second dip half an orbit later) |
| Depth | below 5% | stellar eclipses, which are far deeper than planets |
| Duration / period | below 0.1 | signals that are too long to be a planet transit |

The red-noise SNR (`snr_red`) is also reported. It accounts for correlated noise, so it is more honest than
the plain SNR when the star is variable.

### How to run

```bash
# the multi-sector star made in Phase 1, step 7 (takes about 4 to 5 minutes)
python src/detect.py --csv data/full/processed_TIC_307210830.csv

# a star from the classification dataset
python src/detect.py --npz data/processed/TIC_101961987.npz
```

| Option | Meaning |
|---|---|
| `--max-planets N` | maximum number of signals to search for (default 4) |
| `--min-snr X` | SNR threshold to stop the search (default 7.1) |
| `--period-max D` | largest period to search, in days |
| `--out DIR` | output folder (default `results/detection`) |

For each input, `detect.py` writes into `results/detection/`:

- `<name>_detections.csv`: the result table;
- `<name>_overview.png`: the full light curve with the detected transits marked;
- `<name>_periodogram_<n>.png`: the BLS periodogram for each signal;
- `<name>_fold_<n>.png`: the light curve folded on each signal's period.

These files are generated locally and are not in git.

### Result table columns

| Column | Meaning |
|---|---|
| `planet` | order in which the signal was found (strongest first) |
| `period_d`, `t0_btjd`, `duration_h`, `depth_ppm` | BLS estimates of the period, mid-transit time (BTJD), duration and depth |
| `snr`, `snr_red`, `sde` | signal-to-noise, red-noise signal-to-noise, signal detection efficiency |
| `n_transits` | number of transits covered by the data |
| `odd_even_sigma`, `secondary_sigma` | results of the two eclipsing-binary tests |
| `flags` | any vetting check that failed |
| `accepted` | the signal passed the SNR threshold |
| `status` | `candidate` (passed everything), `flagged` (SNR ok but a vetting check failed) or `rejected` (below the SNR threshold) |

### Result on the benchmark star (TIC 307210830, L 98-59 / TOI-175)

L 98-59 is a bright M dwarf with three confirmed terrestrial planets, so it is a good test of multi-planet
detection.

| Signal | Period (d) | Duration (h) | Depth (ppm) | SNR | Transits | Status |
|---|---|---|---|---|---|---|
| 1 | 3.6908 | 1.11 | 1367 | 55.5 | 30 | candidate |
| 2 | 7.4505 | 0.68 | 1333 | 31.0 | 16 | candidate |
| 3 | 2.2531 | 0.90 | 515 | 23.5 | 46 | candidate |
| 4 | 1.0491 | 1.24 | 177 | 13.3 | 94 | flagged (secondary eclipse) |

Signals 1 to 3 match the three confirmed planets. Signal 4 is flagged by the secondary-eclipse test and is
most likely a false positive, so the vetting is doing its job.

### Known limitations of the detection phase

- **BLS gives a box-shaped estimate.** Period, duration and depth are good starting values, but they carry
  no uncertainties, and the planet-to-star radius ratio, impact parameter and similar quantities are not
  estimated. That is the job of the parameter fitting stage.
- **Detrending bias.** The `flux` column was flattened without a transit mask, so depths are slightly
  underestimated. Re-flatten `flux_norm` with the transits masked before fitting.
- **One sector per star in the classification dataset.** Planets with long periods may show only one
  transit, or none, in a single sector, so the vetting (which needs at least 2 transits) can reject them.
- **Weak signals in noisy stars.** For example, TIC 101961987 gives a best peak with SNR 2.5, which is
  correctly rejected as noise.

---

## Known limitations of the dataset

- **Small "Other" class (24 stars).** Results for this class are only indicative. A fixed test split has just
  3 Other stars, so use stratified cross-validation over all stars and report per-class precision, recall
  and F1, not only accuracy.
- **Class imbalance (78 / 56 / 24).** Use balanced class weights.
- **Label noise.** Catalog labels are not perfect, and candidate (PC) stars are not all confirmed planets.
- **One sector per star.** See the detection limitations above.

---

## Remaining work

1. **Transit parameter fitting** (starting from the BLS values and using `batman`): fit the radius ratio, impact
   parameter, scaled semi-major axis and mid-transit time with uncertainties, re-flattening `flux_norm` with
   the transits masked.
2. **Feature extraction** from the light curves and the detection results.
3. **Classifier training and evaluation** (stratified cross-validation, balanced class weights, per-class
   metrics).

## Data Sources

- TESS mission data (SPOC pipeline, 2-minute cadence) from MAST, accessed with
  [lightkurve](https://docs.lightkurve.org/).
- TESS Objects of Interest (TOI) table, NASA Exoplanet Archive.
- TESS Eclipsing Binary catalog: Prša et al. 2022, *TESS Eclipsing Binary Stars. I. Short-cadence
  Observations of 4584 Eclipsing Binaries in Sectors 1 to 26*, ApJS 258, 16 (VizieR `J/ApJS/258/16`).