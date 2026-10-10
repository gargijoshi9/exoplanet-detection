# Exoplanet Detection using Machine Learning

This repository implements a pipeline for detecting and classifying exoplanet signals in noisy
**TESS** (Transiting Exoplanet Survey Satellite) light curves using machine learning.

A light curve is a star's brightness measured over time. When a planet passes in front of its star,
the brightness dips slightly. The aim is to tell real planet transits apart from other signals, such as
eclipsing binary stars and rejected planet candidates.

## Project Structure

```text
exoplanet-detection/
│
├── data/
│   │                         # NOTE: everything inside data/ is generated locally and is NOT in git
│   ├── raw/                  # temporary FITS downloads (deleted automatically after processing)
│   ├── processed/            # one TIC_<id>.npz file per star (created by preprocess.py)
│   ├── full/                 # optional full 2-minute CSVs for single stars (--full mode)
│   ├── labels.csv            # star list: tic_id, label, source, train/test split (created by build_labels.py)
│   ├── failed.csv            # stars that could not be processed, with the reason
│   └── preprocess.log        # log of the preprocessing run
│
├── notebooks/
│   ├── 01_perprocessing.ipynb    # step-by-step demo of the preprocessing on a single star
│   └── 02_ml_classification.ipynb
│
├── src/
│   ├── build_labels.py       # builds data/labels.csv from public catalogs
│   ├── preprocess.py         # downloads and cleans light curves for every star in labels.csv
│   ├── feature_extraction.py
│   ├── train_classifier.py
│   └── evaluate.py
│
├── models/
├── results/
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

## Getting the Data (read this first)

The data files are **not stored in this repository**. `.gitignore` excludes `data/raw/`, `data/processed/`,
all `*.csv` files and `*.log` files, so after you pull you only have the code. You create the dataset on your
own machine by running two scripts, in this order, from the repository root:

```bash
python src/build_labels.py                       # Step 1: makes data/labels.csv   (about 1 minute)
python src/preprocess.py --workers 4             # Step 2: makes data/processed/   (about 20 to 30 minutes)
```

Run them exactly as shown, without changing any options (the defaults are 50 stars per class, sampled with a fixed seed), so that you get the same stars as everyone else. You need an internet
connection. Run Step 1 only once: running it again overwrites `data/labels.csv` with a new sampling.

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

## Pipeline Status

| Stage | Status |
|---|---|
| 1. Build labelled star list (`build_labels.py`) | Done |
| 2. Download and preprocess light curves (`preprocess.py`) | Done |
| 3. Feature extraction (`feature_extraction.py`) | In progress |
| 4. Classifier training and evaluation | Not started |

---

## Data Pipeline: What Was Done and Why

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

Implemented in `src/preprocess.py`. They are the same steps as the demo notebook, applied to every star:

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

- `flux`: detrended and normalized. Use this for features and classification.
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

### 7. Known limitations

- **Small "Other" class (24 stars).** Results for this class are only indicative. A fixed test split has just
  3 Other stars, so use stratified cross-validation over all stars and report per-class precision, recall
  and F1, not only accuracy.
- **Class imbalance (78 / 56 / 24).** Use balanced class weights.
- **Label noise.** Catalog labels are not perfect, and candidate (PC) stars are not all confirmed planets.
- **One sector per star.** Planets with long periods may show only one transit or none in a single sector.
- **Detrending bias.** Flattening without a transit mask slightly reduces transit depth.

---

## How to Run

Run all commands from the repository root.

### Step 1: Build the star list

```bash
python src/build_labels.py
```

Downloads the TOI table and the TESS-EBs catalog, assigns labels, samples stars and writes
`data/labels.csv`. It samples more stars than needed (1.6 times, so 80 per class here), because some stars
have no usable data. Run it once: it overwrites the existing `labels.csv`.

Useful options:

| Option | Meaning |
|---|---|
| `--n-per-class N` | target number of stars per class (default 50) |
| `--take Other=190` | sample an exact number for one class |
| `--max-toi 2000` | era matching by TOI number (`0` turns it off) |
| `--eb-csv file.csv` | use your own EB list (needs a TIC column) if the VizieR download fails |

### Step 2: Download and preprocess

```bash
python src/preprocess.py --workers 4
```

- Runs in parallel, retries failed downloads, and is **safe to stop (Ctrl+C) and re-run**. Stars that are
  already done are skipped.
- Writes `data/processed/TIC_<id>.npz` with the arrays `time`, `flux`, `flux_err`, `flux_norm`, `sector`.
- Stars that cannot be used are written to `data/failed.csv` with the reason. Stars with no 2-minute data,
  or whose first sector is after 26, are skipped on later runs. Other failures are retried automatically.
- Takes about 20 to 30 minutes for the full list. If some stars fail, run the same command again (see
  "Getting the Data" above).

A quick test with 10 stars per class:

```bash
python src/preprocess.py --per-class 10 --workers 4
```

If you see strange download errors, run with `--workers 1`.

### Optional: one star at full 2-minute cadence

For transit searches (BLS) or transit fitting on a single star:

```bash
python src/preprocess.py --tic 307210830 --sectors 8 9 10 11 12 --full
```

This writes `data/full/processed_TIC_307210830.csv` with the columns `time, flux, flux_err, flux_norm`.

### Loading the processed data in Python

Always read stars from `labels.csv`, because `data/processed/` may also contain files from earlier runs.

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from preprocess import load_star

labels = pd.read_csv("data/labels.csv")
star = load_star(labels.tic_id.iloc[0])     # DataFrame: time, flux, flux_err, flux_norm
```

## Data Sources

- TESS mission data (SPOC pipeline, 2-minute cadence) from MAST, accessed with
  [lightkurve](https://docs.lightkurve.org/).
- TESS Objects of Interest (TOI) table, NASA Exoplanet Archive.
- TESS Eclipsing Binary catalog: Prša et al. 2022, *TESS Eclipsing Binary Stars. I. Short-cadence
  Observations of 4584 Eclipsing Binaries in Sectors 1 to 26*, ApJS 258, 16 (VizieR `J/ApJS/258/16`).