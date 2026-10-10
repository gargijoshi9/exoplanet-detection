"""
Preprocessing for the TESS exoplanet project.

Same steps as notebooks/01_perprocessing.ipynb, wrapped in a function so it can run on many stars:
    download (2-min SPOC, PDCSAP) -> drop NaNs -> drop outliers -> flatten (Savitzky-Golay) -> normalize
and then (batch mode only) bin to 10-minute cadence so later steps (tsfresh) stay fast.

Batch mode: many stars from data/labels.csv, parallel, resumable (stars first observed after sector 26 are skipped)
    python src/preprocess.py --per-class 10 --workers 4     # pilot: 10 stars per class
    python src/preprocess.py --workers 4                    # full run (safe to stop and re-run)
    python src/preprocess.py --retry-failed                 # try the failed stars again

Single star, full 2-min cadence, for BLS / batman (writes a CSV)
    python src/preprocess.py --tic 307210830 --sectors 8 9 10 11 12 --full

Outputs
    data/processed/TIC_<id>.npz   time, flux, flux_err, flux_norm, sector   (batch mode)
    data/full/processed_TIC_<id>.csv   time, flux, flux_err, flux_norm      (--full mode)
    data/failed.csv               stars that failed, with the reason
    data/preprocess.log           log of the run

flux       = detrended + normalized (what BLS and tsfresh should use)
flux_norm  = normalized but NOT detrended (Stage 3 needs it to re-flatten with the transit masked)
"""

import argparse
import csv
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FULL_DIR = ROOT / "data" / "full"
LABELS_PATH = ROOT / "data" / "labels.csv"
FAILED_PATH = ROOT / "data" / "failed.csv"
LOG_PATH = ROOT / "data" / "preprocess.log"

# ---- same settings as the notebook ----
SIGMA_LOWER = 10        # remove very deep downward outliers only (keeps real transits)
SIGMA_UPPER = 5
WINDOW_LENGTH = 301     # cadences (about 10 hours at 2-min cadence)
POLYORDER = 2
BIN_MINUTES = 10
MIN_POINTS = 5000       # after removing NaNs; about 7 days of 2-min data
RETRIES = 3
MAX_SECTOR = 26         # TESS-EBs only covers sectors 1-26; keep every class in the same era

log = logging.getLogger("preprocess")
_failed_lock = threading.Lock()
_download_lock = threading.Lock()     # MAST search + download are not thread-safe (caused corrupt files)


class PreprocessError(Exception):
    """A star that cannot be used (no data, too few points, ...). Not retried."""


# ----------------------------------------------------------------------------
# Core processing (works on a lightkurve LightCurve)
# ----------------------------------------------------------------------------
def process_lightcurve(lc, bin_minutes=BIN_MINUTES):
    """Clean, detrend and normalize one light curve. Returns a dict of numpy arrays."""
    import astropy.units as u

    lc = lc.remove_nans()
    if len(lc) < MIN_POINTS:
        raise PreprocessError(f"too few points ({len(lc)} < {MIN_POINTS})")

    clean = lc.remove_outliers(sigma_lower=SIGMA_LOWER, sigma_upper=SIGMA_UPPER)
    flat = clean.flatten(window_length=WINDOW_LENGTH, polyorder=POLYORDER).normalize()
    norm = clean.normalize()

    if bin_minutes:
        size = bin_minutes * u.min
        flat = flat.bin(time_bin_size=size).remove_nans()
        norm = norm.bin(time_bin_size=size).remove_nans()

    if len(flat) != len(norm):
        raise PreprocessError("binned curves are misaligned")

    time_arr = np.asarray(flat.time.value, dtype="float64")
    flux = np.asarray(flat.flux.value, dtype="float64")
    flux_norm = np.asarray(norm.flux.value, dtype="float64")
    if flat.flux_err is not None:
        flux_err = np.asarray(flat.flux_err.value, dtype="float64")
    else:
        flux_err = np.full_like(flux, np.nan)
    bad = ~np.isfinite(flux_err)
    if bad.any():                                   # fall back to a robust scatter estimate
        sigma = 1.4826 * np.median(np.abs(flux - np.median(flux)))
        flux_err[bad] = sigma

    return {"time": time_arr, "flux": flux, "flux_err": flux_err, "flux_norm": flux_norm}


# ----------------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------------
def clear_raw(tic):
    """Delete leftover raw FITS files of one star (e.g. from an interrupted download)."""
    for d in (RAW_DIR / "mastDownload" / "TESS").glob(f"*{int(tic):016d}*"):
        shutil.rmtree(d, ignore_errors=True)


def fetch_lightcurve(tic, sectors=None, keep_raw=False, max_sector=None):
    """Download 2-min SPOC PDCSAP light curve(s) for one star. Default: the earliest sector only."""
    import lightkurve as lk

    if not keep_raw:
        clear_raw(tic)       # a half-downloaded file from an earlier attempt would make lightkurve fail

    kw = dict(mission="TESS", author="SPOC", exptime=120)
    with _download_lock:     # search + download one star at a time; processing still runs in parallel
        if sectors:
            res = lk.search_lightcurve(f"TIC {tic}", sector=list(sectors), **kw)
        else:
            res = lk.search_lightcurve(f"TIC {tic}", **kw)
        if len(res) == 0:
            raise PreprocessError("no 2-min SPOC light curve found")

        if not sectors:                              # pick one sector: the earliest
            seq = np.asarray(res.table["sequence_number"], dtype=int)
            if max_sector and seq.min() > max_sector:
                raise PreprocessError(f"first sector {seq.min()} > {max_sector} (outside matched era)")
            res = res[int(np.argmin(seq))]

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        lcs = res.download_all(download_dir=str(RAW_DIR), quality_bitmask="default",
                               flux_column="pdcsap_flux")
    if lcs is None or len(lcs) == 0:
        raise RuntimeError("download returned nothing")

    sector_nums = [int(l.meta.get("SECTOR", -1)) for l in lcs]
    files = [l.meta.get("FILENAME") for l in lcs]
    lc = lcs.stitch()

    if not keep_raw:                                 # save disk space
        for f in files:
            try:
                if f:
                    Path(f).unlink()
            except OSError:
                pass
    return lc, sector_nums


# ----------------------------------------------------------------------------
# One star, with retries
# ----------------------------------------------------------------------------
def npz_path(tic):
    return PROC_DIR / f"TIC_{int(tic)}.npz"


def process_star(tic, keep_raw=False, max_sector=None):
    """Download + process + save one star (batch mode). Returns number of points saved."""
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            lc, sectors = fetch_lightcurve(tic, keep_raw=keep_raw, max_sector=max_sector)
            data = process_lightcurve(lc, bin_minutes=BIN_MINUTES)
            PROC_DIR.mkdir(parents=True, exist_ok=True)
            tmp = PROC_DIR / f"TIC_{int(tic)}.tmp.npz"       # write to a temp file, then rename
            np.savez_compressed(
                tmp,
                time=data["time"],
                flux=data["flux"].astype("float32"),
                flux_err=data["flux_err"].astype("float32"),
                flux_norm=data["flux_norm"].astype("float32"),
                sector=np.array(sectors, dtype="int16"),
            )
            os.replace(tmp, npz_path(tic))
            return len(data["time"])
        except PreprocessError:
            raise
        except Exception as e:                       # network errors, MAST hiccups, ...
            last_err = e
            log.warning("TIC %s attempt %d/%d failed: %r", tic, attempt, RETRIES, str(e)[:120])
            if not keep_raw:
                clear_raw(tic)
            time.sleep(5 * attempt)
    raise RuntimeError(f"gave up after {RETRIES} attempts: {last_err!r}")


def record_failure(tic, label, reason):
    with _failed_lock:
        new = not FAILED_PATH.exists()
        FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILED_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tic_id", "label", "reason"])
            w.writerow([tic, label, str(reason).replace("\n", " ")[:200]])


def load_failed():
    if not FAILED_PATH.exists():
        return set()
    f = pd.read_csv(FAILED_PATH)
    reason = f["reason"].astype(str)
    permanent = reason.str.startswith("no 2-min") | reason.str.startswith("first sector")  # will never change
    return set(f.loc[permanent, "tic_id"].astype("int64"))


# ----------------------------------------------------------------------------
# Batch mode
# ----------------------------------------------------------------------------
def run_batch(args):
    if not LABELS_PATH.exists():
        sys.exit(f"{LABELS_PATH} not found. Run: python src/build_labels.py")
    labels = pd.read_csv(LABELS_PATH)

    if args.per_class:
        labels = labels.groupby("label", group_keys=False).head(args.per_class)

    # interleave the classes, so stopping early still leaves a balanced set of stars
    labels = (labels.assign(_rank=labels.groupby("label").cumcount())
                    .sort_values(["_rank", "label"]).drop(columns="_rank"))

    if args.max_sector:                      # drop old files whose sector is outside the matched era
        for t in labels.tic_id:
            if npz_path(t).exists():
                with np.load(npz_path(t)) as z:
                    sec = int(z["sector"].min())
                if sec > args.max_sector:
                    npz_path(t).unlink()
                    record_failure(int(t), "", f"first sector {sec} > {args.max_sector} (outside matched era)")
                    log.info("TIC %s: removed old file (sector %d is outside the matched era)", t, sec)

    failed = set() if args.retry_failed else load_failed()
    todo = [(int(r.tic_id), r.label) for r in labels.itertuples()
            if not npz_path(r.tic_id).exists() and int(r.tic_id) not in failed]
    already = int(sum(npz_path(t).exists() for t in labels.tic_id))
    log.info("%d stars selected: %d already done, %d previously failed (skipped), %d to process",
             len(labels), already, len(failed & set(labels.tic_id)), len(todo))
    if not todo:
        return

    done = errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_star, tic, args.keep_raw, args.max_sector or None): (tic, label) for tic, label in todo}
        for fut in as_completed(futures):
            tic, label = futures[fut]
            try:
                n = fut.result()
                done += 1
                log.info("[%d/%d] TIC %s (%s): ok, %d points", done + errors, len(todo), tic, label, n)
            except Exception as e:
                errors += 1
                record_failure(tic, label, e)
                log.info("[%d/%d] TIC %s (%s): FAILED - %s", done + errors, len(todo), tic, label, e)

    mins = (time.time() - t0) / 60
    log.info("Finished in %.1f min: %d ok, %d failed", mins, done, errors)

    ok = labels[[npz_path(t).exists() for t in labels.tic_id]]
    log.info("Stars available per class and split:\n%s",
             ok.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())


# ----------------------------------------------------------------------------
# Single-star, full-cadence mode (for detection / fitting)
# ----------------------------------------------------------------------------
def run_full(args):
    lc, sectors = fetch_lightcurve(args.tic, sectors=args.sectors, keep_raw=args.keep_raw)
    data = process_lightcurve(lc, bin_minutes=0)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    out = FULL_DIR / f"processed_TIC_{args.tic}.csv"
    pd.DataFrame(data).to_csv(out, index=False)
    log.info("Sectors %s -> %d points (2-min cadence) saved to %s", sectors, len(data["time"]), out)


# ----------------------------------------------------------------------------
# Helper for later stages
# ----------------------------------------------------------------------------
def load_star(tic) -> pd.DataFrame:
    """Load one processed star (batch mode) as a DataFrame: time, flux, flux_err, flux_norm."""
    with np.load(npz_path(tic)) as z:
        return pd.DataFrame({k: z[k] for k in ("time", "flux", "flux_err", "flux_norm")})


def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH)):
        h.setFormatter(fmt)
        log.addHandler(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (default 4)")
    ap.add_argument("--per-class", type=int, default=None, help="only the first N stars per class (pilot)")
    ap.add_argument("--retry-failed", action="store_true", help="also retry stars that have no 2-min data")
    ap.add_argument("--max-sector", type=int, default=MAX_SECTOR,
                    help="batch mode: skip stars whose first 2-min sector is above this (default 26; 0 = off)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded FITS files")
    ap.add_argument("--tic", type=int, default=None, help="single-star mode: TIC id")
    ap.add_argument("--sectors", type=int, nargs="+", default=None, help="single-star mode: sectors to use")
    ap.add_argument("--full", action="store_true", help="single-star mode: keep 2-min cadence, write a CSV")
    args = ap.parse_args()

    setup_logging()
    if args.tic is not None:
        if not args.full:
            ap.error("--tic needs --full (batch mode reads stars from data/labels.csv)")
        run_full(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()