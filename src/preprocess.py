"""
Preprocessing for the TESS exoplanet project (ExoScope).

Pipeline steps:
    1. Filter & search observations (2-min SPOC, PDCSAP)
    2. Deterministically select the earliest valid sector <= 26
    3. Download target FITS payload with default quality bitmask
    4. Clean light curve: drop NaNs -> drop outliers -> flatten (Savitzky-Golay) -> normalize
    5. Bin (batch mode only) to 10-minute cadence for fast downstream feature extraction
    6. Atomically save compressed .npz archives and validate array integrity

Batch mode: many stars from data/labels.csv, parallel, resumable
    python src/preprocess.py --per-class 10 --workers 4     # pilot: 10 stars per class
    python src/preprocess.py --workers 4                    # full run (safe to stop and re-run)
    python src/preprocess.py --retry-failed                 # retry previously failed stars

Single star, full 2-min cadence, for BLS / parameter fitting (writes a CSV):
    python src/preprocess.py --tic 307210830 --sectors 8 9 10 11 12 --full

Outputs:
    data/processed/TIC_<id>.npz        time, flux, flux_err, flux_norm, sector (batch mode)
    data/full/processed_TIC_<id>.csv   time, flux, flux_err, flux_norm         (--full mode)
    data/failed.csv                    stars that failed, with deduplicated diagnostic reason
    data/preprocess.log                log of the run

Arrays:
    flux       = detrended + normalized (for BLS transit detection and ML features)
    flux_norm  = normalized but NOT detrended (for transit parameter fitting with batman)
    flux_err   = measurement uncertainty (or robust scatter estimate if absent)
    sector     = observation sector numbers
"""

from __future__ import annotations

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

# Core preprocessing parameters
SIGMA_LOWER = 10        # remove deep downward outliers only (keeps real transits)
SIGMA_UPPER = 5
WINDOW_LENGTH = 301     # cadences (about 10 hours at 2-min cadence)
POLYORDER = 2
BIN_MINUTES = 10
MIN_POINTS = 5000       # after removing NaNs; about 7 days of 2-min data
RETRIES = 3
MAX_SECTOR = 26         # Primary mission era constraint (Sectors 1-26)

log = logging.getLogger("preprocess")
_failed_lock = threading.Lock()
_download_lock = threading.Lock()     # MAST search + download thread safety

REQUIRED_NPZ_KEYS = ("time", "flux", "flux_err", "flux_norm", "sector")


class PreprocessError(Exception):
    """A star that cannot be used (no data, wrong sector, too few points, malformed metadata)."""


# ----------------------------------------------------------------------------
# Processed file validation and caching helpers
# ----------------------------------------------------------------------------
def validate_processed_npz(path: Path) -> tuple[bool, str]:
    """
    Validate that an existing .npz file is uncorrupted, contains all required
    arrays with matching dimensions, and contains usable finite values.
    Returns (True, "valid") or (False, reason).
    """
    if not path.is_file():
        return False, "file does not exist"
    try:
        size = path.stat().st_size
    except OSError as e:
        return False, f"cannot stat file: {e}"
    if size == 0:
        return False, "file is empty (0 bytes)"

    try:
        with np.load(path) as data:
            keys = set(data.files)
            missing_keys = [k for k in REQUIRED_NPZ_KEYS if k not in keys]
            if missing_keys:
                return False, f"missing required keys: {', '.join(missing_keys)}"

            time_arr = data["time"]
            flux_arr = data["flux"]
            err_arr = data["flux_err"]
            norm_arr = data["flux_norm"]
            sector_arr = data["sector"]

            if time_arr.ndim != 1 or flux_arr.ndim != 1 or err_arr.ndim != 1 or norm_arr.ndim != 1:
                return False, "photometric arrays must be 1-dimensional"

            n = len(time_arr)
            if n == 0:
                return False, "photometric arrays are empty"

            if len(flux_arr) != n or len(err_arr) != n or len(norm_arr) != n:
                return False, (
                    f"array length mismatch (time={n}, flux={len(flux_arr)}, "
                    f"err={len(err_arr)}, norm={len(norm_arr)})"
                )

            if not np.isfinite(time_arr).all():
                return False, "time array contains NaN or non-finite values"

            if not np.isfinite(flux_arr).any() or not np.isfinite(norm_arr).any():
                return False, "flux array contains no finite data points"

            if sector_arr.size == 0:
                return False, "sector metadata array is empty"

    except Exception as e:
        return False, f"corrupt npz archive ({type(e).__name__}: {e})"

    return True, "valid"


def is_valid_npz(path: Path) -> bool:
    """Return True if path is a valid processed .npz file, False otherwise."""
    valid, _ = validate_processed_npz(path)
    return valid


# ----------------------------------------------------------------------------
# Failure log management
# ----------------------------------------------------------------------------
def is_permanent_failure(reason: str | Exception) -> bool:
    """Check if a failure is an unrecoverable scientific/data incompatibility."""
    r = str(reason).lower().strip()
    return (
        r.startswith("no 2-min")
        or r.startswith("first sector")
        or "outside matched era" in r
        or "too few points" in r
        or "missing or malformed sector" in r
        or "no usable pdcsap_flux" in r
        or "no finite pdcsap_flux" in r
        or "misaligned" in r
    )


def record_failure(tic: int, label: str, reason: str | Exception):
    """
    Record or update a target failure in FAILED_PATH safely and without duplicate rows.
    Preserves existing order, replaces existing reason for the TIC, and writes atomically.
    """
    clean_reason = str(reason).replace("\n", " ").replace("\r", " ").strip()[:200]
    tic_int = int(tic)

    with _failed_lock:
        rows: dict[int, dict] = {}
        if FAILED_PATH.exists():
            try:
                with open(FAILED_PATH, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        try:
                            tid = int(r["tic_id"])
                            rows[tid] = {
                                "tic_id": tid,
                                "label": r.get("label", ""),
                                "reason": r.get("reason", ""),
                            }
                        except (ValueError, TypeError, KeyError):
                            continue
            except Exception as e:
                log.warning("Error reading %s for update: %s", FAILED_PATH, e)

        # Update or insert entry
        rows[tic_int] = {
            "tic_id": tic_int,
            "label": label or (rows.get(tic_int, {}).get("label", "")),
            "reason": clean_reason,
        }

        FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = FAILED_PATH.with_suffix(".tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tic_id", "label", "reason"])
            writer.writeheader()
            for r in rows.values():
                writer.writerow(r)
        os.replace(tmp_path, FAILED_PATH)


def resolve_failure(tic: int):
    """Remove a target from FAILED_PATH when it has been successfully processed."""
    tic_int = int(tic)
    with _failed_lock:
        if not FAILED_PATH.exists():
            return
        try:
            with open(FAILED_PATH, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            return

        new_rows = []
        found = False
        for r in rows:
            try:
                if int(r["tic_id"]) == tic_int:
                    found = True
                    continue
            except (ValueError, TypeError, KeyError):
                pass
            new_rows.append(r)

        if found:
            tmp_path = FAILED_PATH.with_suffix(".tmp")
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["tic_id", "label", "reason"])
                writer.writeheader()
                writer.writerows(new_rows)
            os.replace(tmp_path, FAILED_PATH)
            log.info("TIC %s: cleared previous failure record following successful processing.", tic_int)


def load_failed(include_temporary: bool = False) -> set[int]:
    """
    Load failed TIC IDs from FAILED_PATH.
    By default returns only permanently failed TICs (which will not be retried unless requested).
    If include_temporary=True, returns all failed TICs.
    """
    if not FAILED_PATH.exists():
        return set()
    try:
        f = pd.read_csv(FAILED_PATH)
        if "tic_id" not in f.columns or "reason" not in f.columns:
            return set()
        if include_temporary:
            return set(f["tic_id"].dropna().astype("int64"))

        permanent_mask = f["reason"].astype(str).map(is_permanent_failure)
        return set(f.loc[permanent_mask, "tic_id"].dropna().astype("int64"))
    except Exception as e:
        log.warning("Could not read %s: %s", FAILED_PATH, e)
        return set()


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
    if len(clean) < 7:
        raise PreprocessError(f"too few points after outlier removal ({len(clean)})")

    # Adjust window length dynamically if light curve has fewer points than window
    wl = min(WINDOW_LENGTH, len(clean) - 1)
    if wl % 2 == 0:
        wl -= 1
    if wl < 7:
        raise PreprocessError(f"light curve too short for detrending filter ({len(clean)} points)")

    flat = clean.flatten(window_length=wl, polyorder=POLYORDER).normalize()
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
# Observation selection and downloading
# ----------------------------------------------------------------------------
def clear_raw(tic):
    """Delete leftover raw FITS files of one star (e.g. from an interrupted download)."""
    for d in (RAW_DIR / "mastDownload" / "TESS").glob(f"*{int(tic):016d}*"):
        shutil.rmtree(d, ignore_errors=True)


def extract_sectors_from_search(res) -> list[tuple[int, int]]:
    """
    Safely extract sector numbers from a Lightkurve SearchResult.
    Returns list of (sector_int, row_index).
    """
    table = getattr(res, "table", None)
    if table is None or len(table) == 0:
        raise PreprocessError("search result table is empty or missing")

    col = None
    for candidate in ("sequence_number", "sector", "SECTOR"):
        if candidate in table.colnames:
            col = candidate
            break

    if col is None:
        raise PreprocessError("missing or malformed sector metadata")

    sectors_with_idx = []
    for idx, val in enumerate(table[col]):
        try:
            if hasattr(val, "mask") and val.mask:
                continue
            if val is None or val == "" or str(val).strip().lower() in ("nan", "none"):
                continue
            sectors_with_idx.append((int(val), idx))
        except (ValueError, TypeError):
            continue

    if not sectors_with_idx:
        raise PreprocessError("missing or malformed sector metadata")

    return sectors_with_idx


def select_observation(res, max_sector=MAX_SECTOR):
    """
    Deterministically select the earliest observation within max_sector.
    If no observations are within max_sector, raises PreprocessError with the earliest sector.
    """
    sectors_with_idx = extract_sectors_from_search(res)
    all_sectors = [sec for sec, _ in sectors_with_idx]

    if max_sector and max_sector > 0:
        valid_observations = [(sec, idx) for sec, idx in sectors_with_idx if sec <= max_sector]
        if not valid_observations:
            earliest = min(all_sectors)
            raise PreprocessError(f"first sector {earliest} > {max_sector} (outside matched era)")
    else:
        valid_observations = sectors_with_idx

    # Sort deterministically by (sector_number, table_index)
    valid_observations.sort(key=lambda item: (item[0], item[1]))
    best_sector, best_idx = valid_observations[0]
    return res[best_idx], best_sector


def fetch_lightcurve(tic, sectors=None, keep_raw=False, max_sector=MAX_SECTOR):
    """
    Download 2-min SPOC PDCSAP light curve(s) for one star.
    Default: the earliest sector <= max_sector.
    """
    import lightkurve as lk

    if not keep_raw:
        clear_raw(tic)       # clear half-downloaded files from earlier attempts

    kw = dict(mission="TESS", author="SPOC", exptime=120)
    with _download_lock:     # search + download one star at a time
        if sectors:
            res = lk.search_lightcurve(f"TIC {tic}", sector=list(sectors), **kw)
            if len(res) == 0:
                raise PreprocessError(f"no 2-min SPOC light curve found for sectors {sectors}")
            selected_res = res
        else:
            res = lk.search_lightcurve(f"TIC {tic}", **kw)
            if len(res) == 0:
                raise PreprocessError("no 2-min SPOC light curve found")
            selected_res, _ = select_observation(res, max_sector=max_sector)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        lcs = selected_res.download_all(
            download_dir=str(RAW_DIR),
            quality_bitmask="default",
            flux_column="pdcsap_flux",
        )

    if lcs is None or len(lcs) == 0:
        raise RuntimeError("download returned nothing")

    # Extract sector numbers and file paths safely
    sector_nums = []
    files = []
    for l in lcs:
        meta = getattr(l, "meta", {})
        sec = meta.get("SECTOR")
        if sec is not None:
            try:
                sector_nums.append(int(sec))
            except (ValueError, TypeError):
                pass
        fn = meta.get("FILENAME")
        if fn:
            files.append(fn)

    lc = lcs.stitch() if len(lcs) > 1 else lcs[0]

    # Validate usable pdcsap_flux data
    flux_val = getattr(lc, "flux", None)
    if flux_val is None or len(flux_val) == 0:
        raise PreprocessError("downloaded light curve has no usable pdcsap_flux")

    flux_arr = np.asarray(flux_val.value if hasattr(flux_val, "value") else flux_val, dtype=float)
    if not np.isfinite(flux_arr).any():
        raise PreprocessError("downloaded light curve has no finite pdcsap_flux")

    if not keep_raw:                                 # save disk space
        for f in files:
            try:
                if f:
                    p = Path(f)
                    if p.exists() and p.is_file():
                        p.unlink()
                    parent = p.parent
                    if RAW_DIR in parent.parents and parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
            except OSError:
                pass

    return lc, sector_nums


# ----------------------------------------------------------------------------
# One star processing with retries and atomic output
# ----------------------------------------------------------------------------
def npz_path(tic) -> Path:
    return PROC_DIR / f"TIC_{int(tic)}.npz"


def process_star(
    tic: int,
    keep_raw: bool = False,
    max_sector: int | None = None,
    retries: int = RETRIES,
    base_backoff: float = 5.0,
    max_backoff: float = 30.0,
) -> int:
    """Download + process + save one star (batch mode). Returns number of points saved."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            lc, sectors = fetch_lightcurve(tic, keep_raw=keep_raw, max_sector=max_sector)
            data = process_lightcurve(lc, bin_minutes=BIN_MINUTES)
            PROC_DIR.mkdir(parents=True, exist_ok=True)

            pid = os.getpid()
            tid = threading.get_ident()
            tmp = PROC_DIR / f"TIC_{int(tic)}.tmp_{pid}_{tid}.npz"
            np.savez_compressed(
                tmp,
                time=data["time"],
                flux=data["flux"].astype("float32"),
                flux_err=data["flux_err"].astype("float32"),
                flux_norm=data["flux_norm"].astype("float32"),
                sector=np.array(sectors, dtype="int16"),
            )

            is_valid, reason = validate_processed_npz(tmp)
            if not is_valid:
                tmp.unlink(missing_ok=True)
                raise PreprocessError(f"saved npz failed validation: {reason}")

            os.replace(tmp, npz_path(tic))
            resolve_failure(int(tic))
            return len(data["time"])
        except PreprocessError:
            # Permanent data/astronomical incompatibility: do not retry
            raise
        except Exception as e:                       # transient network errors, MAST drops, etc.
            last_err = e
            log.warning("TIC %s attempt %d/%d failed: %r", tic, attempt, retries, str(e)[:120])
            if not keep_raw:
                clear_raw(tic)
            if attempt < retries:
                delay = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                time.sleep(delay)

    raise RuntimeError(f"gave up after {retries} attempts: {last_err!r}")


# ----------------------------------------------------------------------------
# Batch mode
# ----------------------------------------------------------------------------
def print_summary_table(labels: pd.DataFrame):
    ok_mask = [is_valid_npz(npz_path(t)) for t in labels.tic_id]
    ok = labels[ok_mask]
    if len(ok) == 0:
        log.info("No processed stars currently available.")
        return
    table_str = ok.groupby(["label", "split"]).size().unstack(fill_value=0).to_string()
    log.info("Stars available per class and split:\n%s", table_str)


def run_batch(args):
    if not LABELS_PATH.exists():
        sys.exit(f"{LABELS_PATH} not found. Run: python src/build_labels.py")
    labels = pd.read_csv(LABELS_PATH)

    if args.per_class:
        labels = labels.groupby("label", group_keys=False).head(args.per_class)

    # Interleave classes so stopping early still leaves a balanced set of stars
    labels = (
        labels.assign(_rank=labels.groupby("label").cumcount())
        .sort_values(["_rank", "label"])
        .drop(columns="_rank")
    )

    # Validate existing cache files and handle max_sector constraint
    valid_cached: set[int] = set()
    corrupt_cached: list[tuple[int, str]] = []
    out_of_era: list[tuple[int, int]] = []

    for t in labels.tic_id:
        p = npz_path(t)
        if p.exists():
            is_valid, reason = validate_processed_npz(p)
            if not is_valid:
                corrupt_cached.append((int(t), reason))
                log.warning("TIC %s: existing cache is invalid (%s). Will reprocess.", t, reason)
                p.unlink(missing_ok=True)
                continue

            if args.max_sector and args.max_sector > 0:
                with np.load(p) as z:
                    sec = int(z["sector"].min())
                if sec > args.max_sector:
                    p.unlink(missing_ok=True)
                    out_of_era.append((int(t), sec))
                    record_failure(int(t), "", f"first sector {sec} > {args.max_sector} (outside matched era)")
                    log.info("TIC %s: removed old file (sector %d is outside the matched era)", t, sec)
                    continue

            valid_cached.add(int(t))

    failed_skipped = set() if args.retry_failed else load_failed(include_temporary=False)
    matched_failed = failed_skipped & set(labels.tic_id.astype(int))

    todo = [
        (int(r.tic_id), r.label)
        for r in labels.itertuples()
        if int(r.tic_id) not in valid_cached and int(r.tic_id) not in failed_skipped
    ]

    total_targets = len(labels)
    total_todo = len(todo)

    print("\n" + "=" * 65)
    print(" ExoScope TESS Preprocessing Pipeline ")
    print("=" * 65)
    print(f"Total targets in catalog:         {total_targets}")
    print(f"Already valid in cache (skipped): {len(valid_cached)}")
    print(f"Permanent failures (skipped):     {len(matched_failed)}")
    if corrupt_cached:
        print(f"Corrupt cache files removed:      {len(corrupt_cached)}")
    if out_of_era:
        print(f"Out-of-era files removed:         {len(out_of_era)}")
    print(f"Targets to process this run:      {total_todo}")
    print("=" * 65 + "\n")

    log.info(
        "%d stars in catalog: %d already valid, %d previously failed (skipped), %d to process",
        total_targets,
        len(valid_cached),
        len(matched_failed),
        total_todo,
    )

    if not todo:
        log.info("No targets to process. Everything is up to date.")
        print_summary_table(labels)
        return

    done_ok = 0
    perm_errors = 0
    temp_errors = 0
    t0 = time.time()

    retries_count = getattr(args, "retries", RETRIES)
    backoff_base = getattr(args, "backoff", 5.0)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_star,
                tic,
                args.keep_raw,
                args.max_sector or None,
                retries_count,
                backoff_base,
            ): (tic, label)
            for tic, label in todo
        }

        for fut in as_completed(futures):
            tic, label = futures[fut]
            total_completed = done_ok + perm_errors + temp_errors + 1
            progress_pct = (total_completed / total_todo) * 100.0
            try:
                n = fut.result()
                done_ok += 1
                log.info(
                    "[%d/%d (%.1f%%)] TIC %s (%s): OK, %d points",
                    total_completed,
                    total_todo,
                    progress_pct,
                    tic,
                    label,
                    n,
                )
            except PreprocessError as e:
                perm_errors += 1
                record_failure(tic, label, e)
                log.info(
                    "[%d/%d (%.1f%%)] TIC %s (%s): PERMANENT FAILURE - %s",
                    total_completed,
                    total_todo,
                    progress_pct,
                    tic,
                    label,
                    e,
                )
            except Exception as e:
                temp_errors += 1
                record_failure(tic, label, e)
                log.info(
                    "[%d/%d (%.1f%%)] TIC %s (%s): TEMPORARY FAILURE - %s",
                    total_completed,
                    total_todo,
                    progress_pct,
                    tic,
                    label,
                    e,
                )

    mins = (time.time() - t0) / 60
    print("\n" + "=" * 65)
    print(" Preprocessing Run Completed ")
    print("=" * 65)
    print(f"Time elapsed:                {mins:.2f} minutes")
    print(f"Successfully processed:      {done_ok}")
    print(f"Permanent incompatibilities: {perm_errors}")
    print(f"Temporary failures:          {temp_errors}")
    print("=" * 65 + "\n")
    log.info(
        "Finished in %.1f min: %d ok, %d permanent failed, %d temporary failed",
        mins,
        done_ok,
        perm_errors,
        temp_errors,
    )

    print_summary_table(labels)


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
    p = npz_path(tic)
    is_valid, reason = validate_processed_npz(p)
    if not is_valid:
        raise ValueError(f"Processed light curve for TIC {tic} is invalid or missing: {reason}")
    with np.load(p) as z:
        return pd.DataFrame({k: z[k] for k in ("time", "flux", "flux_err", "flux_norm")})


def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    # Prevent duplicated handlers if called multiple times
    if not log.handlers:
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH)):
            h.setFormatter(fmt)
            log.addHandler(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (default 4)")
    ap.add_argument("--per-class", type=int, default=None, help="only the first N stars per class (pilot)")
    ap.add_argument("--retry-failed", action="store_true", help="also retry stars that failed previously")
    ap.add_argument(
        "--max-sector",
        type=int,
        default=MAX_SECTOR,
        help="batch mode: skip stars whose first 2-min sector is above this (default 26; 0 = off)",
    )
    ap.add_argument("--retries", type=int, default=RETRIES, help=f"number of retries on transient errors (default {RETRIES})")
    ap.add_argument("--backoff", type=float, default=5.0, help="base exponential backoff delay in seconds (default 5.0)")
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