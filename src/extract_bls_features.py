
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Allow importing detect.py when this script is run from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detect import detect


INPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "features"
OUTPUT_PATH = OUTPUT_DIR / "bls_features.csv"

# Numeric BLS measurements we want to investigate as classifier features.
BLS_NUMERIC_COLUMNS = [
    "period_d",
    "duration_h",
    "depth_ppm",
    "snr",
    "snr_red",
    "sde",
    "n_transits",
    "odd_even_sigma",
    "secondary_sigma",
]

# Metadata/diagnostics: do not automatically use these as classifier features.
DIAGNOSTIC_COLUMNS = ["accepted", "status", "flags"]


def empty_result(tic_id, error=None):
    """Return a consistent row when no usable detection is available."""
    row = {"tic_id": tic_id, "detection_found": 0}

    for col in BLS_NUMERIC_COLUMNS:
        row[col] = np.nan

    row.update({
        "accepted": np.nan,
        "status": "error" if error else "no_detection",
        "flags": error or "",
    })
    return row


def extract_one(npz_path):
    """Run the existing BLS detector on one processed light curve."""
    tic_id = npz_path.stem.replace("TIC_", "")

    try:
        with np.load(npz_path) as data:
            if "time" not in data or "flux" not in data:
                return empty_result(
                    tic_id, "missing_time_or_flux"
                )

            time_arr = data["time"].astype(float)
            flux_arr = data["flux"].astype(float)

            flux_err = (
                data["flux_err"].astype(float)
                if "flux_err" in data
                else None
            )

        table, _ = detect(
            time_arr,
            flux_arr,
            flux_err=flux_err,
            verbose=False,
        )

        if table.empty:
            return empty_result(tic_id)

        # The first result is the strongest signal found in the first BLS
        # iteration. The detector may stop after rejecting that signal.
        signal = table.iloc[0]
        row = {
            "tic_id": tic_id,
            "detection_found": 1,
        }

        for col in BLS_NUMERIC_COLUMNS:
            row[col] = (
                signal[col] if col in signal.index else np.nan
            )

        row.update({
            "accepted": bool(signal["accepted"]),
            "status": str(signal["status"]),
            "flags": str(signal["flags"]),
        })
        return row

    except Exception as exc:
        return empty_result(
            tic_id,
            f"{type(exc).__name__}: {exc}",
        )


def main():
    files = sorted(INPUT_DIR.glob("TIC_*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No TIC_*.npz files found in {INPUT_DIR}"
        )

    print(f"Found {len(files)} processed light curves.")
    print("Running BLS detection; this may take a while.")

    rows = []
    start = time.time()

    for i, path in enumerate(files, start=1):
        row = extract_one(path)
        rows.append(row)

        if i % 10 == 0 or i == len(files):
            print(
                f"Processed {i}/{len(files)} | "
                f"elapsed: {(time.time() - start) / 60:.1f} min"
            )

    result = pd.DataFrame(rows)

    # Keep IDs as strings for a safe join with the classification table.
    result["tic_id"] = result["tic_id"].astype(str)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nExtraction complete.")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(result)}")
    print(f"Signals found: {int(result['detection_found'].sum())}")
    print(f"No signal: {(result['status'] == 'no_detection').sum()}")
    print(f"Errors: {(result['status'] == 'error').sum()}")

    print("\nDetection status counts:")
    print(result["status"].value_counts(dropna=False).to_string())

    print("\nPreview:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
