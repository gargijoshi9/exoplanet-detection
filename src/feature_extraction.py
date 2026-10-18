
"""
Feature extraction module for Exoplanet Detection.

Extracts statistical features from processed TESS light curves
and combines them with the corresponding class labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_FILE = PROJECT_ROOT / "data" / "labels.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "features.csv"


# --------------------------------------------------
# 2. Extract features from one light curve
# --------------------------------------------------

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract statistical and variability features from a light curve.

    Expected columns:
        flux: Detrended flux values
        flux_norm: Normalized flux values

    Returns:
        A DataFrame containing one row of extracted features.
    """

    # Use detrended flux for feature extraction
    if "flux" in data.columns:
        flux = data["flux"].to_numpy(dtype=float)
    elif "flux_norm" in data.columns:
        flux = data["flux_norm"].to_numpy(dtype=float)
    else:
        raise ValueError("No flux or flux_norm column found.")

    # Remove NaN and infinite values
    flux = flux[np.isfinite(flux)]

    if len(flux) < 10:
        raise ValueError("Not enough valid flux measurements.")

    # Basic statistical features
    median = np.median(flux)

    features = {
        "mean": np.mean(flux),
        "std": np.std(flux),
        "min": np.min(flux),
        "max": np.max(flux),
        "median": median,
        "range": np.ptp(flux),
        "variance": np.var(flux),
        "mad": np.median(np.abs(flux - median)),
        "q25": np.percentile(flux, 25),
        "q75": np.percentile(flux, 75),
        "skewness": pd.Series(flux).skew(),
        "kurtosis": pd.Series(flux).kurtosis(),
    }

    # Variability features
    features.update({
        "mean_absolute_deviation": np.mean(
            np.abs(flux - np.mean(flux))
        ),
        "fraction_below_median": np.mean(flux < median),
        "fraction_below_1pct": np.mean(
            flux < np.percentile(flux, 1)
        ),
        "fraction_above_99pct": np.mean(
            flux > np.percentile(flux, 99)
        ),
    })

    # Changes between consecutive measurements
    differences = np.diff(flux)

    features.update({
        "mean_abs_diff": np.mean(np.abs(differences)),
        "std_diff": np.std(differences),
        "max_abs_diff": np.max(np.abs(differences)),
    })

    return pd.DataFrame([features])


# --------------------------------------------------
# 3. Load a processed light curve
# --------------------------------------------------

def load_lightcurve(npz_path: Path) -> pd.DataFrame:
    """
    Load a processed light curve saved by preprocess.py.
    """

    with np.load(npz_path) as data:
        return pd.DataFrame({
            "time": data["time"],
            "flux": data["flux"],
            "flux_err": data["flux_err"],
            "flux_norm": data["flux_norm"],
        })


# --------------------------------------------------
# 4. Build the complete feature dataset
# --------------------------------------------------

def build_feature_dataset() -> pd.DataFrame:
    """
    Extract features from all available light curves
    and attach their original labels and train/test splits.
    """

    labels = pd.read_csv(LABELS_FILE)
    all_features = []

    for _, star in labels.iterrows():

        tic_id = int(star["tic_id"])
        npz_path = PROCESSED_DIR / f"TIC_{tic_id}.npz"

        # Skip files that have not been downloaded successfully
        if not npz_path.exists():
            print(f"Skipping TIC {tic_id}: file not found.")
            continue

        try:
            lightcurve = load_lightcurve(npz_path)

            # Extract features for this star
            features = extract_features(lightcurve)

            # Attach metadata from labels.csv
            features["tic_id"] = tic_id
            features["label"] = star["label"]
            features["source"] = star["source"]
            features["split"] = star["split"]

            all_features.append(features)

            print(f"Processed TIC {tic_id}")

        except Exception as exc:
            print(f"Skipping TIC {tic_id}: {exc}")

    if not all_features:
        raise RuntimeError(
            "No features extracted. Check data/processed and labels.csv."
        )

    # Combine all individual feature rows
    result = pd.concat(all_features, ignore_index=True)

    # Save the dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature extraction completed!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total stars processed: {len(result)}")

    print("\nClass distribution by split:")
    print(result.groupby(["split", "label"]).size())

    return result


# --------------------------------------------------
# 5. Run the module directly
# --------------------------------------------------

if __name__ == "__main__":
    build_feature_dataset()
