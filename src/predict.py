
from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lightgbm_bls_pipeline.joblib"

# Reuse the project's existing implementations.
sys.path.insert(0, str(ROOT / "src"))
from feature_extraction import extract_features
from detect import detect


BLS_FEATURES = [
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


def load_lightcurve(npz_path):
    """Load a processed light curve without assuming optional fields exist."""
    with np.load(npz_path) as data:
        required = {"time", "flux"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing required NPZ fields: {sorted(missing)}")

        arrays = {
            "time": data["time"].astype(float),
            "flux": data["flux"].astype(float),
        }

        if "flux_err" in data.files:
            arrays["flux_err"] = data["flux_err"].astype(float)

        if "flux_norm" in data.files:
            arrays["flux_norm"] = data["flux_norm"].astype(float)

    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Light-curve arrays have inconsistent lengths.")

    return pd.DataFrame(arrays)


def extract_bls_features(lightcurve):
    """Run BLS and return the first detected signal's features."""
    table, _ = detect(
        lightcurve["time"].to_numpy(),
        lightcurve["flux"].to_numpy(),
        flux_err=(
            lightcurve["flux_err"].to_numpy()
            if "flux_err" in lightcurve.columns
            else None
        ),
        verbose=False,
    )

    values = {name: np.nan for name in BLS_FEATURES}
    diagnostics = {
        "detection_found": False,
        "status": "no_detection",
        "flags": [],
    }

    if table.empty:
        return values, diagnostics

    # Consistent with the training feature extraction:
    # use the first (strongest/first searched) BLS signal.
    signal = table.iloc[0]

    for name in BLS_FEATURES:
        if name in signal.index:
            value = pd.to_numeric(signal[name], errors="coerce")
            values[name] = float(value) if pd.notna(value) else np.nan

    flags = signal.get("flags", "")
    if pd.isna(flags):
        flags = ""

    diagnostics = {
        "detection_found": True,
        "status": str(signal.get("status", "unknown")),
        "flags": [
            flag.strip()
            for flag in str(flags).split(",")
            if flag.strip()
        ],
    }

    return values, diagnostics


def predict_lightcurve(npz_path):
    """Predict the class of one processed TESS light curve."""
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"Light curve not found: {npz_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}. "
            "Run python3 src/train_final_model.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_names = artifact["features"]
    medians = artifact["medians"]

    lightcurve = load_lightcurve(npz_path)

    # Reuse the exact statistical feature implementation used for training.
    statistical = extract_features(lightcurve).iloc[0].to_dict()

    bls_values, diagnostics = extract_bls_features(lightcurve)

    all_features = {**statistical, **bls_values}
    X = pd.DataFrame(
        [[all_features.get(name, np.nan) for name in feature_names]],
        columns=feature_names,
    )

    # Use medians saved from the training partition only.
    X = X.fillna(pd.Series(medians)).fillna(0)

    predicted_class = str(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    class_probabilities = {
        str(class_name): round(float(probability), 4)
        for class_name, probability in zip(model.classes_, probabilities)
    }

    # This is a model score, not a calibrated probability of correctness.
    result = {
        "tic_id": npz_path.stem.removeprefix("TIC_"),
        "predicted_class": predicted_class,
        "class_probabilities": class_probabilities,
        "bls": {
            key: (
                round(float(value), 6)
                if np.isfinite(value)
                else None
            )
            for key, value in bls_values.items()
        },
        "bls_diagnostics": diagnostics,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Classify one processed TESS light curve."
    )
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to a processed TIC_*.npz light-curve file.",
    )
    args = parser.parse_args()

    try:
        result = predict_lightcurve(args.npz)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")


if __name__ == "__main__":
    main()
