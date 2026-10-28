
# src/train_final_model.py

from pathlib import Path
import joblib
import pandas as pd

from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = ROOT / "data" / "features.csv"
BLS_PATH = ROOT / "data" / "features" / "bls_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lightgbm_bls_pipeline.joblib"

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

EXCLUDED = {
    "tic_id", "label", "source", "split", "mean",
    "fraction_below_1pct", "fraction_above_99pct",
}


def normalize_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"^TIC_", "", regex=True)
    )


def main():
    base = pd.read_csv(BASE_PATH)
    bls = pd.read_csv(BLS_PATH)

    base["tic_id"] = normalize_id(base["tic_id"])
    bls["tic_id"] = normalize_id(bls["tic_id"])

    data = base.merge(
        bls[["tic_id"] + BLS_FEATURES],
        on="tic_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (data["_merge"] == "both").all():
        raise ValueError("Some TIC IDs did not match during merging.")

    data = data.drop(columns="_merge")

    features = [
        col for col in base.columns
        if col not in EXCLUDED
    ] + BLS_FEATURES

    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Train only on the existing training partition.
    train = data[data["split"] == "train"].copy()

    if len(train) == 0:
        raise ValueError("Training split is empty.")

    if train["label"].isna().any():
        raise ValueError("Training labels contain missing values.")

    X = train[features]
    y = train["label"]

    # Save imputation values learned from training rows only.
    medians = X.median()
    X = X.fillna(medians).fillna(0)

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=10,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )

    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "features": features,
        "medians": medians.to_dict(),
        "classes": list(model.classes_),
        "model_name": "LightGBM Reg1 (d=3, leaves=7, min_child=10, colsample=0.8)",
        "training_rows": len(train),
    }

    joblib.dump(artifact, MODEL_PATH)

    print("Model training complete.")
    print(f"Training rows: {len(train)}")
    print(f"Features: {len(features)}")
    print(f"Classes: {list(model.classes_)}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
