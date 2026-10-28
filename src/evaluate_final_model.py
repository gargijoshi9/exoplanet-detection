
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = ROOT / "data" / "features.csv"
BLS_PATH = ROOT / "data" / "features" / "bls_features.csv"
OUT_DIR = ROOT / "reports" / "final_evaluation"

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

    if base["tic_id"].duplicated().any():
        raise ValueError("Duplicate IDs in baseline features.")
    if bls["tic_id"].duplicated().any():
        raise ValueError("Duplicate IDs in BLS features.")

    data = base.merge(
        bls[["tic_id"] + BLS_FEATURES],
        on="tic_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (data["_merge"] == "both").all():
        raise ValueError("Some TIC IDs did not match across datasets.")

    data = data.drop(columns="_merge")

    statistical_features = [
        col for col in base.columns if col not in EXCLUDED
    ]
    features = statistical_features + BLS_FEATURES

    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "test"].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Train or test split is empty: train={len(train)}, test={len(test)}"
        )

    if train["label"].isna().any() or test["label"].isna().any():
        raise ValueError("Missing labels detected.")

    # Fit imputation on training data only.
    medians = train[features].median()
    X_train = train[features].fillna(medians).fillna(0)
    X_test = test[features].fillna(medians).fillna(0)

    y_train = train["label"]
    y_test = test["label"]

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    labels = ["Eclipsing Binary", "Other", "Transit"]

    print("=== FINAL HELD-OUT TEST RESULTS ===")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Feature count: {len(features)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Macro F1: "
        f"{f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        "Weighted F1: "
        f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=labels, zero_division=0
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "tic_id": test["tic_id"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    }).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    pd.Series(
        model.feature_importances_, index=features, name="importance"
    ).sort_values(ascending=False).to_csv(
        OUT_DIR / "feature_importance.csv"
    )

    print(f"\nSaved reports to: {OUT_DIR}")


if __name__ == "__main__":
    main()
