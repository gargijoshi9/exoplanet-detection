
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]

base = pd.read_csv(ROOT / "data" / "features.csv")
data = pd.read_csv(
    ROOT / "reports" / "bls_experiment" /
    "merged_experiment_features.csv"
)

for df in (base, data):
    df["tic_id"] = (
        df["tic_id"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

# Use the current 16 statistical features plus 9 BLS features.
excluded = {
    "tic_id", "label", "source", "split", "mean",
    "fraction_below_1pct", "fraction_above_99pct",
}
stat_features = [c for c in base.columns if c not in excluded]

bls_features = [
    "period_d", "duration_h", "depth_ppm", "snr", "snr_red",
    "sde", "n_transits", "odd_even_sigma", "secondary_sigma",
]
features = stat_features + bls_features

train = data[data["split"] == "train"].copy().reset_index(drop=True)
X = train[features].apply(pd.to_numeric, errors="coerce")
y = train["label"].to_numpy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importance_rows = []

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    X_train = X.iloc[train_idx].copy()
    X_valid = X.iloc[valid_idx].copy()
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    # Impute using only this fold's training data.
    medians = X_train.median()
    X_train = X_train.fillna(medians).fillna(0)
    X_valid = X_valid.fillna(medians).fillna(0)

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    # Measure importance on held-out rows, not training rows.
    result = permutation_importance(
        model,
        X_valid,
        y_valid,
        scoring="f1_macro",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    for i, name in enumerate(features):
        importance_rows.append({
            "fold": fold,
            "feature": name,
            "importance": result.importances_mean[i],
            "std": result.importances_std[i],
        })

importance = pd.DataFrame(importance_rows)

summary = (
    importance.groupby("feature")
    .agg(
        mean_importance=("importance", "mean"),
        fold_std=("importance", "std"),
        permutation_std=("std", "mean"),
    )
    .sort_values("mean_importance", ascending=False)
)

out = ROOT / "reports" / "bls_experiment"
summary.to_csv(out / "permutation_importance.csv")

print("\nTop 15 features by mean permutation importance:")
print(summary.head(15).round(4).to_string())
print(f"\nSaved to: {out / 'permutation_importance.csv'}")
