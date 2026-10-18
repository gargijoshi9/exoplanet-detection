
"""
Train a LightGBM classifier for exoplanet light-curve classification.

Classes:
- Transit
- Eclipsing Binary
- Other
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = PROJECT_ROOT / "data" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "lightgbm_model.joblib"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"


# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------

def load_dataset():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_FILE}\n"
            "Run: python src/feature_extraction.py"
        )

    df = pd.read_csv(FEATURES_FILE)

    if df.empty:
        raise ValueError("features.csv is empty.")

    required_columns = {"tic_id", "label", "source", "split"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Metadata must never be used as model input.
    excluded = ["tic_id", "label", "source", "split"]
    feature_columns = [
        col for col in df.columns if col not in excluded
        if col not in excluded and col != "mean"
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    # Ensure all model inputs are numeric.
    X = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Replace infinite values and handle missing feature values.
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric feature values found in: "
            f"{bad_columns}\n"
            "Fix these values before training."
        )

    y = df["label"].astype(str)
    split = df["split"].astype(str).str.strip().str.lower()

    train_mask = split == "train"
    test_mask = split == "test"

    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "The split column must contain both 'train' and 'test'."
        )

    # Verify that no star occurs in both sets.
    train_ids = set(df.loc[train_mask, "tic_id"])
    test_ids = set(df.loc[test_mask, "tic_id"])

    if train_ids & test_ids:
        raise ValueError("Data leakage: stars appear in both splits.")

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print("\nDataset loaded successfully!")
    print(f"Total stars: {len(df)}")
    print(f"Features per star: {len(feature_columns)}")
    print(f"Training stars: {len(X_train)}")
    print(f"Testing stars: {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test, feature_columns


# --------------------------------------------------
# 3. Train LightGBM
# --------------------------------------------------

def train_model(X_train, y_train):
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

    return model


# --------------------------------------------------
# 4. Evaluate the model
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\n" + "=" * 50)
    print("LIGHTGBM CLASSIFICATION RESULTS")
    print("=" * 50)

    print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1-score:    {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=model.classes_,
            zero_division=0,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    plt.title("LightGBM - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    return y_pred


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

def plot_feature_importance(model, feature_columns):
    """
    Plot LightGBM feature importance using total gain.
    """

    importance_values = model.booster_.feature_importance(
        importance_type="gain"
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance_values,
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    # Save the complete importance table.
    importance_csv = MODEL_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    # Plot the 15 most important features.
    top_features = importance_df.head(15).sort_values(
        by="importance",
        ascending=True,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )
    plt.xlabel("Total Gain")
    plt.ylabel("Feature")
    plt.title("LightGBM Feature Importance (Gain)")
    plt.tight_layout()

    plot_path = MODEL_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))

    print(f"\nFeature importance chart saved to: {plot_path}")
    print(f"Feature importance table saved to: {importance_csv}")

    return importance_df

# --------------------------------------------------
# 5. Save model and feature information
# --------------------------------------------------

def save_model(model, feature_columns):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save both the model and the exact input column order.
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "classes": list(model.classes_),
        },
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")


# --------------------------------------------------
# 6. Run complete pipeline
# --------------------------------------------------

def main():
    X_train, X_test, y_train, y_test, feature_columns = load_dataset()

    print("\nTraining LightGBM...")
    model = train_model(X_train, y_train)
    print("Training completed!")

    evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, feature_columns)

    save_model(model, feature_columns)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
