"""
Helpers for the Streamlit dashboard.

All paths are resolved from the repository root so the app does not depend
on the process working directory. Missing artifacts stay missing: this module
never fills them with demo values.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detect import RESULT_COLUMNS, detect, load_input, phase_offset, save_outputs
from feature_extraction import extract_features

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FULL_DIR = DATA_DIR / "full"
FEATURES_DIR = DATA_DIR / "features"
LABELS_PATH = DATA_DIR / "labels.csv"
FAILED_PATH = DATA_DIR / "failed.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
BLS_FEATURES_PATH = FEATURES_DIR / "bls_features.csv"

MODELS_DIR = ROOT / "models"
PIPELINE_MODEL_PATH = MODELS_DIR / "lightgbm_bls_pipeline.joblib"
LEGACY_MODEL_PATH = MODELS_DIR / "lightgbm_model.joblib"

RESULTS_DIR = ROOT / "results"
DETECTION_DIR = RESULTS_DIR / "detection"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"

EVAL_DIR = ROOT / "reports" / "final_evaluation"
TEST_PREDICTIONS_PATH = EVAL_DIR / "test_predictions.csv"
EVAL_IMPORTANCE_PATH = EVAL_DIR / "feature_importance.csv"

TIC_RE = re.compile(r"(?:TIC[_\s-]?)(\d+)", re.IGNORECASE)

METADATA_COLUMNS = {"tic_id", "label", "source", "split"}
PLACEHOLDER_REPORT_MARKERS = ("placeholder", "run src/evaluate.py")

CLASS_ORDER = ["Transit", "Eclipsing Binary", "Other"]
DETECTION_STATUS_ORDER = ["candidate", "flagged", "rejected", "no_detection", "error"]


def normalize_tic(value) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    text = re.sub(r"\.0$", "", text)
    match = TIC_RE.search(text)
    if match:
        return match.group(1)
    digits = re.sub(r"[^\d]", "", text)
    return digits or None


def parse_tic_from_path(path: Path) -> str | None:
    return normalize_tic(path.stem)


def missing_artifact(path: Path, description: str) -> dict:
    return {
        "status": "missing",
        "path": path,
        "description": description,
        "data": None,
        "message": f"{description} was not found at {path.relative_to(ROOT)}.",
    }


def empty_artifact(path: Path, description: str) -> dict:
    return {
        "status": "empty",
        "path": path,
        "description": description,
        "data": None,
        "message": f"{description} exists at {path.relative_to(ROOT)} but contains no rows.",
    }


def loaded_artifact(path: Path, description: str, data) -> dict:
    return {
        "status": "loaded",
        "path": path,
        "description": description,
        "data": data,
        "message": None,
    }


def read_csv_artifact(path: Path, description: str) -> dict:
    if not path.exists():
        return missing_artifact(path, description)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {
            "status": "error",
            "path": path,
            "description": description,
            "data": None,
            "message": f"Could not read {path.relative_to(ROOT)}: {exc}",
        }
    if df.empty:
        return empty_artifact(path, description)
    return loaded_artifact(path, description, df)


def format_count(value) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def discover_lightcurve_files() -> dict[str, list[dict]]:
    """Map TIC ID -> list of available light-curve files."""
    found: dict[str, list[dict]] = {}
    patterns = [
        (PROCESSED_DIR, "npz", "processed_npz"),
        (PROCESSED_DIR, "csv", "processed_csv"),
        (FULL_DIR, "csv", "full_csv"),
    ]
    for directory, suffix, kind in patterns:
        if not directory.exists():
            continue
        glob = f"*.{suffix}" if suffix != "npz" else "TIC_*.npz"
        extra = []
        if suffix == "npz":
            extra = list(directory.glob("TIC_*.npz"))
        else:
            extra = list(directory.glob("*.csv")) + list(directory.glob("processed_TIC_*.csv"))
        files = extra if suffix == "npz" else list(directory.glob("*.csv"))
        for path in files:
            tic = parse_tic_from_path(path)
            if tic is None:
                continue
            found.setdefault(tic, []).append(
                {
                    "path": path,
                    "kind": kind,
                    "suffix": path.suffix.lower(),
                    "relative": path.relative_to(ROOT).as_posix(),
                }
            )
    # De-duplicate identical paths
    for tic, items in found.items():
        unique = []
        seen = set()
        for item in items:
            key = str(item["path"].resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        found[tic] = unique
    return found


def load_labels() -> dict:
    artifact = read_csv_artifact(LABELS_PATH, "Star list (data/labels.csv)")
    if artifact["status"] != "loaded":
        return artifact
    df = artifact["data"].copy()
    if "tic_id" not in df.columns:
        return {
            "status": "error",
            "path": LABELS_PATH,
            "description": artifact["description"],
            "data": None,
            "message": "data/labels.csv is missing the tic_id column.",
        }
    df["tic_id"] = df["tic_id"].map(normalize_tic)
    artifact["data"] = df
    return artifact


def load_failed() -> dict:
    return read_csv_artifact(FAILED_PATH, "Failed preprocessing list")


def load_features() -> dict:
    return read_csv_artifact(FEATURES_PATH, "Statistical feature table")


def load_bls_features() -> dict:
    return read_csv_artifact(BLS_FEATURES_PATH, "Saved BLS feature table")


def load_test_predictions() -> dict:
    return read_csv_artifact(TEST_PREDICTIONS_PATH, "Held-out test predictions")


def discover_detection_files() -> dict[str, dict]:
    """Map TIC ID -> saved detection CSV and related plot files."""
    out: dict[str, dict] = {}
    if not DETECTION_DIR.exists():
        return out
    for csv_path in DETECTION_DIR.glob("*_detections.csv"):
        tic = parse_tic_from_path(csv_path)
        if tic is None:
            continue
        stem = csv_path.name.replace("_detections.csv", "")
        out[tic] = {
            "csv": csv_path,
            "stem": stem,
            "overview": DETECTION_DIR / f"{stem}_overview.png",
            "periodograms": sorted(DETECTION_DIR.glob(f"{stem}_periodogram_*.png")),
            "folds": sorted(DETECTION_DIR.glob(f"{stem}_fold_*.png")),
        }
    return out


def load_detection_table(csv_path: Path) -> dict:
    artifact = read_csv_artifact(csv_path, "Detection results")
    if artifact["status"] != "loaded":
        return artifact
    artifact["data"] = artifact["data"].copy()
    missing = [column for column in RESULT_COLUMNS if column not in artifact["data"].columns]
    if missing:
        artifact["missing_columns"] = missing
    return artifact


def load_model_artifact(path: Path | None = None) -> dict:
    model_path = path or PIPELINE_MODEL_PATH
    if not model_path.exists():
        return missing_artifact(model_path, "Saved LightGBM pipeline")
    try:
        import joblib

        artifact = joblib.load(model_path)
    except Exception as exc:
        return {
            "status": "error",
            "path": model_path,
            "description": "Saved LightGBM pipeline",
            "data": None,
            "message": f"The model file exists but could not be loaded: {exc}",
        }

    if not isinstance(artifact, dict) or "model" not in artifact:
        return {
            "status": "error",
            "path": model_path,
            "description": "Saved LightGBM pipeline",
            "data": None,
            "message": "The joblib file is not the expected dictionary artifact "
            "({'model', 'features', 'medians', 'classes'}).",
        }

    features = artifact.get("features") or artifact.get("feature_columns")
    if not features:
        return {
            "status": "error",
            "path": model_path,
            "description": "Saved LightGBM pipeline",
            "data": None,
            "message": "The saved artifact does not include feature column names.",
        }

    payload = {
        "model": artifact["model"],
        "features": list(features),
        "medians": dict(artifact.get("medians") or {}),
        "classes": list(artifact.get("classes") or getattr(artifact["model"], "classes_", [])),
        "model_name": artifact.get("model_name", model_path.name),
        "training_rows": artifact.get("training_rows"),
        "path": model_path,
    }
    return loaded_artifact(model_path, "Saved LightGBM pipeline", payload)


def load_lightcurve(path: Path) -> dict:
    """Load one processed light curve from CSV or NPZ."""
    path = Path(path)
    if not path.exists():
        return missing_artifact(path, "Light curve")
    try:
        if path.suffix.lower() == ".npz":
            time_arr, flux_arr, err_arr, name = load_input(npz=str(path))
            with np.load(path) as data:
                keys = set(data.files)
                flux_norm = data["flux_norm"].astype(float) if "flux_norm" in keys else None
                sector = data["sector"] if "sector" in keys else None
            df = pd.DataFrame({"time": time_arr, "flux": flux_arr})
            if err_arr is not None:
                df["flux_err"] = err_arr
            if flux_norm is not None and len(flux_norm) == len(df):
                df["flux_norm"] = flux_norm
            stages = {
                "normalized_detrended": "flux",
                "normalized_not_detrended": "flux_norm" if "flux_norm" in df.columns else None,
                "raw": None,
                "cleaned": None,
            }
            notes = [
                "NPZ contents: time, flux (detrended and normalized), flux_err"
                + (", flux_norm (normalized, not detrended)" if "flux_norm" in df.columns else "")
                + (", sector" if sector is not None else "")
                + ".",
                "Raw and outlier-cleaned (pre-flatten) curves are not stored.",
            ]
        else:
            original = pd.read_csv(path)
            col_map = {str(c).strip().lower(): c for c in original.columns}
            if "time" not in col_map or "flux" not in col_map:
                raise ValueError(
                    f"CSV must contain time and flux columns. Found: {list(original.columns)}"
                )
            df = pd.DataFrame({
                "time": original[col_map["time"]].astype(float),
                "flux": original[col_map["flux"]].astype(float),
            })
            err_col = col_map.get("flux_err") or col_map.get("flux_error") or col_map.get("err")
            if err_col is not None:
                df["flux_err"] = original[err_col].astype(float)
            stages = {
                "normalized_detrended": "flux",
                "normalized_not_detrended": None,
                "raw": None,
                "cleaned": None,
            }
            notes = [f"CSV columns present: {list(original.columns)}."]
            if "flux_norm" in col_map:
                df["flux_norm"] = original[col_map["flux_norm"]].astype(float)
                stages["normalized_not_detrended"] = "flux_norm"
            else:
                notes.append("Non-detrended flux_norm is not stored in this file.")
            notes.append("Raw and outlier-cleaned (pre-flatten) curves are not stored.")
            known = {"time", "flux", "flux_err", "flux_error", "err", "flux_norm"}
            extra_cols = [c for c in original.columns if str(c).strip().lower() not in known]
            if extra_cols:
                notes.append(f"Additional columns were not plotted: {extra_cols}.")
            name = path.stem
        tic = parse_tic_from_path(path)
        stats = lightcurve_stats(df, tic)
        return {
            "status": "loaded",
            "path": path,
            "description": "Light curve",
            "data": df,
            "name": name,
            "tic_id": tic,
            "stages": stages,
            "notes": notes,
            "stats": stats,
            "message": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "path": path,
            "description": "Light curve",
            "data": None,
            "message": f"Failed to load {path}: {exc}",
        }


def lightcurve_stats(df: pd.DataFrame, tic_id: str | None) -> dict:
    n = int(len(df))
    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    t = pd.to_numeric(df["time"], errors="coerce") if "time" in df.columns else pd.Series(dtype=float)
    finite_t = t[np.isfinite(t)]
    coverage = float(finite_t.max() - finite_t.min()) if len(finite_t) else np.nan
    return {
        "tic_id": tic_id,
        "n_points": n,
        "missing": missing,
        "time_start": float(finite_t.min()) if len(finite_t) else np.nan,
        "time_end": float(finite_t.max()) if len(finite_t) else np.nan,
        "time_coverage_d": coverage,
        "columns": list(df.columns),
    }


def downsample_frame(df: pd.DataFrame, max_points: int = 8000) -> tuple[pd.DataFrame, bool]:
    if len(df) <= max_points:
        return df, False
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy(), True


def build_catalog() -> pd.DataFrame:
    """One row per known TIC from labels and/or available files."""
    rows: dict[str, dict] = {}
    lightcurves = discover_lightcurve_files()
    detections = discover_detection_files()
    labels = load_labels()
    bls = load_bls_features()
    preds = load_test_predictions()

    def ensure(tic: str) -> dict:
        if tic not in rows:
            rows[tic] = {
                "tic_id": tic,
                "label": pd.NA,
                "source": pd.NA,
                "split": pd.NA,
                "light_curve_available": False,
                "light_curve_files": "",
                "primary_light_curve": pd.NA,
                "detection_csv": pd.NA,
                "detection_status": pd.NA,
                "n_detected_signals": pd.NA,
                "n_candidates": pd.NA,
                "predicted_class": pd.NA,
                "prediction_origin": pd.NA,
            }
        return rows[tic]

    if labels["status"] == "loaded":
        for rec in labels["data"].to_dict(orient="records"):
            tic = rec.get("tic_id")
            if not tic:
                continue
            row = ensure(tic)
            row["label"] = rec.get("label", pd.NA)
            row["source"] = rec.get("source", pd.NA)
            row["split"] = rec.get("split", pd.NA)

    for tic, files in lightcurves.items():
        row = ensure(tic)
        row["light_curve_available"] = True
        row["light_curve_files"] = "; ".join(item["relative"] for item in files)
        preferred = next((item for item in files if item["kind"] == "processed_npz"), files[0])
        row["primary_light_curve"] = preferred["relative"]

    for tic, info in detections.items():
        row = ensure(tic)
        row["detection_csv"] = str(info["csv"].relative_to(ROOT))
        loaded = load_detection_table(info["csv"])
        if loaded["status"] == "loaded":
            table = loaded["data"]
            row["n_detected_signals"] = int(len(table))
            if "status" in table.columns:
                row["n_candidates"] = int((table["status"].astype(str) == "candidate").sum())
                if len(table):
                    row["detection_status"] = str(table.iloc[0]["status"])
            elif "accepted" in table.columns:
                row["n_candidates"] = int(table["accepted"].astype(bool).sum())

    if bls["status"] == "loaded" and "tic_id" in bls["data"].columns:
        bls_df = bls["data"].copy()
        bls_df["tic_id"] = bls_df["tic_id"].map(normalize_tic)
        for rec in bls_df.to_dict(orient="records"):
            tic = rec.get("tic_id")
            if not tic:
                continue
            row = ensure(tic)
            if pd.isna(row["detection_status"]) and rec.get("status") is not None:
                row["detection_status"] = rec.get("status")
            if pd.isna(row["n_detected_signals"]) and rec.get("detection_found") is not None:
                try:
                    row["n_detected_signals"] = int(rec.get("detection_found"))
                except (TypeError, ValueError):
                    pass

    if preds["status"] == "loaded":
        pred_df = preds["data"].copy()
        id_col = "tic_id" if "tic_id" in pred_df.columns else None
        pred_col = "predicted" if "predicted" in pred_df.columns else (
            "predicted_class" if "predicted_class" in pred_df.columns else None
        )
        if id_col and pred_col:
            pred_df[id_col] = pred_df[id_col].map(normalize_tic)
            for rec in pred_df.to_dict(orient="records"):
                tic = rec.get(id_col)
                if not tic:
                    continue
                row = ensure(tic)
                row["predicted_class"] = rec.get(pred_col)
                row["prediction_origin"] = "held-out test predictions"

    catalog = pd.DataFrame(rows.values())
    if catalog.empty:
        return catalog
    catalog = catalog.sort_values("tic_id").reset_index(drop=True)
    return catalog


def overview_metrics(catalog: pd.DataFrame) -> dict:
    labels = load_labels()
    detections = discover_detection_files()
    bls = load_bls_features()
    preds = load_test_predictions()

    n_available = int(catalog["light_curve_available"].sum()) if not catalog.empty else 0
    n_in_labels = int(len(labels["data"])) if labels["status"] == "loaded" else None

    analyzed = None
    if detections or bls["status"] in {"loaded", "empty"}:
        analyzed = 0
        if not catalog.empty:
            has_det = catalog["detection_csv"].notna() if "detection_csv" in catalog.columns else False
            has_status = catalog["detection_status"].notna() if "detection_status" in catalog.columns else False
            analyzed = int((has_det | has_status).sum())
    elif bls["status"] == "missing" and not detections:
        analyzed = None

    n_candidates = None
    candidate_source = None
    candidate_rows = []
    for tic, info in detections.items():
        loaded = load_detection_table(info["csv"])
        if loaded["status"] == "loaded" and "status" in loaded["data"].columns:
            table = loaded["data"].copy()
            table["tic_id"] = tic
            candidate_rows.append(table)
    if candidate_rows:
        all_det = pd.concat(candidate_rows, ignore_index=True)
        n_candidates = int((all_det["status"].astype(str) == "candidate").sum())
        candidate_source = "results/detection/*_detections.csv"
    elif bls["status"] == "loaded" and "status" in bls["data"].columns:
        n_candidates = int((bls["data"]["status"].astype(str) == "candidate").sum())
        candidate_source = str(BLS_FEATURES_PATH.relative_to(ROOT))

    class_counts = {name: None for name in CLASS_ORDER}
    class_source = None
    if preds["status"] == "loaded":
        pred_col = "predicted" if "predicted" in preds["data"].columns else (
            "predicted_class" if "predicted_class" in preds["data"].columns else None
        )
        if pred_col:
            counts = preds["data"][pred_col].astype(str).value_counts()
            class_counts = {name: int(counts.get(name, 0)) for name in CLASS_ORDER}
            class_source = f"held-out test file {TEST_PREDICTIONS_PATH.relative_to(ROOT)}"
    elif not catalog.empty and catalog["predicted_class"].notna().any():
        counts = catalog["predicted_class"].dropna().astype(str).value_counts()
        class_counts = {name: int(counts.get(name, 0)) for name in CLASS_ORDER}
        class_source = "catalog predictions"

    label_counts = {name: None for name in CLASS_ORDER}
    label_source = None
    if labels["status"] == "loaded" and "label" in labels["data"].columns:
        counts = labels["data"]["label"].dropna().astype(str).value_counts()
        label_counts = {name: int(counts.get(name, 0)) for name in CLASS_ORDER}
        label_source = str(LABELS_PATH.relative_to(ROOT))

    return {
        "n_available_targets": n_available,
        "n_targets_in_labels": n_in_labels,
        "labels_status": labels["status"],
        "labels_message": labels.get("message"),
        "n_analyzed": analyzed,
        "n_candidates": n_candidates,
        "candidate_source": candidate_source,
        "class_counts": class_counts,
        "class_source": class_source,
        "label_counts": label_counts,
        "label_source": label_source,
        "n_catalog_rows": int(len(catalog)),
    }


def evaluation_bundle() -> dict:
    """Load held-out evaluation artifacts only. Never treat training metrics as test scores."""
    preds = load_test_predictions()
    report = CLASSIFICATION_REPORT_PATH
    report_text = None
    report_usable = False
    if report.exists():
        report_text = report.read_text(encoding="utf-8", errors="replace")
        lowered = report_text.lower()
        report_usable = not any(marker in lowered for marker in PLACEHOLDER_REPORT_MARKERS)

    bundle = {
        "predictions": preds,
        "report_path": report,
        "report_text": report_text,
        "report_usable": report_usable,
        "metrics": None,
        "confusion": None,
        "classes": None,
        "message": None,
    }

    if preds["status"] != "loaded":
        if report_usable:
            bundle["message"] = (
                "A classification report file exists, but held-out prediction rows "
                f"were not found at {TEST_PREDICTIONS_PATH.relative_to(ROOT)}, so "
                "metrics cannot be verified against an evaluation split."
            )
        else:
            bundle["message"] = (
                "Held-out evaluation is unavailable. "
                f"{TEST_PREDICTIONS_PATH.relative_to(ROOT)} was not found, and "
                "results/classification_report.txt is a placeholder, not measured scores."
            )
        return bundle

    df = preds["data"]
    actual_col = next((c for c in ("actual", "y_true", "label", "true_label") if c in df.columns), None)
    pred_col = next((c for c in ("predicted", "predicted_class", "y_pred") if c in df.columns), None)
    if actual_col is None or pred_col is None:
        bundle["message"] = (
            f"{TEST_PREDICTIONS_PATH.relative_to(ROOT)} does not contain both actual "
            "and predicted columns, so test metrics cannot be computed."
        )
        return bundle

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = df[actual_col].astype(str)
    y_pred = df[pred_col].astype(str)
    labels = [c for c in CLASS_ORDER if c in set(y_true) | set(y_pred)]
    if not labels:
        labels = sorted(set(y_true) | set(y_pred))

    report_dict = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    bundle["metrics"] = {
        "split": "held-out test (reports/final_evaluation/test_predictions.csv)",
        "n_test": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": report_dict,
    }
    bundle["confusion"] = confusion_matrix(y_true, y_pred, labels=labels)
    bundle["classes"] = labels
    bundle["message"] = None
    return bundle


def feature_importance_frame(model_payload: dict) -> pd.DataFrame | None:
    model = model_payload["model"]
    features = model_payload["features"]
    try:
        values = model.booster_.feature_importance(importance_type="gain")
    except Exception:
        values = getattr(model, "feature_importances_", None)
        if values is None:
            return None
    if len(values) != len(features):
        return None
    return (
        pd.DataFrame({"feature": features, "importance_gain": values})
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )


def prepare_model_matrix(feature_values: dict, model_payload: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build a single-row matrix in the saved feature order, with training medians then 0."""
    names = model_payload["features"]
    medians = model_payload.get("medians") or {}
    row = {name: feature_values.get(name, np.nan) for name in names}
    X = pd.DataFrame([[row[name] for name in names]], columns=names)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    imputed = [name for name in names if pd.isna(X.at[0, name])]
    median_series = pd.Series(medians)
    X = X.fillna(median_series).fillna(0)
    leaked = [name for name in names if name in METADATA_COLUMNS]
    if leaked:
        raise ValueError(f"Refusing to score: metadata columns present in model features: {leaked}")
    return X, imputed


def bls_values_from_table(table: pd.DataFrame) -> tuple[dict, dict]:
    names = [
        "period_d", "duration_h", "depth_ppm", "snr", "snr_red", "sde",
        "n_transits", "odd_even_sigma", "secondary_sigma",
    ]
    values = {name: np.nan for name in names}
    diagnostics = {"detection_found": False, "status": "no_detection", "flags": []}
    if table is None or table.empty:
        return values, diagnostics
    signal = table.iloc[0]
    for name in names:
        if name in signal.index:
            number = pd.to_numeric(signal[name], errors="coerce")
            values[name] = float(number) if pd.notna(number) else np.nan
    flags = signal.get("flags", "")
    if pd.isna(flags):
        flags = ""
    diagnostics = {
        "detection_found": True,
        "status": str(signal.get("status", "unknown")),
        "flags": [part.strip() for part in str(flags).split(",") if part.strip()],
        "accepted": signal.get("accepted"),
    }
    return values, diagnostics


def run_detection(time, flux, flux_err=None, **kwargs) -> tuple[pd.DataFrame, list]:
    """Thin wrapper around the existing detector. Call only from an explicit UI action."""
    return detect(time, flux, flux_err, **kwargs)


def persist_detection_outputs(table: pd.DataFrame, details: list, time, flux, name: str) -> Path:
    """Write the detector's CSV and PNG products using src.detect.save_outputs."""
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    save_outputs(table, details, np.asarray(time, dtype=float), np.asarray(flux, dtype=float), name, DETECTION_DIR)
    return DETECTION_DIR / f"{name}_detections.csv"


def classify_lightcurve(lc: pd.DataFrame, model_payload: dict, detection_table: pd.DataFrame | None = None) -> dict:
    statistical = extract_features(lc).iloc[0].to_dict()
    if detection_table is None:
        table, _details = detect(
            lc["time"].to_numpy(),
            lc["flux"].to_numpy(),
            flux_err=lc["flux_err"].to_numpy() if "flux_err" in lc.columns else None,
            verbose=False,
        )
    else:
        table = detection_table
    bls_values, diagnostics = bls_values_from_table(table)
    combined = {**statistical, **bls_values}
    X, imputed = prepare_model_matrix(combined, model_payload)
    model = model_payload["model"]
    predicted = str(model.predict(X)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probabilities = {
            str(cls): float(score)
            for cls, score in zip(model.classes_, proba)
        }
    return {
        "predicted_class": predicted,
        "probabilities": probabilities,
        "imputed_features": imputed,
        "bls": bls_values,
        "bls_diagnostics": diagnostics,
        "statistical_features": statistical,
        "feature_row": X.iloc[0].to_dict(),
        "probability_note": (
            "These are the classifier's predict_proba scores. They are not a "
            "calibrated probability that the predicted class is correct."
        ),
    }


def phase_fold_frame(lc: pd.DataFrame, period, t0, duration_h, max_points: int = 8000) -> pd.DataFrame:
    t = pd.to_numeric(lc["time"], errors="coerce").to_numpy()
    f = pd.to_numeric(lc["flux"], errors="coerce").to_numpy()
    period = float(period)
    t0 = float(t0)
    duration = float(duration_h) / 24.0
    half = max(3 * duration, 4 / 24)
    ph = phase_offset(t, t0, period)
    sel = np.isfinite(ph) & np.isfinite(f) & (np.abs(ph) < half)
    folded = pd.DataFrame({
        "hours_from_mid_transit": ph[sel] * 24.0,
        "flux": f[sel],
    })
    folded, _ = downsample_frame(folded, max_points=max_points)
    return folded.sort_values("hours_from_mid_transit")


def bls_box_trace(duration_h, depth_ppm, half_hours) -> tuple[list[float], list[float]]:
    dur_h = float(duration_h)
    depth = float(depth_ppm) / 1e6
    xs = [-half_hours, -dur_h / 2, -dur_h / 2, dur_h / 2, dur_h / 2, half_hours]
    ys = [1.0, 1.0, 1.0 - depth, 1.0 - depth, 1.0, 1.0]
    return xs, ys


def fitting_status() -> dict:
    """Transit-parameter fitting is not implemented in this repository."""
    fit_modules = list(SRC_DIR.glob("*fit*.py")) + list(SRC_DIR.glob("*batman*"))
    batman_mentions = []
    for path in SRC_DIR.glob("*.py"):
        if path.name == "dashboard_utils.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\bbatman\b|transit parameter", text, re.IGNORECASE):
            batman_mentions.append(str(path.relative_to(ROOT)))
    result_dirs = [
        RESULTS_DIR / "fitting",
        RESULTS_DIR / "transit_fit",
        ROOT / "reports" / "fitting",
    ]
    existing_results = [p for p in result_dirs if p.exists() and any(p.iterdir())]
    implemented = bool(fit_modules) or bool(existing_results)
    if implemented:
        reason = "Fitting-related files were found; inspect them before plotting."
    else:
        reason = (
            "Transit parameter fitting has not been implemented. There is no fitting "
            "module under src/, batman is only mentioned as a planned dependency, and "
            "no fitted-parameter tables or model curves were found in results/."
        )
    return {
        "available": implemented,
        "reason": reason,
        "modules_named_fit": [str(p.relative_to(ROOT)) for p in fit_modules],
        "batman_mentions": batman_mentions,
        "result_dirs_found": [str(p.relative_to(ROOT)) for p in existing_results],
    }


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def filter_catalog(catalog: pd.DataFrame, predicted_class: str | None, detection_status: str | None, query: str | None) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    out = catalog.copy()
    if predicted_class and predicted_class != "All":
        out = out[out["predicted_class"].astype(str) == predicted_class]
    if detection_status and detection_status != "All":
        out = out[out["detection_status"].astype(str) == detection_status]
    if query:
        needle = query.strip().lower()
        if needle:
            as_text = out.astype(str).apply(lambda col: col.str.lower())
            mask = as_text.apply(lambda col: col.str.contains(needle, na=False)).any(axis=1)
            out = out[mask]
    return out.reset_index(drop=True)


def target_export_payload(tic_id: str, catalog: pd.DataFrame, lc_bundle: dict | None, detection_table: pd.DataFrame | None) -> pd.DataFrame:
    """Combine the selected target's actually available rows for download."""
    blocks = []
    if not catalog.empty:
        match = catalog[catalog["tic_id"].astype(str) == str(tic_id)].copy()
        if not match.empty:
            match.insert(0, "section", "catalog")
            blocks.append(match)
    if detection_table is not None and not detection_table.empty:
        det = detection_table.copy()
        det.insert(0, "section", "detection")
        det.insert(1, "tic_id", tic_id)
        blocks.append(det)
    if lc_bundle and lc_bundle.get("stats"):
        stats = pd.DataFrame([lc_bundle["stats"]])
        stats.insert(0, "section", "light_curve_stats")
        blocks.append(stats)
    if not blocks:
        return pd.DataFrame({"tic_id": [tic_id], "note": ["No saved results were available for this target."]})
    return pd.concat(blocks, ignore_index=True, sort=False)


FLAG_METADATA = {
    "odd_even_mismatch": {
        "label": "Odd-Even Depth Mismatch (> 3σ)",
        "description": "Odd and even transit depths differ by > 3σ. Strongly suggests alternating primary and secondary eclipses of an eclipsing binary star system.",
        "category": "Astrophysical False Positive",
        "severity": "high",
    },
    "secondary_eclipse": {
        "label": "Secondary Eclipse (> 3σ)",
        "description": "A secondary dip is detected at phase 0.5 with > 3σ significance, characteristic of a stellar occultation in a binary system.",
        "category": "Astrophysical False Positive",
        "severity": "high",
    },
    "too_long_for_period": {
        "label": "Unphysical Duration (T_dur / P > 10%)",
        "description": "Transit duration exceeds 10% of the orbital period. Keplerian dynamics constrain planetary transits across solar-type stars to T_dur / P < 0.10.",
        "category": "Orbital Rule",
        "severity": "medium",
    },
    "too_deep": {
        "label": "Excessive Depth (> 50,000 ppm / 5%)",
        "description": "Transit depth exceeds 5% of host flux, which exceeds theoretical planetary radius limits and points to a stellar companion.",
        "category": "Astrophysical False Positive",
        "severity": "high",
    },
    "few_transits": {
        "label": "Few Transits (< 2 Events)",
        "description": "Fewer than 2 transit events covered during the observation window, preventing period verification.",
        "category": "Observational Constraint",
        "severity": "medium",
    },
    "negative_depth": {
        "label": "Negative / Inverted Depth",
        "description": "The BLS model fit an emission or brightening spike rather than an absorption dip.",
        "category": "Detector Artifact",
        "severity": "high",
    },
    "insufficient_data": {
        "label": "Insufficient Points (< 5 Points)",
        "description": "Fewer than 5 valid finite observations in the input series.",
        "category": "Data Quality",
        "severity": "high",
    },
    "compute_stats_failed": {
        "label": "Stats Computation Failed",
        "description": "Astropy BoxLeastSquares was unable to compute secondary and odd/even statistics.",
        "category": "Algorithmic Error",
        "severity": "high",
    },
}


def explain_flag(flag: str) -> dict:
    flag_key = str(flag).strip().lower()
    if flag_key in FLAG_METADATA:
        return FLAG_METADATA[flag_key]
    return {
        "label": flag,
        "description": f"Triggered diagnostic flag: {flag}.",
        "category": "Diagnostic Flag",
        "severity": "medium",
    }


def calculate_transit_windows(
    time_min: float, time_max: float, period_d: float, t0_btjd: float, duration_h: float
) -> list[dict]:
    """Calculate expected transit epochs and windows within [time_min, time_max]."""
    if not (np.isfinite(period_d) and period_d > 0 and np.isfinite(t0_btjd) and np.isfinite(duration_h) and duration_h > 0):
        return []
    dur_d = duration_h / 24.0
    half_dur = dur_d / 2.0
    n_start = int(np.floor((time_min - t0_btjd - half_dur) / period_d))
    n_end = int(np.ceil((time_max - t0_btjd + half_dur) / period_d))
    windows = []
    for n in range(n_start, n_end + 1):
        mid = t0_btjd + n * period_d
        start = mid - half_dur
        end = mid + half_dur
        if end >= time_min and start <= time_max:
            windows.append({
                "transit_num": n,
                "t_mid": float(mid),
                "t_start": float(start),
                "t_end": float(end),
            })
    return windows


def load_uploaded_lightcurve(uploaded_file) -> dict:
    """Safely parse and validate an uploaded .npz or .csv light curve file in memory."""
    max_bytes = 25 * 1024 * 1024  # 25 MB limit
    file_size = getattr(uploaded_file, "size", None)
    if file_size is None and hasattr(uploaded_file, "getvalue"):
        file_size = len(uploaded_file.getvalue())
    elif file_size is None:
        file_size = 0

    if file_size > max_bytes:
        return {
            "status": "error",
            "path": None,
            "description": "Uploaded light curve",
            "data": None,
            "message": f"File size exceeds 25 MB limit ({file_size / (1024*1024):.1f} MB).",
        }
    filename = getattr(uploaded_file, "name", "uploaded_file.csv")
    suffix = Path(filename).suffix.lower()

    try:
        if suffix == ".npz":
            data_bytes = io.BytesIO(uploaded_file.getvalue())
            with np.load(data_bytes) as npz:
                files = set(npz.files)
                if "time" not in files or "flux" not in files:
                    return {
                        "status": "error",
                        "path": None,
                        "description": "Uploaded light curve",
                        "data": None,
                        "message": f"NPZ file missing required arrays 'time' and 'flux'. Found: {list(files)}",
                    }
                time_arr = npz["time"].astype(float)
                flux_arr = npz["flux"].astype(float)
                err_arr = npz["flux_err"].astype(float) if "flux_err" in files else None
                flux_norm = npz["flux_norm"].astype(float) if "flux_norm" in files else None
                sector = npz["sector"] if "sector" in files else None
            df = pd.DataFrame({"time": time_arr, "flux": flux_arr})
            if err_arr is not None and len(err_arr) == len(df):
                df["flux_err"] = err_arr
            if flux_norm is not None and len(flux_norm) == len(df):
                df["flux_norm"] = flux_norm
            stages = {
                "normalized_detrended": "flux",
                "normalized_not_detrended": "flux_norm" if "flux_norm" in df.columns else None,
                "raw": None,
                "cleaned": None,
            }
            notes = [
                f"Uploaded NPZ file '{filename}' with {len(df):,} data points.",
                "Contents: time, flux" + (", flux_err" if "flux_err" in df.columns else "") + (", flux_norm" if "flux_norm" in df.columns else ""),
            ]
        elif suffix == ".csv":
            df_raw = pd.read_csv(uploaded_file)
            col_map = {str(c).strip().lower(): c for c in df_raw.columns}
            if "time" not in col_map or "flux" not in col_map:
                return {
                    "status": "error",
                    "path": None,
                    "description": "Uploaded light curve",
                    "data": None,
                    "message": f"CSV must contain 'time' and 'flux' columns (case-insensitive). Found: {list(df_raw.columns)}",
                }
            df = pd.DataFrame({
                "time": pd.to_numeric(df_raw[col_map["time"]], errors="coerce").astype(float),
                "flux": pd.to_numeric(df_raw[col_map["flux"]], errors="coerce").astype(float),
            })
            err_col = col_map.get("flux_err") or col_map.get("flux_error") or col_map.get("err")
            if err_col is not None:
                df["flux_err"] = pd.to_numeric(df_raw[err_col], errors="coerce").astype(float)
            if "flux_norm" in col_map:
                df["flux_norm"] = pd.to_numeric(df_raw[col_map["flux_norm"]], errors="coerce").astype(float)
            stages = {
                "normalized_detrended": "flux",
                "normalized_not_detrended": "flux_norm" if "flux_norm" in df.columns else None,
                "raw": None,
                "cleaned": None,
            }
            notes = [f"Uploaded CSV file '{filename}' with {len(df):,} data points."]
        else:
            return {
                "status": "error",
                "path": None,
                "description": "Uploaded light curve",
                "data": None,
                "message": f"Unsupported file extension '{suffix}'. Please upload a .npz or .csv light-curve file.",
            }

        valid = np.isfinite(df["time"]) & np.isfinite(df["flux"])
        df = df[valid].sort_values("time").reset_index(drop=True)
        if len(df) < 10:
            return {
                "status": "error",
                "path": None,
                "description": "Uploaded light curve",
                "data": None,
                "message": f"Light curve contains too few valid observations ({len(df)} points). At least 10 points are required.",
            }

        tic_id = normalize_tic(filename) or "Custom"
        stats = lightcurve_stats(df, tic_id)

        return {
            "status": "loaded",
            "path": Path(filename),
            "description": f"Uploaded file ({filename})",
            "data": df,
            "stages": stages,
            "stats": stats,
            "notes": notes,
        }
    except Exception as exc:
        return {
            "status": "error",
            "path": None,
            "description": "Uploaded light curve",
            "data": None,
            "message": f"Failed to parse uploaded file '{filename}': {exc}",
        }


def interpret_prediction(
    predicted_class: str,
    probabilities: dict | None,
    bls_values: dict | None,
    diagnostics: dict | None,
) -> str:
    """Generate a scientific narrative interpretation of the model classification."""
    probs = probabilities or {}
    bls = bls_values or {}
    diag = diagnostics or {}

    score = probs.get(predicted_class, 0.0)
    score_str = f"{score * 100:.1f}%" if score > 0 else "N/A"

    period = bls.get("period_d")
    depth = bls.get("depth_ppm")
    snr = bls.get("snr")
    sde = bls.get("sde")
    odd_even = bls.get("odd_even_sigma")
    sec_sigma = bls.get("secondary_sigma")
    flags = diag.get("flags") or []

    if predicted_class == "Transit":
        msg = f"The LightGBM model classified this target as a **Transit** candidate with an output model score of **{score_str}**."
        details = []
        if period is not None and np.isfinite(period):
            details.append(f"orbital period of **{period:.4f} days**")
        if depth is not None and np.isfinite(depth):
            details.append(f"transit depth of **{depth:.0f} ppm** ({depth / 10000:.3f}%)")
        if snr is not None and np.isfinite(snr):
            details.append(f"signal-to-noise ratio SNR = **{snr:.1f}** (SDE = {sde:.1f})" if sde is not None else f"SNR = **{snr:.1f}**")
        if details:
            msg += f" The detected signal demonstrates an {', a '.join(details)}."

        if odd_even is not None and np.isfinite(odd_even) and odd_even <= 3.0:
            msg += f" Odd-even transit symmetry is consistent ({odd_even:.2f}σ <= 3.0σ), disfavoring an eclipsing binary."
        if sec_sigma is not None and np.isfinite(sec_sigma) and sec_sigma <= 3.0:
            msg += f" No statistically significant secondary eclipse was detected at phase 0.5 ({sec_sigma:.2f}σ <= 3.0σ), supporting a planetary transit interpretation."
        return msg

    elif predicted_class == "Eclipsing Binary":
        msg = f"The LightGBM model classified this target as an **Eclipsing Binary (EB)** with an output model score of **{score_str}**."
        eb_reasons = []
        if odd_even is not None and np.isfinite(odd_even) and odd_even > 3.0:
            eb_reasons.append(f"significant odd-even depth difference of **{odd_even:.2f}σ** (primary vs. secondary transit mismatch)")
        if sec_sigma is not None and np.isfinite(sec_sigma) and sec_sigma > 3.0:
            eb_reasons.append(f"a detectable secondary eclipse at phase 0.5 (**{sec_sigma:.2f}σ**)")
        if depth is not None and np.isfinite(depth) and depth > 50000:
            eb_reasons.append(f"a deep transit depth of **{depth:.0f} ppm** ({depth / 10000:.1f}%), exceeding typical exoplanetary radii")
        if "too_long_for_period" in flags:
            eb_reasons.append("a transit duration that exceeds orbital kinematic limits for planetary bodies")

        if eb_reasons:
            msg += f" This classification is supported by {', '.join(eb_reasons)}."
        else:
            msg += " The combination of statistical transit shape features and BLS depth signatures closely matches the eclipsing binary training distribution."
        return msg

    else:  # Other
        msg = f"The LightGBM model classified this target as **Other** with an output model score of **{score_str}**."
        reasons = []
        if snr is not None and np.isfinite(snr) and snr < 7.1:
            reasons.append(f"low BLS detection significance (SNR = {snr:.1f} < 7.1)")
        if flags:
            reasons.append(f"diagnostic flags: `{', '.join(flags)}`")
        if reasons:
            msg += f" The signal exhibits {', '.join(reasons)}, consistent with stellar variability, instrumental noise artifacts, or unclassified non-planetary fluctuations."
        else:
            msg += " The extracted photometric features and transit parameters are inconsistent with coherent planetary transits or clean eclipsing binaries."
        return msg

