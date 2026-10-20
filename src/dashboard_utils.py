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
