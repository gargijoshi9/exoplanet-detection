"""
Streamlit dashboard for AI-enabled exoplanet detection and classification
from noisy TESS light curves.

Launch from any directory:
    python -m streamlit run app.py
The app resolves all files from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dashboard_utils as du

PAGES = [
    "Project Overview",
    "Light Curve Explorer",
    "BLS Detection",
    "ML Classification",
    "Transit Parameter Fitting",
    "Results and Export",
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(11,18,32,0)",
    plot_bgcolor="#111827",
    font=dict(color="#e8eef7", size=13),
    margin=dict(l=50, r=20, t=50, b=50),
    legend=dict(bgcolor="rgba(21,32,51,0.8)"),
    xaxis=dict(gridcolor="#1f2a3d", zerolinecolor="#1f2a3d"),
    yaxis=dict(gridcolor="#1f2a3d", zerolinecolor="#1f2a3d"),
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0b1220; }
    [data-testid="stSidebar"] { background-color: #10192b; }
    .hero {
        padding: 0.4rem 0 0.8rem 0;
        border-bottom: 1px solid #243044;
        margin-bottom: 1.1rem;
    }
    .hero h1 { font-size: 1.55rem; margin-bottom: 0.25rem; color: #e8eef7; }
    .hero p { color: #9fb0c8; margin: 0; }
    .note, .warn, .ok {
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.95rem;
        line-height: 1.45;
    }
    .note { background: #152033; border: 1px solid #2a3b57; color: #c5d3e6; }
    .warn { background: #2a1c14; border: 1px solid #8a4b24; color: #f0d2b6; }
    .ok { background: #10261c; border: 1px solid #2f6a4a; color: #c6ead6; }
    div[data-testid="stMetric"] {
        background: #152033;
        border: 1px solid #2a3b57;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    div[data-testid="stMetric"] label { color: #9fb0c8; }
</style>
"""


def apply_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    fig.update_xaxes(title=x_title, gridcolor="#1f2a3d", zerolinecolor="#1f2a3d")
    fig.update_yaxes(title=y_title, gridcolor="#1f2a3d", zerolinecolor="#1f2a3d")
    return fig


@st.cache_data(show_spinner=False)
def cached_catalog() -> pd.DataFrame:
    return du.build_catalog()


@st.cache_resource(show_spinner=False)
def cached_model() -> dict:
    return du.load_model_artifact()


@st.cache_data(show_spinner=False)
def cached_evaluation() -> dict:
    return du.evaluation_bundle()


@st.cache_data(show_spinner=False)
def cached_lightcurve(path_str: str) -> dict:
    bundle = du.load_lightcurve(Path(path_str))
    # DataFrames are cacheable; Path objects are converted below in the UI.
    if bundle.get("path") is not None:
        bundle = dict(bundle)
        bundle["path"] = str(bundle["path"])
    return bundle


@st.cache_data(show_spinner=False)
def cached_detection_csv(path_str: str) -> dict:
    artifact = du.load_detection_table(Path(path_str))
    if artifact.get("path") is not None:
        artifact = dict(artifact)
        artifact["path"] = str(artifact["path"])
    return artifact


def unavailable_note(message: str, kind: str = "note"):
    st.markdown(f'<div class="{kind}">{message}</div>', unsafe_allow_html=True)


def metric_or_na(label: str, value, help_text: str):
    display = du.format_count(value) if value is not None else "n/a"
    st.metric(label, display, help=help_text)


def selected_target_files(catalog: pd.DataFrame, tic: str) -> dict:
    detections = du.discover_detection_files()
    lightcurves = du.discover_lightcurve_files()
    return {
        "lightcurves": lightcurves.get(tic, []),
        "detection": detections.get(tic),
        "catalog_row": catalog[catalog["tic_id"].astype(str) == str(tic)] if not catalog.empty else pd.DataFrame(),
    }


def session_detection(tic: str) -> dict | None:
    store = st.session_state.get("detection_runs") or {}
    return store.get(tic)


def store_detection(tic: str, table: pd.DataFrame, details: list, source: str):
    runs = st.session_state.setdefault("detection_runs", {})
    serializable_details = []
    for item in details:
        serializable_details.append({
            "planet": int(item["planet"]),
            "periods": np.asarray(item["periods"]),
            "power": np.asarray(item["power"]),
            "period": float(item["period"]),
            "t0": float(item["t0"]),
            "duration": float(item["duration"]),
            "depth": float(item["depth"]),
            "snr": float(item["snr"]),
        })
    runs[tic] = {"table": table, "details": serializable_details, "source": source}


def detection_table_for(tic: str, files: dict) -> tuple[pd.DataFrame | None, str]:
    live = session_detection(tic)
    if live is not None:
        return live["table"], f"In-session BLS run ({live['source']})"
    info = files.get("detection")
    if info:
        loaded = cached_detection_csv(str(info["csv"]))
        if loaded["status"] == "loaded":
            return loaded["data"], f"Saved file {info['csv'].relative_to(du.ROOT)}"
        return None, loaded.get("message") or "Detection CSV could not be read."
    bls = du.load_bls_features()
    if bls["status"] == "loaded" and "tic_id" in bls["data"].columns:
        sub = bls["data"].copy()
        sub["tic_id"] = sub["tic_id"].map(du.normalize_tic)
        sub = sub[sub["tic_id"] == tic]
        if not sub.empty:
            return sub.reset_index(drop=True), f"Saved BLS features {du.BLS_FEATURES_PATH.relative_to(du.ROOT)}"
    return None, "No saved detection table for this target."


def render_sidebar(catalog: pd.DataFrame) -> tuple[str, str | None]:
    st.sidebar.markdown("### Exoplanet Detection")
    st.sidebar.caption("TESS light curves · BLS · LightGBM")
    page = st.sidebar.radio("Navigate", PAGES, index=0)
    st.sidebar.divider()
    targets = catalog["tic_id"].astype(str).tolist() if not catalog.empty else []
    if not targets:
        st.sidebar.info("No TIC IDs found yet. Add processed NPZ/CSV files or data/labels.csv.")
        return page, None
    tic = st.sidebar.selectbox("Target (TIC ID)", targets, index=0)
    row = catalog.loc[catalog["tic_id"].astype(str) == tic].iloc[0]
    st.sidebar.markdown(
        f"**Light curve:** {'yes' if bool(row['light_curve_available']) else 'no'}  \n"
        f"**Catalog label:** {row['label'] if pd.notna(row['label']) else 'not in labels.csv'}  \n"
        f"**Saved class:** {row['predicted_class'] if pd.notna(row['predicted_class']) else 'none'}"
    )
    return page, tic


def page_overview(catalog: pd.DataFrame, metrics: dict, evaluation: dict):
    st.markdown(
        '<div class="hero"><h1>Project overview</h1>'
        "<p>Counts and tables below are computed from files that currently exist "
        "in this repository checkout. Missing artifacts are shown as n/a, not zero.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_or_na(
            "Available targets",
            metrics["n_available_targets"],
            "Unique TIC IDs with a loadable processed light curve (NPZ or CSV).",
        )
    with c2:
        metric_or_na(
            "Listed in labels.csv",
            metrics["n_targets_in_labels"],
            "Rows in data/labels.csv. n/a means the file is absent, not that the count is zero.",
        )
    with c3:
        metric_or_na(
            "Targets analyzed",
            metrics["n_analyzed"],
            "Targets with a saved detection CSV or a row in data/features/bls_features.csv.",
        )
    with c4:
        metric_or_na(
            "Transit candidates",
            metrics["n_candidates"],
            "Rows with status == 'candidate' in saved detection outputs.",
        )

    if metrics["n_targets_in_labels"] is None:
        unavailable_note(metrics["labels_message"] or "data/labels.csv is not present.")
    if metrics["n_analyzed"] is None:
        unavailable_note(
            "No detection outputs were found under results/detection/ and "
            "data/features/bls_features.csv is missing, so the number of analyzed "
            "targets cannot be computed yet."
        )
    if metrics["n_candidates"] is None:
        unavailable_note(
            "Detected transit-candidate counts are unavailable until a detection "
            "table is saved. Zero would mean the files exist and contain no candidates."
        )

    st.subheader("Catalog labels")
    label_source = metrics.get("label_source")
    label_cols = st.columns(3)
    for col, name in zip(label_cols, du.CLASS_ORDER):
        with col:
            metric_or_na(
                name,
                metrics["label_counts"].get(name),
                f"From {label_source}" if label_source else "data/labels.csv is not present.",
            )
    if label_source is None:
        unavailable_note(
            "Catalog class counts are unavailable because data/labels.csv was not found. "
            "That is not the same as zero stars in each class."
        )
    else:
        st.caption(f"Catalog labels source: {label_source}. These are catalog dispositions, not model predictions.")

    st.subheader("Predicted classes")
    class_cols = st.columns(3)
    source = metrics["class_source"]
    for col, name in zip(class_cols, du.CLASS_ORDER):
        with col:
            metric_or_na(
                name,
                metrics["class_counts"].get(name),
                f"From {source}" if source else "No saved classification table was found.",
            )
    if source is None:
        unavailable_note(
            "Predicted class counts are unavailable. There is no held-out prediction "
            f"file at {du.TEST_PREDICTIONS_PATH.relative_to(du.ROOT)}, and the dashboard "
            "does not invent class labels."
        )
    else:
        st.caption(f"Predicted class source: {source}. These are not live scores from the sidebar target.")

    st.subheader("Model evaluation")
    if evaluation.get("metrics"):
        m = evaluation["metrics"]
        st.caption(m["split"])
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Test stars", du.format_count(m["n_test"]))
        e2.metric("Accuracy", f"{m['accuracy']:.3f}")
        e3.metric("Macro precision", f"{m['precision_macro']:.3f}")
        e4.metric("Macro recall", f"{m['recall_macro']:.3f}")
        e5.metric("Macro F1", f"{m['f1_macro']:.3f}")
    else:
        unavailable_note(evaluation.get("message") or "Evaluation metrics are unavailable.")

    st.subheader("Target summary")
    if catalog.empty:
        unavailable_note("No targets could be assembled from labels or light-curve files.")
    else:
        show = catalog.copy()
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption(f"{len(show)} row(s) assembled from local artifacts.")


def page_explorer(tic: str | None, files: dict):
    st.markdown(
        '<div class="hero"><h1>Light curve explorer</h1>'
        "<p>Plots use the selected target's existing processed photometry. "
        "Stages that were never saved are not reconstructed.</p></div>",
        unsafe_allow_html=True,
    )
    if tic is None:
        unavailable_note("Select a TIC ID once a light curve or labels file is present.")
        return
    lc_files = files["lightcurves"]
    if not lc_files:
        unavailable_note(f"No processed light curve file was found for TIC {tic}.")
        return

    labels = [item["relative"] for item in lc_files]
    choice = st.selectbox("Light-curve file", labels, index=0)
    chosen = next(item for item in lc_files if item["relative"] == choice)
    bundle = cached_lightcurve(str(chosen["path"]))
    if bundle["status"] != "loaded":
        unavailable_note(bundle.get("message") or "Failed to load light curve.", "warn")
        return

    df = bundle["data"]
    stats = bundle["stats"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TIC ID", stats["tic_id"] or "unknown")
    m2.metric("Observations", du.format_count(stats["n_points"]))
    coverage = stats["time_coverage_d"]
    m3.metric("Time coverage (days)", f"{coverage:.2f}" if np.isfinite(coverage) else "n/a")
    missing_total = sum(stats["missing"].values())
    m4.metric("Missing values", du.format_count(missing_total))
    st.caption(
        f"Time range BTJD {stats['time_start']:.4f} to {stats['time_end']:.4f}. "
        f"Columns: {', '.join(stats['columns'])}."
    )
    for note in bundle.get("notes") or []:
        st.caption(note)

    display, downsampled = du.downsample_frame(df, max_points=8000)
    fig = go.Figure()
    error_y = None
    if "flux_err" in display.columns:
        error_y = dict(type="data", array=display["flux_err"], visible=True, thickness=0.6, width=0)
    fig.add_trace(go.Scatter(
        x=display["time"],
        y=display["flux"],
        mode="markers",
        marker=dict(size=3, color="#7eb6ff"),
        name="Normalized flux",
        error_y=error_y,
    ))
    fig = apply_layout(
        fig,
        f"TIC {tic}: processed flux vs time",
        "Time (BTJD, days)",
        "Normalized flux",
    )
    st.plotly_chart(fig, width="stretch")
    if downsampled:
        st.caption(
            f"The interactive plot shows every {int(np.ceil(len(df)/8000))}th point "
            f"({len(display):,} of {len(df):,}) so the browser stays responsive. "
            "The metric cards use the full series. Flux error bars are the saved flux_err values."
        )
    elif "flux_err" in df.columns:
        st.caption("Error bars are the saved flux_err column.")
    else:
        unavailable_note("No flux_err column is present, so uncertainty bars cannot be drawn.")

    st.subheader("Processing stages")
    stages = bundle.get("stages") or {}
    stage_labels = {
        "raw": "Raw PDCSAP",
        "cleaned": "Outlier-cleaned, before flatten",
        "normalized_not_detrended": "Normalized, not detrended (flux_norm)",
        "normalized_detrended": "Detrended and normalized (flux)",
    }
    available = {k: v for k, v in stages.items() if v and v in df.columns}
    if len(available) <= 1:
        unavailable_note(
            "Only the processed flux series is stored for this file. Raw, cleaned, "
            "and (for this CSV) non-detrended flux_norm arrays were not saved, so a "
            "stage comparison cannot be shown without fabricating photometry."
        )
    else:
        compare = st.multiselect(
            "Series to overlay",
            options=list(available.keys()),
            default=list(available.keys()),
            format_func=lambda k: stage_labels.get(k, k),
        )
        fig2 = go.Figure()
        for key in compare:
            series, ds = du.downsample_frame(df[["time", available[key]]].rename(columns={available[key]: "y"}))
            fig2.add_trace(go.Scatter(
                x=series["time"], y=series["y"], mode="markers",
                marker=dict(size=3), name=stage_labels.get(key, key),
            ))
        fig2 = apply_layout(fig2, "Available processing stages", "Time (BTJD, days)", "Flux")
        st.plotly_chart(fig2, width="stretch")
    missing_stages = [stage_labels[k] for k, v in stages.items() if not v]
    if missing_stages:
        st.caption("Not available for this file: " + "; ".join(missing_stages) + ".")


def page_detection(tic: str | None, files: dict):
    st.markdown(
        '<div class="hero"><h1>BLS / TLS detection</h1>'
        "<p>Values come from src/detect.py (Box Least Squares). TLS is not implemented. "
        "Periodogram arrays are only available after an in-session run; saved outputs are CSV tables and PNG plots.</p></div>",
        unsafe_allow_html=True,
    )
    if tic is None:
        unavailable_note("Select a target to inspect detection results.")
        return

    notice = st.session_state.pop("detection_notice", None)
    if notice:
        st.success(notice)

    table, origin = detection_table_for(tic, files)
    if table is None:
        unavailable_note(origin)
    else:
        st.caption(f"Detection table source: {origin}")
        display_cols = [c for c in du.RESULT_COLUMNS if c in table.columns]
        st.dataframe(
            table[display_cols] if display_cols else table,
            width="stretch",
            hide_index=True,
        )
        first = table.iloc[0]
        kpis = st.columns(6)
        mapping = [
            ("Orbital period (days)", "period_d", "{:.4f}"),
            ("Transit depth (ppm)", "depth_ppm", "{:.1f}"),
            ("Duration (hours)", "duration_h", "{:.2f}"),
            ("SNR", "snr", "{:.2f}"),
            ("Red-noise SNR", "snr_red", "{:.2f}"),
            ("SDE", "sde", "{:.2f}"),
        ]
        for col, (label, key, fmt) in zip(kpis, mapping):
            value = first.get(key, np.nan) if key in first.index else np.nan
            text = fmt.format(float(value)) if pd.notna(value) else "n/a"
            col.metric(label, text)
        status = first.get("status", "n/a") if "status" in first.index else "n/a"
        flags = first.get("flags", "") if "flags" in first.index else ""
        st.markdown(
            f"**First-signal status:** `{status}` &nbsp;&nbsp; **Flags:** "
            f"`{flags if str(flags).strip() else 'none'}`"
        )

    lc_files = files["lightcurves"]
    lc_bundle = cached_lightcurve(str(lc_files[0]["path"])) if lc_files else None
    if table is not None and lc_bundle and lc_bundle["status"] == "loaded":
        st.subheader("Phase-folded light curve")
        planet_options = table["planet"].tolist() if "planet" in table.columns else list(range(1, len(table) + 1))
        pick = st.selectbox("Signal to fold", planet_options, index=0)
        row = table[table["planet"] == pick].iloc[0] if "planet" in table.columns else table.iloc[0]
        needed = ["period_d", "t0_btjd", "duration_h"]
        if any(k not in row.index or pd.isna(row[k]) for k in needed):
            unavailable_note("This row does not include period, t0, and duration, so it cannot be folded.")
        else:
            folded = du.phase_fold_frame(
                lc_bundle["data"], row["period_d"], row["t0_btjd"], row["duration_h"]
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=folded["hours_from_mid_transit"],
                y=folded["flux"],
                mode="markers",
                marker=dict(size=3, color="#c5d3e6"),
                name="Data",
            ))
            if "depth_ppm" in row.index and pd.notna(row["depth_ppm"]):
                half = float(folded["hours_from_mid_transit"].abs().max()) if len(folded) else 3.0
                xs, ys = du.bls_box_trace(row["duration_h"], row["depth_ppm"], half)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="BLS box", line=dict(color="#5b9fd4")))
            fig = apply_layout(
                fig,
                f"TIC {tic} folded at P = {row['period_d']:.4f} d",
                "Hours from mid-transit",
                "Normalized flux",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("The blue box is the BLS rectangular model, not a physical transit fit.")
    elif table is not None:
        unavailable_note("A detection table exists, but no light curve is available to phase-fold.")

    st.subheader("Periodogram")
    live = session_detection(tic)
    if live and live.get("details"):
        details = live["details"]
        planets = [int(d["planet"]) for d in details]
        planet = st.selectbox("Periodogram signal", planets, index=0, key="pg_signal")
        det = next(d for d in details if int(d["planet"]) == planet)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=det["periods"], y=det["power"], mode="lines",
            line=dict(color="#e8eef7", width=1), name="BLS SNR",
        ))
        fig.add_vline(x=det["period"], line_color="#e05d5d")
        fig.update_xaxes(type="log")
        fig = apply_layout(fig, f"BLS periodogram · signal {planet}", "Period (days)", "BLS SNR")
        st.plotly_chart(fig, width="stretch")
        st.caption("Periodogram arrays come from the in-session detect() run, not from the CSV.")
    else:
        pngs = files["detection"]["periodograms"] if files.get("detection") else []
        if pngs:
            labels = [p.name for p in pngs]
            chosen = st.selectbox("Saved periodogram image", labels, index=0)
            path = next(p for p in pngs if p.name == chosen)
            st.image(str(path), caption=str(path.relative_to(du.ROOT)))
            st.caption(
                "The detector saves periodogram PNG images, not the underlying period/power arrays. "
                "Run BLS below if you need an interactive periodogram."
            )
        else:
            unavailable_note(
                "No periodogram arrays or PNG files are stored for this target. "
                "src/detect.py writes images only when you run it; the CSV does not include the grid."
            )

    saved = files.get("detection")
    if saved:
        overview = saved.get("overview")
        folds = saved.get("folds") or []
        if overview is not None and Path(overview).exists():
            st.subheader("Saved overview plot")
            st.image(str(overview), caption=str(Path(overview).relative_to(du.ROOT)))
        if folds:
            st.subheader("Saved folded-transit images")
            fold_name = st.selectbox("Fold image", [p.name for p in folds], index=0)
            fold_path = next(p for p in folds if p.name == fold_name)
            st.image(str(fold_path), caption=str(fold_path.relative_to(du.ROOT)))

    st.subheader("Run BLS on this target")
    if not lc_files:
        unavailable_note("A light curve is required to run detection.")
        return
    unavailable_note(
        "BLS on a full-cadence TESS light curve can take several minutes. "
        "It runs only when you submit the form, not on every dashboard rerun.",
        "warn",
    )
    lc_labels = [item["relative"] for item in lc_files]
    with st.form("bls_form"):
        lc_choice = st.selectbox("Light-curve file", lc_labels, index=0)
        c1, c2 = st.columns(2)
        max_planets = c1.number_input("Maximum signals", min_value=1, max_value=4, value=4, step=1)
        min_snr = c2.number_input("Minimum SNR", min_value=0.1, max_value=50.0, value=7.1, step=0.1)
        save_outputs = st.checkbox(
            "Save CSV and PNG files to results/detection using src.detect.save_outputs",
            value=False,
        )
        submitted = st.form_submit_button("Run BLS detection")
    if submitted:
        chosen = next(item for item in lc_files if item["relative"] == lc_choice)
        bundle = cached_lightcurve(str(chosen["path"]))
        if bundle["status"] != "loaded":
            unavailable_note(bundle.get("message") or "Could not load light curve.", "warn")
            return
        lc = bundle["data"]
        try:
            with st.spinner("Running src.detect.detect — this may take several minutes..."):
                new_table, details = du.run_detection(
                    lc["time"].to_numpy(),
                    lc["flux"].to_numpy(),
                    flux_err=lc["flux_err"].to_numpy() if "flux_err" in lc.columns else None,
                    max_planets=int(max_planets),
                    min_snr=float(min_snr),
                    verbose=False,
                )
            store_detection(tic, new_table, details, "src.detect.detect")
            notice = "BLS finished. Metrics and the periodogram now use the in-session table."
            if save_outputs:
                out_csv = du.persist_detection_outputs(
                    new_table,
                    details,
                    lc["time"].to_numpy(),
                    lc["flux"].to_numpy(),
                    Path(chosen["path"]).stem,
                )
                cached_catalog.clear()
                cached_detection_csv.clear()
                notice += f" Saved {out_csv.relative_to(du.ROOT)}."
            st.session_state["detection_notice"] = notice
            st.rerun()
        except Exception as exc:
            unavailable_note(f"BLS detection failed: {exc}", "warn")


def page_classification(tic: str | None, files: dict, catalog: pd.DataFrame):
    st.markdown(
        '<div class="hero"><h1>Machine learning classification</h1>'
        "<p>Inference uses models/lightgbm_bls_pipeline.joblib with the exact 25-feature "
        "schema and training-set medians from src/train_final_model.py.</p></div>",
        unsafe_allow_html=True,
    )
    model_art = cached_model()
    if model_art["status"] != "loaded":
        unavailable_note(model_art.get("message") or "Model could not be loaded.", "warn")
        return
    payload = model_art["data"]
    st.caption(
        f"Loaded {payload['model_name']} from {payload['path'].relative_to(du.ROOT)} · "
        f"training rows recorded in artifact: {payload['training_rows'] if payload['training_rows'] is not None else 'n/a'} · "
        f"classes: {', '.join(payload['classes'])}"
    )
    st.caption("Feature order: " + ", ".join(payload["features"]))
    unavailable_note(
        "A scikit-learn version warning may appear because the artifact was pickled with "
        "an older sklearn than this environment. Predictions are still produced by the "
        "saved LightGBM booster; treat them as research-grade, not a re-validated release."
    )

    st.subheader("Prediction for selected target")
    saved_row = files["catalog_row"]
    if not saved_row.empty and pd.notna(saved_row.iloc[0].get("predicted_class")):
        st.markdown(
            f"**Saved predicted class:** `{saved_row.iloc[0]['predicted_class']}` "
            f"({saved_row.iloc[0].get('prediction_origin', 'saved table')})"
        )
    else:
        st.caption("No saved prediction row exists for this target.")

    if tic is None or not files["lightcurves"]:
        unavailable_note("A processed light curve is required to run live inference.")
    else:
        unavailable_note(
            "Live classification runs extract_features and, unless a detection table is "
            "already in memory or on disk, the BLS detector. Submit the form to run it.",
            "warn",
        )
        with st.form("classify_form"):
            reuse = st.checkbox(
                "Reuse the current detection table when available (otherwise BLS is run again)",
                value=True,
            )
            go_classify = st.form_submit_button("Classify this target")
        if go_classify:
            bundle = cached_lightcurve(str(files["lightcurves"][0]["path"]))
            if bundle["status"] != "loaded":
                unavailable_note(bundle.get("message") or "Light curve missing.", "warn")
            else:
                table, origin = detection_table_for(tic, files)
                use_table = table if reuse and table is not None else None
                try:
                    with st.spinner("Running the saved inference pipeline..."):
                        result = du.classify_lightcurve(bundle["data"], payload, detection_table=use_table)
                    st.session_state.setdefault("classifications", {})[tic] = result
                    if use_table is not None:
                        st.caption(f"BLS features taken from: {origin}")
                    else:
                        st.caption("BLS features computed now with src.detect.detect (first signal).")
                except Exception as exc:
                    unavailable_note(f"Classification failed: {exc}", "warn")

    live = (st.session_state.get("classifications") or {}).get(tic)
    if live:
        st.markdown(f"**Predicted class:** `{live['predicted_class']}`")
        if live.get("probabilities"):
            probs = pd.DataFrame(
                [{"class": k, "score": v} for k, v in live["probabilities"].items()]
            )
            fig = px.bar(probs, x="class", y="score", color="class")
            fig.update_layout(**PLOTLY_LAYOUT, title="Class scores (predict_proba)", showlegend=False)
            fig.update_yaxes(range=[0, 1], title="Score")
            st.plotly_chart(fig, width="stretch")
            st.caption(live["probability_note"])
        if live.get("imputed_features"):
            unavailable_note(
                "These features were missing after extraction and were filled with "
                "training medians, then 0 if still missing: "
                + ", ".join(live["imputed_features"]),
                "warn",
            )
        diag = live.get("bls_diagnostics") or {}
        st.caption(
            f"BLS diagnostics used for the feature vector: found={diag.get('detection_found')}, "
            f"status={diag.get('status')}, flags={diag.get('flags') or 'none'}."
        )

    st.subheader("Held-out evaluation")
    evaluation = cached_evaluation()
    if evaluation.get("metrics"):
        m = evaluation["metrics"]
        st.caption(m["split"] + f" · n = {m['n_test']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']:.3f}")
        c2.metric("Macro precision", f"{m['precision_macro']:.3f}")
        c3.metric("Macro recall", f"{m['recall_macro']:.3f}")
        c4.metric("Macro F1", f"{m['f1_macro']:.3f}")
        per_class = []
        for name in evaluation["classes"]:
            row = m["per_class"].get(name, {})
            per_class.append({
                "class": name,
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "f1-score": row.get("f1-score"),
                "support": row.get("support"),
            })
        st.dataframe(pd.DataFrame(per_class), width="stretch", hide_index=True)
        cm = evaluation["confusion"]
        fig = px.imshow(
            cm,
            x=evaluation["classes"],
            y=evaluation["classes"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Confusion matrix (held-out test)")
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="Actual")
        st.plotly_chart(fig, width="stretch")
    else:
        unavailable_note(evaluation.get("message") or "Held-out metrics are unavailable.")
        st.caption(
            "Training-row counts stored in the joblib artifact are not test performance "
            "and are not shown as accuracy."
        )

    st.subheader("Feature importance")
    importance = du.feature_importance_frame(payload)
    if importance is None:
        unavailable_note("The saved model does not expose feature importance.")
    else:
        top = importance.head(15).sort_values("importance_gain")
        fig = px.bar(top, x="importance_gain", y="feature", orientation="h")
        fig.update_layout(**PLOTLY_LAYOUT, title="LightGBM gain importance from the saved booster")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "These values are total gain from the model fitted on the training partition. "
            "They are not permutation importance on the held-out test set. "
            f"A test-set importance CSV would be read from {du.EVAL_IMPORTANCE_PATH.relative_to(du.ROOT)} if present."
        )
        eval_imp = du.read_csv_artifact(du.EVAL_IMPORTANCE_PATH, "Held-out feature importance")
        if eval_imp["status"] == "loaded":
            st.dataframe(eval_imp["data"], width="stretch", hide_index=True)
        else:
            st.caption(eval_imp.get("message"))


def page_fitting():
    st.markdown(
        '<div class="hero"><h1>Transit parameter fitting</h1>'
        "<p>This page reports whether a physical transit-model fit exists in the project. "
        "It does not generate synthetic batman curves.</p></div>",
        unsafe_allow_html=True,
    )
    status = du.fitting_status()
    unavailable_note(status["reason"], "warn")
    if status["modules_named_fit"]:
        st.write("Files whose names suggest fitting code:", status["modules_named_fit"])
    else:
        st.caption("No src/*fit*.py module is present.")
    mentions = status.get("batman_mentions") or []
    if mentions:
        st.caption("Source files that mention batman or transit-parameter fitting: " + ", ".join(mentions))
    st.markdown(
        """
        What a later fitting stage would need, based on the existing pipeline notes:

        - Start from BLS `period_d`, `t0_btjd`, `duration_h`, and `depth_ppm`
        - Re-flatten `flux_norm` with transits masked (that column is stored in batch NPZ files;
          the current TIC 307210830 CSV does **not** include `flux_norm`)
        - Fit physical parameters with uncertainties

        None of those fitted outputs are in `results/` yet.
        """
    )


def page_export(catalog: pd.DataFrame, tic: str | None, files: dict):
    st.markdown(
        '<div class="hero"><h1>Results and export</h1>'
        "<p>Filter and download only tables that actually exist. Empty downloads are labeled as such.</p></div>",
        unsafe_allow_html=True,
    )
    if catalog.empty:
        unavailable_note("There is no results table to search yet.")
        return

    predicted_options = ["All"] + sorted(
        {str(v) for v in catalog["predicted_class"].dropna().unique()}
    )
    status_options = ["All"] + sorted(
        {str(v) for v in catalog["detection_status"].dropna().unique()}
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    predicted = c1.selectbox("Predicted class", predicted_options)
    status = c2.selectbox("Detection status", status_options)
    query = c3.text_input("Search", placeholder="TIC id, filename, status…")
    filtered = du.filter_catalog(catalog, predicted, status, query)
    st.caption(f"{len(filtered)} of {len(catalog)} catalog row(s) match the filters.")
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button(
        "Download displayed catalog CSV",
        data=du.dataframe_csv_bytes(filtered),
        file_name="exoplanet_catalog_filtered.csv",
        mime="text/csv",
        disabled=filtered.empty,
        key="download_filtered_catalog",
    )

    st.subheader("Export selected target")
    if tic is None:
        unavailable_note("Choose a target in the sidebar.")
        return
    table, origin = detection_table_for(tic, files)
    lc_bundle = None
    if files["lightcurves"]:
        lc_bundle = cached_lightcurve(str(files["lightcurves"][0]["path"]))
    payload = du.target_export_payload(tic, catalog, lc_bundle if lc_bundle and lc_bundle["status"] == "loaded" else None, table)
    st.caption(f"Target TIC {tic}. Detection source: {origin}")
    st.dataframe(payload, width="stretch", hide_index=True)
    st.download_button(
        f"Download TIC {tic} available results",
        data=du.dataframe_csv_bytes(payload),
        file_name=f"TIC_{tic}_available_results.csv",
        mime="text/csv",
        key=f"download_target_{tic}",
    )
    st.markdown(
        "Units in detection tables: `period_d` in days, `t0_btjd` in Barycentric TESS JD, "
        "`duration_h` in hours, `depth_ppm` in parts per million, `snr` / `snr_red` / `sde` dimensionless."
    )


def main():
    st.set_page_config(
        page_title="TESS Exoplanet Detection",
        page_icon="🪐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    catalog = cached_catalog()
    metrics = du.overview_metrics(catalog)
    evaluation = cached_evaluation()
    page, tic = render_sidebar(catalog)
    files = selected_target_files(catalog, tic) if tic else {"lightcurves": [], "detection": None, "catalog_row": pd.DataFrame()}

    if page == "Project Overview":
        page_overview(catalog, metrics, evaluation)
    elif page == "Light Curve Explorer":
        page_explorer(tic, files)
    elif page == "BLS Detection":
        page_detection(tic, files)
    elif page == "ML Classification":
        page_classification(tic, files, catalog)
    elif page == "Transit Parameter Fitting":
        page_fitting()
    else:
        page_export(catalog, tic, files)


if __name__ == "__main__":
    main()
