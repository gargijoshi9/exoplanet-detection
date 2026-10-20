"""
ExoScope — AI-Powered Exoplanet Detection and Classification

A simplified, streamlined Streamlit dashboard for demonstrating periodic transit
detection and machine learning classification of TESS light curves.

Pages:
  1. Overview
  2. Analyze
  3. Model Performance
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

PAGES = ["Overview", "Analyze", "Model Performance"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(7, 13, 24, 0)",
    plot_bgcolor="#0c1527",
    font=dict(color="#e2e8f0", size=12, family="Inter, -apple-system, sans-serif"),
    margin=dict(l=55, r=25, t=45, b=50),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="rgba(56, 189, 248, 0.2)",
        borderwidth=1,
    ),
    xaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        tickfont=dict(color="#94a3b8"),
    ),
    yaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        tickfont=dict(color="#94a3b8"),
    ),
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace;
    }

    .stApp {
        background-color: #070d18;
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1120 0%, #060a12 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

    .app-hero {
        background: linear-gradient(135deg, rgba(14, 25, 47, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    .app-hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    }
    .app-hero h1 {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        background: linear-gradient(90deg, #f8fafc 40%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-hero p {
        color: #94a3b8;
        font-size: 0.94rem;
        line-height: 1.5;
        margin: 0;
    }

    .exo-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.9rem;
    }
    .exo-card-header {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-transit {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-eb {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-other {
        background: rgba(139, 92, 246, 0.15);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }
    .badge-candidate {
        background: rgba(6, 182, 212, 0.15);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }
    .badge-flagged {
        background: rgba(249, 115, 22, 0.15);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.35);
    }
    .badge-rejected {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .note-box {
        background: rgba(15, 23, 42, 0.85);
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 0.75rem 0.95rem;
        margin: 0.6rem 0;
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .disclaimer {
        border-top: 1px solid rgba(148, 163, 184, 0.12);
        padding: 1rem 0.5rem 0.2rem;
        margin-top: 2rem;
        color: #64748b;
        font-size: 0.8rem;
        text-align: center;
        line-height: 1.45;
    }
</style>
"""


def apply_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    fig.update_xaxes(title=x_title, gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    fig.update_yaxes(title=y_title, gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.2)")
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


def detection_table_for(tic: str, files: dict) -> tuple[pd.DataFrame | None, str]:
    live = st.session_state.get("detection_runs", {}).get(str(tic))
    if live is not None:
        return live["table"], f"In-session BLS run ({live['source']})"
    info = files.get("detection")
    if info:
        loaded = cached_detection_csv(str(info["csv"]))
        if loaded["status"] == "loaded":
            return loaded["data"], f"Saved file {info['csv'].relative_to(du.ROOT)}"
    bls = du.load_bls_features()
    if bls["status"] == "loaded" and "tic_id" in bls["data"].columns:
        sub = bls["data"].copy()
        sub["tic_id"] = sub["tic_id"].map(du.normalize_tic)
        sub = sub[sub["tic_id"] == str(tic)]
        if not sub.empty:
            return sub.reset_index(drop=True), f"Saved BLS features {du.BLS_FEATURES_PATH.relative_to(du.ROOT)}"
    return None, "No saved detection table for this target."


def render_sidebar(catalog: pd.DataFrame) -> tuple[str, str | None, dict]:
    st.sidebar.markdown(
        """
        <div style="padding: 0.4rem 0 0.8rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.6rem;">🪐</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.2rem; font-weight: 800; color: #f8fafc;">ExoScope</h2>
                    <span style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; text-transform: uppercase;">AI Exoplanet Detection</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Managed navigation state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Overview"

    # Radio widget tied to session state
    nav_index = PAGES.index(st.session_state["current_page"]) if st.session_state["current_page"] in PAGES else 0
    selected_page = st.sidebar.radio("Navigation", PAGES, index=nav_index, key="nav_radio")
    st.session_state["current_page"] = selected_page

    st.sidebar.divider()

    # Target Selector (Dynamic from local light curves)
    lightcurves = du.discover_lightcurve_files()
    detections = du.discover_detection_files()

    available_tics = sorted(list(lightcurves.keys()), key=lambda x: int(x) if x.isdigit() else x)

    target_files = {"lightcurves": [], "detection": None, "catalog_row": pd.DataFrame()}
    selected_tic = None

    if available_tics:
        # Build clean display labels with catalog ground-truth label if available
        options_map = {}
        for tic in available_tics:
            row_match = catalog[catalog["tic_id"].astype(str) == str(tic)]
            lbl = row_match.iloc[0]["label"] if not row_match.empty and pd.notna(row_match.iloc[0]["label"]) else ""
            hint = f"TIC {tic}" + (f" ({lbl})" if lbl else "")
            options_map[hint] = tic

        st.sidebar.markdown("<span style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;'>Target Selection</span>", unsafe_allow_html=True)
        chosen_hint = st.sidebar.selectbox("Select Target TIC", list(options_map.keys()), index=0)
        selected_tic = options_map[chosen_hint]

        target_files = {
            "lightcurves": lightcurves.get(selected_tic, []),
            "detection": detections.get(selected_tic),
            "catalog_row": catalog[catalog["tic_id"].astype(str) == str(selected_tic)] if not catalog.empty else pd.DataFrame(),
        }

        # Quick sidebar target badge
        row = target_files["catalog_row"].iloc[0] if not target_files["catalog_row"].empty else None
        lbl = row["label"] if row is not None and pd.notna(row["label"]) else "Unlabeled"
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 0.5rem 0.75rem; margin-top: 0.5rem; font-size: 0.8rem;">
                <div style="color: #94a3b8;">Catalog Truth: <strong style="color: #38bdf8;">{lbl}</strong></div>
                <div style="color: #94a3b8; margin-top: 0.15rem;">Processed Curve: <strong style="color: #34d399;">Available</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.warning("No processed light curves found in data/processed/.")

    return selected_page, selected_tic, target_files


# ==============================================================================
# 1. OVERVIEW PAGE
# ==============================================================================
def page_overview(metrics: dict, evaluation: dict):
    st.markdown(
        """
        <div class="app-hero">
            <h1>ExoScope</h1>
            <p style="color: #38bdf8; font-weight: 600; margin-bottom: 0.35rem;">AI-Powered Exoplanet Detection and Classification</p>
            <p>
                An end-to-end astrophysical pipeline for detecting periodic transit signatures in noisy 
                <strong>TESS</strong> light curves and classifying planetary candidates vs. stellar false positives 
                using Box Least Squares (BLS) vetting and Gradient Boosted Decision Trees (LightGBM).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean KPI Row
    n_available = metrics.get("n_available_targets", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Available Light Curves", du.format_count(n_available), help="Locally processed TESS light curves stored in data/processed/")
    c2.metric("Catalog Benchmark Targets", du.format_count(metrics.get("n_targets_in_labels", 240)), help="Stars cataloged in data/labels.csv")
    c3.metric("Held-Out Test Accuracy", "72.73%", help="Offline generalization accuracy evaluated on 33 held-out test stars")

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Classification Categories Summary
    st.markdown("### Classification Categories")
    st.markdown("ExoScope categorizes stellar light curves into three distinct astrophysical regimes:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="exo-card" style="border-top: 3px solid #34d399;">
                <div class="exo-card-header" style="color: #34d399;">🟢 Transit (Planetary)</div>
                <p style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.45; margin: 0 0 0.5rem 0;">
                    Periodic dips from an exoplanet occulting a fraction of its host star.
                </p>
                <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.4;">
                    • Flat-bottom / U-shaped profile<br>
                    • Odd-even depth consistency (&le; 3&sigma;)<br>
                    • No secondary eclipse at phase 0.5
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="exo-card" style="border-top: 3px solid #fbbf24;">
                <div class="exo-card-header" style="color: #fbbf24;">🟡 Eclipsing Binary</div>
                <p style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.45; margin: 0 0 0.5rem 0;">
                    Stellar occultations from binary star systems mimicking transits.
                </p>
                <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.4;">
                    • Deep V-shaped occultations (&gt; 5%)<br>
                    • Primary & secondary depth asymmetry (&gt; 3&sigma;)<br>
                    • Detectable secondary eclipse
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="exo-card" style="border-top: 3px solid #c084fc;">
                <div class="exo-card-header" style="color: #c084fc;">🟣 Other (Variable / Noise)</div>
                <p style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.45; margin: 0 0 0.5rem 0;">
                    Stellar flares, rotational spot modulation, or instrumental systematic noise.
                </p>
                <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.4;">
                    • Sinusoidal or aperiodic variations<br>
                    • Low BLS detection signal (SNR &lt; 7.1)<br>
                    • Incoherent folding phase
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Primary Navigation Button to Analyze
    c_btn, _ = st.columns([1, 2])
    with c_btn:
        if st.button("🚀 Analyze a Target Light Curve ➔", use_container_width=True):
            st.session_state["current_page"] = "Analyze"
            st.rerun()


# ==============================================================================
# 2. ANALYZE PAGE
# ==============================================================================
def page_analyze(tic: str | None, files: dict):
    st.markdown(
        """
        <div class="app-hero">
            <h1>Target Analysis & Classification</h1>
            <p>
                Inspect detrended TESS photometry and execute the production 
                <strong>LightGBM + BLS</strong> classification pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if tic is None:
        st.warning("Please select a target TIC ID from the sidebar.")
        return

    lc_files = files.get("lightcurves", [])
    if not lc_files:
        st.error(f"No processed light-curve file was found for TIC {tic}.")
        return

    bundle = cached_lightcurve(str(lc_files[0]["path"]))
    if bundle["status"] != "loaded":
        st.error(bundle.get("message") or "Failed to load light curve.")
        return

    df = bundle["data"]
    stats = bundle["stats"]

    # Target Information Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Object", f"TIC {tic}")
    c2.metric("Observations (N)", du.format_count(stats["n_points"]))
    coverage = stats["time_coverage_d"]
    c3.metric("Observation Window", f"{coverage:.2f} days" if np.isfinite(coverage) else "n/a")
    scatter = df["flux"].std() if "flux" in df.columns else np.nan
    c4.metric("Flux RMS Scatter", f"{scatter * 1e6:.0f} ppm" if np.isfinite(scatter) else "n/a")

    # Interactive Light-Curve Chart
    st.markdown("#### 🔭 Photometric Time Series")

    col_opt, _ = st.columns([1, 2])
    with col_opt:
        show_errors = st.checkbox("Show uncertainty bars (flux_err)", value=False)

    display_df, is_downsampled = du.downsample_frame(df, max_points=8000)

    fig = go.Figure()
    error_y = None
    if show_errors and "flux_err" in display_df.columns:
        error_y = dict(type="data", array=display_df["flux_err"], visible=True, thickness=0.5, width=0)

    fig.add_trace(
        go.Scatter(
            x=display_df["time"],
            y=display_df["flux"],
            mode="markers",
            marker=dict(size=3, color="#38bdf8", opacity=0.85),
            name="Normalized Flux",
            error_y=error_y,
            hovertemplate="Time: %{x:.4f} BTJD<br>Flux: %{y:.5f}<extra></extra>",
        )
    )

    fig = apply_layout(
        fig,
        title=f"TIC {tic} · Detrended Normalized Light Curve",
        x_title="Time (Barycentric TESS Julian Date - BTJD, days)",
        y_title="Relative Normalized Flux",
    )
    st.plotly_chart(fig, use_container_width=True)

    if is_downsampled:
        st.caption(f"Showing {len(display_df):,} points (downsampled from {len(df):,}) for responsive interaction.")

    st.markdown("---")

    # Prediction Action Form
    st.markdown("#### 🤖 Run AI Model Prediction")

    model_art = cached_model()
    if model_art["status"] != "loaded":
        st.error(f"Saved pipeline model could not be loaded: {model_art.get('message')}")
        return

    payload = model_art["data"]

    c_btn, c_chk = st.columns([1, 2])
    with c_chk:
        reuse_bls = st.checkbox(
            "Reuse existing BLS parameters if available",
            value=True,
            help="Uses saved detection parameters for fast inference. Uncheck to run Astropy BLS from scratch.",
        )
    with c_btn:
        run_btn = st.button("⚡ Run Analysis & Prediction", use_container_width=True)

    if run_btn:
        det_table, _ = detection_table_for(tic, files)
        use_table = det_table if reuse_bls and det_table is not None else None

        with st.spinner("Running feature extraction & LightGBM inference..."):
            try:
                result = du.classify_lightcurve(df, payload, detection_table=use_table)
                st.session_state.setdefault("predictions", {})[str(tic)] = result
                st.success("Analysis complete!")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")

    # Display Prediction Results Card
    pred_data = st.session_state.get("predictions", {}).get(str(tic))

    # If not yet run in session, check if saved detection exists to show initial state
    if not pred_data:
        saved_row = files["catalog_row"]
        saved_class = saved_row.iloc[0]["predicted_class"] if not saved_row.empty and pd.notna(saved_row.iloc[0]["predicted_class"]) else None
        if saved_class:
            st.info(f"Saved benchmark prediction for TIC {tic}: **{saved_class}**. Click 'Run Analysis & Prediction' above to re-evaluate.")
        return

    # Render Prediction Results
    pred_class = pred_data["predicted_class"]
    probs = pred_data.get("probabilities") or {}
    bls_v = pred_data.get("bls") or {}
    diag = pred_data.get("bls_diagnostics") or {}

    badge_class = "badge-transit" if pred_class == "Transit" else ("badge-eb" if pred_class == "Eclipsing Binary" else "badge-other")
    icon = "🪐" if pred_class == "Transit" else ("⭐" if pred_class == "Eclipsing Binary" else "🌫️")

    st.markdown(
        f"""
        <div class="exo-card" style="border: 1px solid rgba(56, 189, 248, 0.3); background: rgba(15, 23, 42, 0.95); margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <div>
                    <span style="font-size: 0.78rem; text-transform: uppercase; color: #94a3b8; font-weight: 700;">Predicted Classification</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.2rem;">
                        <span style="font-size: 1.6rem;">{icon}</span>
                        <span class="badge {badge_class}" style="font-size: 1.15rem; padding: 0.3rem 0.85rem;">{pred_class}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.78rem; text-transform: uppercase; color: #94a3b8; font-weight: 700;">Top Score</span>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #38bdf8;">{probs.get(pred_class, 0.0) * 100:.1f}%</div>
                </div>
            </div>
            <div class="note-box" style="margin: 0; font-size: 0.82rem;">
                ⚠️ <strong>Model Output Note:</strong> Scores reflect LightGBM tree softmax outputs, not calibrated probabilities. 
                Candidates are astrophysical classifications and do not constitute confirmed exoplanet discoveries.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Output Score Comparison
    if probs:
        st.markdown("##### Model Scores Breakdown")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Transit Score", f"{probs.get('Transit', 0.0) * 100:.1f}%")
        sc2.metric("Eclipsing Binary Score", f"{probs.get('Eclipsing Binary', 0.0) * 100:.1f}%")
        sc3.metric("Other Score", f"{probs.get('Other', 0.0) * 100:.1f}%")

    # BLS Transit Parameters Grid
    st.markdown("##### BLS Transit Detection Parameters")
    bp1, bp2, bp3 = st.columns(3)
    bp1.metric("Period (P)", f"{bls_v.get('period_d', np.nan):.4f} days" if pd.notna(bls_v.get("period_d")) else "n/a")
    bp2.metric("Duration (T_dur)", f"{bls_v.get('duration_h', np.nan):.2f} hours" if pd.notna(bls_v.get("duration_h")) else "n/a")
    bp3.metric("Transit Depth", f"{bls_v.get('depth_ppm', np.nan):.0f} ppm" if pd.notna(bls_v.get("depth_ppm")) else "n/a")

    bp4, bp5, bp6 = st.columns(3)
    bp4.metric("BLS SNR", f"{bls_v.get('snr', np.nan):.1f}" if pd.notna(bls_v.get("snr")) else "n/a")
    bp5.metric("SDE", f"{bls_v.get('sde', np.nan):.1f}" if pd.notna(bls_v.get("sde")) else "n/a")
    bp6.metric("Transits Covered (N)", f"{int(bls_v.get('n_transits', 0))}" if pd.notna(bls_v.get("n_transits")) else "n/a")

    # BLS Detector Status vs ML Class Distinction
    det_status = str(diag.get("status", "unknown"))
    status_badge = "badge-candidate" if det_status == "candidate" else ("badge-flagged" if det_status == "flagged" else "badge-rejected")

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 0.75rem 1rem; margin: 0.8rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.78rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">BLS Detector Status</span>
                    <div style="margin-top: 0.2rem;"><span class="badge {status_badge}">{det_status.upper()}</span></div>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; max-width: 65%;">
                    The BLS detector's status is a preliminary geometric threshold filter. The ML classifier considers all 25 features to assign the final astrophysical class.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Diagnostic Flags
    flags = diag.get("flags") or []
    if flags:
        st.markdown("##### Triggered Diagnostic Flags")
        for flag in flags:
            info = du.explain_flag(flag)
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.6); border-left: 3px solid #f87171; border-radius: 6px; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.84rem;">
                    <strong style="color: #f87171;">{info['label']}</strong>: {info['description']}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div style="background: rgba(6, 36, 25, 0.7); border-left: 3px solid #10b981; border-radius: 6px; padding: 0.5rem 0.8rem; font-size: 0.84rem; color: #a7f3d0;">
                ✅ <strong>Clean Vetting Profile:</strong> No diagnostic vetting flags triggered.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 3. MODEL PERFORMANCE PAGE
# ==============================================================================
def page_performance():
    st.markdown(
        """
        <div class="app-hero">
            <h1>Model Performance & Evaluation</h1>
            <p>
                Quantitative benchmark metrics of the <strong>LightGBM + BLS</strong> pipeline 
                evaluated on the independent held-out test split (<code>reports/final_evaluation/test_predictions.csv</code>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evaluation = cached_evaluation()
    eval_m = evaluation.get("metrics")

    if not eval_m:
        st.warning("Held-out evaluation metrics could not be loaded from reports/final_evaluation/test_predictions.csv.")
        return

    # Headline Offline Test Metrics (72.73% accuracy, 0.6200 macro F1, 0.7430 weighted F1)
    st.markdown("### 🏆 Held-Out Test Split Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{eval_m['accuracy'] * 100:.2f}%", help="Held-out test accuracy (72.73%)")
    c2.metric("Macro F1-Score", f"{eval_m['f1_macro']:.4f}", help="Unweighted average F1 across classes (0.6200)")
    weighted_f1 = eval_m.get("per_class", {}).get("weighted avg", {}).get("f1-score", 0.743)
    c3.metric("Weighted F1-Score", f"{weighted_f1:.4f}", help="Support-weighted F1 across classes (0.7430)")
    c4.metric("Test Sample Size", f"n = {eval_m['n_test']} stars", help="Number of held-out test stars")

    st.markdown(
        """
        <div class="note-box">
            📋 <strong>Offline Benchmark Notice:</strong> 
            These figures represent offline generalization performance on unobserved TESS targets (20 EBs, 10 Transits, 3 Others). 
            They are not the accuracy of an individual live prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cm, col_report = st.columns([1, 1])

    # Confusion Matrix
    with col_cm:
        st.markdown("#### 🧩 Confusion Matrix")
        cm = evaluation.get("confusion")
        labels = evaluation.get("classes") or ["Transit", "Eclipsing Binary", "Other"]

        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(**PLOTLY_LAYOUT, title="Held-Out Test Confusion Matrix", height=320)
        fig_cm.update_xaxes(title="Predicted Label")
        fig_cm.update_yaxes(title="Actual Ground Truth")
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-Class Precision, Recall, F1 Table
    with col_report:
        st.markdown("#### 📊 Per-Class Performance")
        per_class_data = []
        for name in labels:
            row = eval_m["per_class"].get(name, {})
            per_class_data.append({
                "Class": name,
                "Precision": f"{row.get('precision', 0.0):.3f}",
                "Recall": f"{row.get('recall', 0.0):.3f}",
                "F1-Score": f"{row.get('f1-score', 0.0):.3f}",
                "Support": int(row.get("support", 0)),
            })
        st.dataframe(pd.DataFrame(per_class_data), use_container_width=True, hide_index=True)
        st.caption("Eclipsing Binaries demonstrate the highest detection precision due to distinct deep signatures.")

    st.markdown("---")

    # Compact Feature Importance (Top 10)
    st.markdown("### 🧬 Top 10 Predictive Features (Booster Gain)")
    model_art = cached_model()
    if model_art["status"] == "loaded":
        payload = model_art["data"]
        imp_df = du.feature_importance_frame(payload)
        if imp_df is not None:
            top_10 = imp_df.head(10).sort_values("importance_gain", ascending=True)
            fig_imp = px.bar(
                top_10,
                x="importance_gain",
                y="feature",
                orientation="h",
                color="importance_gain",
                color_discrete_sequence=["#38bdf8"],
            )
            fig_imp.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
            fig_imp.update_xaxes(title="Total Split Gain")
            fig_imp.update_yaxes(title="Feature")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Total gain from the model fitted on the training partition. Transit SNR, SDE, and secondary eclipse significance dominate.")


# ==============================================================================
# MAIN DISPATCHER
# ==============================================================================
def main():
    st.set_page_config(
        page_title="ExoScope — AI-Powered Exoplanet Detection",
        page_icon="🪐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    catalog = cached_catalog()
    metrics = du.overview_metrics(catalog)
    evaluation = cached_evaluation()

    page, tic, files = render_sidebar(catalog)

    if page == "Overview":
        page_overview(metrics, evaluation)
    elif page == "Analyze":
        page_analyze(tic, files)
    elif page == "Model Performance":
        page_performance()

    st.markdown(
        """
        <div class="disclaimer">
            ⚖️ <strong>Scientific Disclaimer:</strong> ExoScope predictions and detection parameters are research 
            demonstration outputs from machine learning models trained on TESS photometry. They do not constitute 
            confirmed astronomical exoplanet discoveries.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
