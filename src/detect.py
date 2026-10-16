"""
Stage 1 - transit detection (Box Least Squares) with iterative multi-planet masking.

Input : a preprocessed (detrended, normalized) light curve: time [days], flux, flux_err
Output: a table with one row per detected periodic signal (period, t0, duration, depth, SNR, vetting flags)

How it works
------------
1. Bin the light curve to 10 minutes (fast search). If the data is already binned, this step is skipped.
2. Run BLS on a log-spaced period grid that is fine enough for the length of the data.
3. Take the highest peak. If its SNR is below the threshold (default 7.1), stop.
4. Refine period / t0 / duration with a narrow BLS search on the full-cadence data.
5. Vet the signal: number of transits, odd/even depth, secondary eclipse, depth and duration sanity.
6. Mask the transits of that signal and go back to step 2 to find the next planet.

Usage (from the repository root)
    python src/detect.py --csv data/full/processed_TIC_307210830.csv
    python src/detect.py --csv data/full/processed_TIC_307210830.csv --max-planets 3 --out results/detection
    python src/detect.py --npz data/processed/TIC_12345.npz            # a star from the classification dataset
"""

import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

ROOT = Path(__file__).resolve().parents[1]

# ---- default settings -------------------------------------------------------
DURATIONS_H = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)   # transit durations to try (hours)
PERIOD_MIN = 0.5            # days
PERIOD_MAX_CAP = 40.0       # days (the real limit is also baseline / 2)
GRID_OVERSAMPLE = 3         # period grid: phase drift over the baseline < duration / 3
MAX_GRID_POINTS = 400_000
BIN_MINUTES = 10
MIN_SNR = 7.1               # same value the TESS pipeline uses for a threshold crossing event
MASK_HALFWIDTH = 1.0        # mask +- (this x duration) around each transit of a found signal
MAX_PLANETS = 4

# vetting limits
MAX_ODD_EVEN_SIGMA = 3.0
MAX_SECONDARY_SIGMA = 3.0
MAX_DEPTH = 0.05            # 5 %: deeper than this is almost never a planet
MAX_DURATION_FRACTION = 0.1  # duration / period
MIN_TRANSITS = 2


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def phase_offset(t, t0, period):
    """Time from the nearest transit centre, in days, in [-P/2, P/2)."""
    return ((np.asarray(t) - t0 + 0.5 * period) % period) - 0.5 * period


RESULT_COLUMNS = [
    "planet", "period_d", "t0_btjd", "duration_h", "depth_ppm", "snr", "snr_red", "sde",
    "n_transits", "odd_even_sigma", "secondary_sigma", "flags", "accepted", "status"
]


def cadence_minutes(t):
    if len(t) < 2:
        return np.nan
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return np.nan
    return float(np.median(diffs) * 1440.0)


def bin_series(t, f, e, minutes):
    """Average a light curve into bins of `minutes`. Bins that are less than half full are dropped."""
    if len(t) == 0:
        return np.array([]), np.array([]), np.array([])
    dt = minutes / 1440.0
    ids = np.floor((t - t[0]) / dt).astype(np.int64)
    _, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    tb = np.bincount(inv, weights=t) / cnt
    fb = np.bincount(inv, weights=f) / cnt
    eb = np.sqrt(np.bincount(inv, weights=e ** 2)) / cnt
    cad = cadence_minutes(t)
    expected = (minutes / cad) if (np.isfinite(cad) and cad > 0) else 1.0
    full = max(1, int(0.5 * expected))
    keep = cnt >= full
    return tb[keep], fb[keep], eb[keep]


def make_period_grid(baseline, period_min, period_max, min_duration, oversample=GRID_OVERSAMPLE):
    """Log-spaced grid. Step: d(ln P) = min_duration / (oversample * baseline).

    A coarse grid lets a true period drift by more than one transit duration over the data, so the signal
    is smeared out and missed. The step shrinks automatically when the data covers more days.
    """
    if baseline <= 0 or period_min <= 0:
        return np.array([period_min])
    period_max = max(period_max, period_min * 1.05)
    dlnp = min_duration / (oversample * baseline)
    if dlnp <= 0:
        return np.array([period_min, period_max])
    n = int(np.ceil(np.log(period_max / period_min) / dlnp))
    n = int(min(max(n, 2000), MAX_GRID_POINTS))
    return np.exp(np.linspace(np.log(period_min), np.log(period_max), n))


def peak_significance(power, n_blocks=200):
    """Peak height above the slowly-rising BLS baseline, in units of robust scatter (a TLS-like SDE)."""
    n = len(power)
    if n == 0:
        return np.nan
    if n < 30:
        sigma = 1.4826 * np.median(np.abs(power - np.median(power)))
        resid = power - np.median(power)
        return float(resid.max() / sigma) if sigma > 0 else np.nan
    n_blocks = min(n_blocks, max(2, n // 10))
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([
        np.median(power[a:b]) if b > a else np.nan
        for a, b in zip(edges[:-1], edges[1:])
    ])
    valid = np.isfinite(med)
    if valid.sum() < 2:
        sigma = 1.4826 * np.median(np.abs(power - np.median(power)))
        resid = power - np.median(power)
        return float(resid.max() / sigma) if sigma > 0 else np.nan
    resid = power - np.interp(np.arange(n), centers[valid], med[valid])
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return float(resid.max() / sigma) if sigma > 0 else np.nan


# ----------------------------------------------------------------------------
# one BLS search + refinement + vetting
# ----------------------------------------------------------------------------
def bls_search(t, f, e, periods, durations_d):
    bls = BoxLeastSquares(t, f, dy=e)
    res = bls.power(periods, durations_d, objective="snr")
    i = int(np.argmax(res.power))
    return {
        "period": float(res.period[i]), "t0": float(res.transit_time[i]),
        "duration": float(res.duration[i]), "depth": float(res.depth[i]),
        "snr": float(res.depth_snr[i]), "sde": peak_significance(np.asarray(res.power)),
        "periods": np.asarray(res.period), "power": np.asarray(res.power),
    }


def refine(t, f, e, period, t0, duration, rel_width=0.002, n_periods=1501):
    """Narrow BLS search around a peak, on the full-cadence data."""
    if len(t) < 10:
        return {"period": period, "t0": t0, "duration": duration, "depth": np.nan, "snr": np.nan}
    periods = np.linspace(period * (1 - rel_width), period * (1 + rel_width), n_periods)
    durations = np.linspace(0.6 * duration, 1.5 * duration, 10)
    try:
        bls = BoxLeastSquares(t, f, dy=e)
        res = bls.power(periods, durations, objective="snr", oversample=20)
        i = int(np.nanargmax(res.power))
        return {"period": float(res.period[i]), "t0": float(res.transit_time[i]),
                "duration": float(res.duration[i]), "depth": float(res.depth[i]),
                "snr": float(res.depth_snr[i])}
    except Exception:
        return {"period": period, "t0": t0, "duration": duration, "depth": np.nan, "snr": np.nan}


def red_noise_snr(t, f, duration, depth, n_transits):
    """SNR that uses the scatter measured on the transit-duration timescale.

    The plain BLS SNR assumes the noise is white. Slow systematics (red noise) make a fake signal look much
    more significant than it is, so we measure how much the flux scatters in bins as long as the transit and
    use that instead. Each transit then counts as one measurement of that scatter.
    """
    if len(t) == 0 or duration <= 0:
        return np.nan
    cad = cadence_minutes(t) / 1440.0
    edges = np.arange(t.min(), t.max() + duration, duration)
    if len(edges) <= 1:
        return np.nan
    idx = np.digitize(t, edges)
    cnt = np.bincount(idx)
    sums = np.bincount(idx, weights=f)
    expected = (duration / cad) if (np.isfinite(cad) and cad > 0) else 1.0
    min_pts = max(1, int(0.5 * expected))
    full = cnt >= min_pts
    if not np.any(full):
        sigma = 1.4826 * np.median(np.abs(f - np.median(f)))
    else:
        means = sums[full] / cnt[full]
        if len(means) < 2:
            sigma = 1.4826 * np.median(np.abs(f - np.median(f)))
        else:
            sigma = 1.4826 * np.median(np.abs(means - np.median(means)))
    return float(depth / (sigma / np.sqrt(max(n_transits, 1)))) if (np.isfinite(sigma) and sigma > 0) else np.nan


def vet(t, f, e, period, t0, duration):
    """Basic false-positive checks. Returns a dict of numbers and a list of flag strings."""
    flags = []
    if len(t) < 5:
        return {
            "n_transits": 0, "depth_err": np.nan, "odd_depth": np.nan, "even_depth": np.nan,
            "odd_even_sigma": np.nan, "secondary_depth": np.nan, "secondary_sigma": np.nan,
            "flags": ["insufficient_data"],
        }
    try:
        bls = BoxLeastSquares(t, f, dy=e)
        s = bls.compute_stats(period, duration, t0)
        depth, depth_err = (float(x) for x in s["depth"])
        odd, odd_err = (float(x) for x in s["depth_odd"])
        even, even_err = (float(x) for x in s["depth_even"])
        sec, sec_err = (float(x) for x in s["depth_phased"])

        cad = cadence_minutes(t) / 1440.0
        min_pts = max(1, int(0.5 * duration / cad)) if (np.isfinite(cad) and cad > 0) else 1
        covered = np.asarray(s["per_transit_count"]) >= min_pts
        n_transits = int(covered.sum())

        with np.errstate(invalid="ignore", divide="ignore"):
            odd_even_denom = np.sqrt(odd_err ** 2 + even_err ** 2)
            odd_even_sigma = abs(odd - even) / odd_even_denom if (np.isfinite(odd_even_denom) and odd_even_denom > 0) else np.nan
            secondary_sigma = sec / sec_err if (np.isfinite(sec_err) and sec_err > 0) else np.nan
    except Exception:
        depth = depth_err = odd = even = odd_even_sigma = sec = secondary_sigma = np.nan
        n_transits = 0
        flags.append("compute_stats_failed")

    if n_transits < MIN_TRANSITS:
        flags.append("few_transits")
    if np.isfinite(odd_even_sigma) and odd_even_sigma > MAX_ODD_EVEN_SIGMA:
        flags.append("odd_even_mismatch")
    if np.isfinite(secondary_sigma) and secondary_sigma > MAX_SECONDARY_SIGMA:
        flags.append("secondary_eclipse")
    if np.isfinite(depth) and depth > MAX_DEPTH:
        flags.append("too_deep")
    elif np.isfinite(depth) and depth <= 0:
        flags.append("negative_depth")
    if duration / period > MAX_DURATION_FRACTION:
        flags.append("too_long_for_period")

    return {
        "n_transits": n_transits, "depth_err": depth_err,
        "odd_depth": odd, "even_depth": even, "odd_even_sigma": float(odd_even_sigma) if np.isfinite(odd_even_sigma) else np.nan,
        "secondary_depth": sec, "secondary_sigma": float(secondary_sigma) if np.isfinite(secondary_sigma) else np.nan,
        "flags": flags,
    }

def calibrate_errors(f, e, verbose=True):
    """Rescale flux_err if it disagrees with the point-to-point scatter by more than 30 %."""
    if len(f) < 2 or len(e) < 2:
        return e
    d = np.diff(f)
    p2p = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    med = float(np.median(e))
    if not (np.isfinite(p2p) and p2p > 0 and med > 0):
        return e
    ratio = med / p2p
    if 0.7 < ratio < 1.3:
        return e
    if verbose:
        print(f"  note: flux_err is {ratio:.2f}x the point-to-point scatter; rescaling it to match", flush=True)
    return e / ratio
# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def detect(time, flux, flux_err=None, *, max_planets=MAX_PLANETS, min_snr=MIN_SNR,
           period_min=PERIOD_MIN, period_max=None, durations_h=DURATIONS_H,
           bin_minutes=BIN_MINUTES, refine_peak=True, verbose=True):
    """Find up to `max_planets` periodic transit signals, masking each one before searching for the next.

    Returns (table, details):
        table   - DataFrame, one row per detection (the last row may be a rejected, low-SNR peak)
        details - list of dicts with the periodogram of every iteration and the arrays used (for plotting)
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    if flux_err is None:
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok]
        if len(f) > 0:
            sigma = 1.4826 * np.median(np.abs(f - np.median(f)))
            sigma = sigma if (np.isfinite(sigma) and sigma > 0) else 1.0
            e = np.full_like(f, sigma)
        else:
            e = np.empty_like(f)
    else:
        e = np.asarray(flux_err, dtype=float)
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
        t, f, e = t[ok], f[ok], e[ok]
        e = calibrate_errors(f, e, verbose)

    if len(t) < 5:
        if verbose:
            print("Insufficient valid data points (< 5).")
        return pd.DataFrame([], columns=RESULT_COLUMNS), []

    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]

    durations_d = np.asarray(durations_h, dtype=float) / 24.0
    baseline = t.max() - t.min()
    pmax = period_max or min(baseline / 2.0, PERIOD_MAX_CAP)
    if pmax <= period_min:
        pmax = period_min * 1.5
    periods = make_period_grid(baseline, period_min, pmax, durations_d.min())

    if bin_minutes and cadence_minutes(t) < 0.8 * bin_minutes:
        tb, fb, eb = bin_series(t, f, e, bin_minutes)
    else:
        tb, fb, eb = t, f, e
    if verbose:
        print(f"{len(t)} points over {baseline:.1f} d | searching {len(periods)} periods "
              f"({period_min}-{pmax:.1f} d) on {len(tb)} points", flush=True)

    mask_full = np.ones(len(t), bool)       # True = keep
    mask_bin = np.ones(len(tb), bool)
    rows, details = [], []

    for k in range(1, max_planets + 1):
        t0_clock = _time.time()
        if mask_bin.sum() < 200:
            break
        if verbose:
            print(f"  signal {k}/{max_planets}: running BLS on {mask_bin.sum():,} points ({len(periods):,} periods)...", flush=True)
        found = bls_search(tb[mask_bin], fb[mask_bin], eb[mask_bin], periods, durations_d)
        passes_white = found["snr"] >= min_snr          # cheap first gate; the final test uses red-noise SNR
        best = dict(found)
        if refine_peak and passes_white:
            best.update(refine(t[mask_full], f[mask_full], e[mask_full],
                               found["period"], found["t0"], found["duration"]))
        v = vet(t[mask_full], f[mask_full], e[mask_full], best["period"], best["t0"], best["duration"])
        snr_red = red_noise_snr(t[mask_full], f[mask_full], best["duration"], best["depth"], v["n_transits"])
        accepted = bool(passes_white and snr_red >= min_snr and best["depth"] > 0)

        row = {
            "planet": k, "period_d": best["period"], "t0_btjd": best["t0"],
            "duration_h": best["duration"] * 24, "depth_ppm": best["depth"] * 1e6,
            "snr": best["snr"], "snr_red": snr_red, "sde": found["sde"], "n_transits": v["n_transits"],
            "odd_even_sigma": v["odd_even_sigma"], "secondary_sigma": v["secondary_sigma"],
            "flags": ",".join(v["flags"]),
            "accepted": bool(accepted),
            # candidate = significant and passed every check; flagged = significant but failed a check
            "status": "rejected" if not accepted else ("flagged" if v["flags"] else "candidate"),
        }
        rows.append(row)
        details.append({"planet": k, "periods": found["periods"], "power": found["power"],
                        "t": t[mask_full].copy(), "f": f[mask_full].copy(), "e": e[mask_full].copy(),
                        **{x: best[x] for x in ("period", "t0", "duration", "depth", "snr")}})
        if verbose:
            tag = "ACCEPTED" if accepted else f"rejected (SNR < {min_snr})"
            print(f"  signal {k}: P = {best['period']:.4f} d, dur = {best['duration']*24:.2f} h, "
                  f"depth = {best['depth']*1e6:.0f} ppm, SNR = {best['snr']:.1f} (red-noise {snr_red:.1f}), "
                  f"transits = {v['n_transits']}, flags = {v['flags'] or 'none'}  "
                  f"[{tag}, {_time.time()-t0_clock:.0f}s]", flush=True)
        if not accepted:
            break

        # mask this signal in both versions of the data, then search again
        hw = MASK_HALFWIDTH * best["duration"]
        mask_full &= np.abs(phase_offset(t, best["t0"], best["period"])) > hw
        mask_bin &= np.abs(phase_offset(tb, best["t0"], best["period"])) > hw

    return pd.DataFrame(rows, columns=RESULT_COLUMNS), details


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def plot_periodogram(det, ax=None, title=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(det["periods"], det["power"], "k-", lw=0.6)
    ax.set_xscale("log")
    p = det["period"]
    ax.axvline(p, color="red", alpha=0.35, lw=3, label=f"peak {p:.4f} d")
    y_top = ax.get_ylim()[1]
    for mult, name in ((0.5, "P/2"), (2, "2P"), (3, "3P")):
        if det["periods"].min() < p * mult < det["periods"].max():
            ax.axvline(p * mult, color="tab:blue", ls=":", lw=1)
            ax.text(p * mult, y_top * 0.96, name, color="tab:blue", ha="center", va="top", fontsize=8)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("BLS SNR")
    ax.set_title(title or f"Signal {det['planet']} periodogram")
    ax.legend(loc="upper left")
    return ax


def plot_fold(det, ax=None, title=None, bin_minutes=10):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.8))
    p, t0, dur = det["period"], det["t0"], det["duration"]
    half = max(3 * dur, 4 / 24)
    ph = phase_offset(det["t"], t0, p)
    sel = np.abs(ph) < half
    x, y = ph[sel] * 24, det["f"][sel]
    ax.plot(x, y, ".", color="0.75", ms=2, label="data")
    if len(x) > 0:
        edges = np.arange(-half * 24, half * 24 + 1e-9, bin_minutes / 60.0)
        idx = np.digitize(x, edges)
        bx = [x[idx == i].mean() for i in range(1, len(edges)) if (idx == i).sum() >= 3]
        by = [y[idx == i].mean() for i in range(1, len(edges)) if (idx == i).sum() >= 3]
        if bx:
            ax.plot(bx, by, "o", color="tab:red", ms=4, label=f"{bin_minutes}-min bins")
    ax.plot([-half * 24, -dur * 12, -dur * 12, dur * 12, dur * 12, half * 24],
            [1, 1, 1 - det["depth"], 1 - det["depth"], 1, 1], "b-", lw=1.2, label="BLS box")
    ax.set_xlabel("Hours from mid-transit")
    ax.set_ylabel("Normalized flux")
    ax.set_title(title or f"Signal {det['planet']}: P = {p:.4f} d, depth = {det['depth']*1e6:.0f} ppm")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_overview(t, f, table, ax=None, bin_minutes=60):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3.5))
    tb, fb, _ = bin_series(t, f, np.ones_like(f), bin_minutes)
    ax.plot(t, f, ".", color="0.8", ms=1)
    if len(tb) > 0:
        ax.plot(tb, fb, ".", color="0.3", ms=2)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    cands = table[table.status == "candidate"] if ("status" in table.columns and not table.empty) else pd.DataFrame()
    for r in cands.itertuples():
        n0 = int(np.ceil((t.min() - r.t0_btjd) / r.period_d))
        n1 = int(np.floor((t.max() - r.t0_btjd) / r.period_d))
        times = r.t0_btjd + r.period_d * np.arange(n0, n1 + 1)
        y = f.max() + 0.0004 * r.planet               # one marker row per signal
        ax.plot(times, np.full_like(times, y), "v", color=colors[(r.planet - 1) % 5],
                ms=4, label=f"P = {r.period_d:.3f} d")
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Normalized flux")
    ax.set_title("Detrended light curve with transits of the signals that passed all checks")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="lower right", ncol=min(4, len(labels)), fontsize=8)
    return ax


def save_outputs(table, details, t, f, name, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}_detections.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 3.5))
    plot_overview(t, f, table, ax)
    fig.tight_layout(); fig.savefig(out / f"{name}_overview.png", dpi=130); plt.close(fig)
    accepted_planets = set(table.loc[table.accepted, "planet"]) if ("accepted" in table.columns and not table.empty) else set()
    for d in details:
        fig, ax = plt.subplots(figsize=(12, 3.2))
        plot_periodogram(d, ax)
        fig.tight_layout(); fig.savefig(out / f"{name}_periodogram_{d['planet']}.png", dpi=130); plt.close(fig)
        if d["planet"] in accepted_planets:
            fig, ax = plt.subplots(figsize=(8, 3.8))
            plot_fold(d, ax)
            fig.tight_layout(); fig.savefig(out / f"{name}_fold_{d['planet']}.png", dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------
def load_input(csv=None, npz=None):
    if csv:
        df = pd.read_csv(csv)
        col_map = {str(c).strip().lower(): c for c in df.columns}
        if "time" not in col_map or "flux" not in col_map:
            raise ValueError(f"CSV must contain 'time' and 'flux' columns. Found: {list(df.columns)}")
        t_col = col_map["time"]
        f_col = col_map["flux"]
        e_col = col_map.get("flux_err", col_map.get("flux_error", col_map.get("err", None)))
        time_arr = df[t_col].values.astype(float)
        flux_arr = df[f_col].values.astype(float)
        err_arr = df[e_col].values.astype(float) if e_col is not None else None
        return time_arr, flux_arr, err_arr, Path(csv).stem
    if npz:
        with np.load(npz) as z:
            time_arr = z["time"].astype(float)
            flux_arr = z["flux"].astype(float)
            err_arr = z["flux_err"].astype(float) if "flux_err" in z else None
            return time_arr, flux_arr, err_arr, Path(npz).stem
    raise ValueError("Either csv or npz file must be provided")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--csv", help="full-cadence CSV (time, flux, flux_err) made by preprocess.py --full")
    src.add_argument("--npz", help="a processed star from data/processed")
    ap.add_argument("--max-planets", type=int, default=MAX_PLANETS)
    ap.add_argument("--min-snr", type=float, default=MIN_SNR)
    ap.add_argument("--period-max", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "detection"))
    args = ap.parse_args()

    csv_path = args.csv
    npz_path = args.npz

    if not csv_path and not npz_path:
        default_csv = ROOT / "data" / "full" / "processed_TIC_307210830.csv"
        if default_csv.exists():
            print(f"No input provided. Defaulting to benchmark light curve: {default_csv.relative_to(ROOT)}\n", flush=True)
            csv_path = str(default_csv)
        else:
            proc_files = list((ROOT / "data" / "processed").glob("*.npz"))
            if proc_files:
                sample_npz = proc_files[0]
                print(f"No input provided. Defaulting to processed star: {sample_npz.relative_to(ROOT)}\n", flush=True)
                npz_path = str(sample_npz)
            else:
                ap.print_help()
                print("\nError: Please provide --csv or --npz with a light curve file.")
                sys.exit(1)

    t, f, e, name = load_input(csv_path, npz_path)
    table, details = detect(t, f, e, max_planets=args.max_planets, min_snr=args.min_snr,
                            period_max=args.period_max)
    print()
    show = [c for c in ["planet", "period_d", "duration_h", "depth_ppm", "snr", "snr_red", "sde", "n_transits",
            "odd_even_sigma", "secondary_sigma", "flags", "status"] if c in table.columns]
    if not table.empty and show:
        print(table[show].round(3).to_string(index=False))
    else:
        print("No signals detected.")
    save_outputs(table, details, t, f, name, args.out)
    print(f"\nSaved table and plots to {args.out}")


if __name__ == "__main__":
    main()
