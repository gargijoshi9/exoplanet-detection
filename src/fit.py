import numpy as np
import pandas as pd
from pathlib import Path
import batman
from scipy.optimize import minimize
from scipy.signal import savgol_filter

# Project paths
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
FULL_DIR = ROOT / "data" / "full"
DET_DIR = ROOT / "results" / "detection"
FIT_DIR = ROOT / "results" / "fitting"

# Constants for flattening (matching preprocess.py)
WINDOW_LENGTH = 301
POLYORDER = 2

def mask_and_flatten(time, flux_norm, period, t0, duration):
    """
    Masks transits and re-flattens the light curve to remove stellar trends
    without biasing the transit depth.
    """
    # Calculate phase
    phase = (time - t0) % period

    # Transit occurs when phase is near 0 or near period
    # duration is in hours, convert to days
    dur_days = duration / 24.0
    mask = (phase < dur_days / 2.0) | (phase > (period - dur_days / 2.0))

    flux_filled = flux_norm.copy()
    flux_filled[mask] = np.nanmedian(flux_norm)

    # Ensure window_length is odd and <= len(flux_filled) - 1
    wl = min(WINDOW_LENGTH, len(flux_filled) - 1)
    if wl % 2 == 0:
        wl -= 1
    if wl < 7:
        return flux_norm

    baseline = savgol_filter(flux_filled, wl, POLYORDER)
    return flux_norm / baseline


def batman_model(params, time, period, limb_darkening=(0.3, 0.2)):
    """
    Generates a transit light curve using batman.
    params: [rp_rs, b, a_rs, t0]
    """
    rp_rs, b, a_rs, t0 = params
    p = batman.TransitParams()
    p.t0 = float(t0)
    p.per = float(period)
    p.rp = float(max(1e-5, rp_rs))
    p.a = float(max(1.1, a_rs))
    cos_i = np.clip(b / p.a, 0.0, 1.0)
    p.inc = float(np.degrees(np.arccos(cos_i)))
    p.ecc = 0.0
    p.w = 90.0
    p.u = list(limb_darkening)
    p.limb_dark = "quadratic"

    m = batman.TransitModel(p, np.asarray(time, dtype=float))
    return m.light_curve(p)


def fit_transit(time, flux, flux_err, period, initial_guesses, duration_h=None):
    """
    Fits a physical batman transit model to the data using Nelder-Mead optimization.
    initial_guesses: [rp_rs, b, a_rs, t0]
    """
    u1, u2 = 0.3, 0.2
    rp_guess, b_guess, a_rs_guess, t0_guess = initial_guesses

    dur_d = (duration_h / 24.0) if duration_h is not None else 3.0 / 24.0

    # Fold to phase offset in [-period/2, period/2)
    phase_offset = ((time - t0_guess + 0.5 * period) % period) - 0.5 * period

    # Restrict fitting window to points around transits (+- 2.5 transit durations)
    # to avoid out-of-transit noise while ensuring fast and robust convergence
    window_half = max(2.5 * dur_d, 0.04 * period)
    in_window = np.abs(phase_offset) < window_half
    if np.sum(in_window) < 15:
        in_window = np.ones_like(time, dtype=bool)

    t_fit = phase_offset[in_window]
    f_fit = flux[in_window]
    ferr_fit = flux_err[in_window] if (flux_err is not None and not np.all(np.isnan(flux_err))) else None

    # Sort phase for numerical stability
    sort_idx = np.argsort(t_fit)
    t_fit = t_fit[sort_idx]
    f_fit = f_fit[sort_idx]
    if ferr_fit is not None:
        ferr_fit = ferr_fit[sort_idx]

    p = batman.TransitParams()
    p.t0 = 0.0  # Center is 0 on the folded phase
    p.per = float(period)
    p.rp = float(max(1e-4, rp_guess))
    p.a = float(max(1.5, a_rs_guess))
    cos_i = np.clip(b_guess / p.a, 0.0, 1.0)
    p.inc = float(np.degrees(np.arccos(cos_i)))
    p.ecc = 0.0
    p.w = 90.0
    p.u = [u1, u2]
    p.limb_dark = "quadratic"

    m = batman.TransitModel(p, t_fit)

    def objective(params):
        rp_rs, b, a_rs, dt0 = params
        if rp_rs <= 1e-4 or rp_rs > 0.5 or a_rs <= 1.0 or b < 0.0 or b > 1.5:
            return 1e9
        p.rp = float(rp_rs)
        p.a = float(a_rs)
        cos_i = np.clip(b / p.a, 0.0, 1.0)
        p.inc = float(np.degrees(np.arccos(cos_i)))
        p.t0 = float(dt0)

        try:
            model_flux = m.light_curve(p)
        except Exception:
            return 1e9

        if ferr_fit is not None and not np.all(np.isnan(ferr_fit)):
            return np.sum(((f_fit - model_flux) / ferr_fit)**2)
        else:
            return np.sum((f_fit - model_flux)**2)

    x0 = [rp_guess, b_guess, a_rs_guess, 0.0]
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500})

    best_rp, best_b, best_a, best_dt0 = res.x
    fitted_t0 = t0_guess + best_dt0
    best_params = [float(best_rp), float(best_b), float(best_a), float(fitted_t0)]
    return best_params, bool(res.success)


def run_fitting_pipeline(tic, planet_idx=0, initial_params=None):
    """
    Full pipeline: Load -> Mask -> Re-flatten -> Fit -> Save.
    Robustly works for any star in the dataset.
    """
    from astropy.timeseries import BoxLeastSquares

    # 1. Load Data
    full_path = FULL_DIR / f"processed_TIC_{tic}.csv"
    if full_path.exists():
        df = pd.read_csv(full_path)
    else:
        npz_path = PROC_DIR / f"TIC_{tic}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"No processed light curve found for TIC {tic}")
        with np.load(npz_path) as z:
            df = pd.DataFrame({k: z[k] for k in ("time", "flux", "flux_err", "flux_norm")})

    t_arr = df['time'].values
    f_arr = df['flux'].values
    e_arr = df['flux_err'].values if 'flux_err' in df else None

    # 2. Determine initial detection parameters (period, t0, duration, depth)
    if initial_params is not None:
        period = float(initial_params.get('period_d', initial_params.get('period', 3.0)))
        duration = float(initial_params.get('duration_h', initial_params.get('duration', 2.0)))
        t0 = initial_params.get('t0_btjd', initial_params.get('t0'))
        depth = float(initial_params.get('depth_ppm', initial_params.get('depth', 1000.0)))
        if depth > 1.0:
            depth = depth * 1e-6
        if t0 is None or pd.isna(t0):
            bls = BoxLeastSquares(t_arr, f_arr, dy=e_arr)
            res = bls.power([period], [duration / 24.0])
            t0 = float(res.transit_time[0])
        else:
            t0 = float(t0)
    else:
        det_path = DET_DIR / f"TIC_{tic}_detections.csv"
        bls_feat_path = ROOT / "data" / "features" / "bls_features.csv"

        if det_path.exists():
            dets = pd.read_csv(det_path)
            if planet_idx >= len(dets):
                raise IndexError(f"Planet index {planet_idx} out of range for TIC {tic}")
            row = dets.iloc[planet_idx]
            period = float(row['period_d'])
            t0 = float(row['t0_btjd'] if 't0_btjd' in row else row['t0'])
            duration = float(row['duration_h'])
            depth = float(row['depth_ppm']) * 1e-6
        elif bls_feat_path.exists():
            bls_df = pd.read_csv(bls_feat_path)
            matches = bls_df[bls_df['tic_id'].astype(str) == str(tic)]
            if not matches.empty:
                row = matches.iloc[0]
                period = float(row['period_d'])
                duration = float(row['duration_h'])
                depth = float(row['depth_ppm']) * 1e-6
                bls = BoxLeastSquares(t_arr, f_arr, dy=e_arr)
                res = bls.power([period], [duration / 24.0])
                t0 = float(res.transit_time[0])
            else:
                bls = BoxLeastSquares(t_arr, f_arr, dy=e_arr)
                baseline = t_arr.max() - t_arr.min()
                periods = np.exp(np.linspace(np.log(0.5), np.log(min(20.0, baseline / 2.0)), 1500))
                durations = np.array([0.05, 0.1, 0.15])
                res = bls.power(periods, durations, objective="snr")
                i_best = int(np.nanargmax(res.power))
                period = float(res.period[i_best])
                duration = float(res.duration[i_best] * 24.0)
                t0 = float(res.transit_time[i_best])
                depth = float(res.depth[i_best])
        else:
            raise FileNotFoundError(f"No detection catalog found for TIC {tic}")

    # 3. Mask and Re-flatten
    flux_input = df['flux_norm'].values if 'flux_norm' in df else df['flux'].values
    flux_refit = mask_and_flatten(t_arr, flux_input, period, t0, duration)

    # 4. Fit
    rp_rs_guess = np.sqrt(max(1e-6, depth))
    b_guess = 0.2
    a_rs_guess = 15.0  # Keplerian reasonable starting point
    t0_guess = t0

    initial_guesses = [rp_rs_guess, b_guess, a_rs_guess, t0_guess]
    best_params, success = fit_transit(
        t_arr, flux_refit, e_arr, period, initial_guesses, duration_h=duration
    )

    # 5. Save Results
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame([{
        'rp_rs': best_params[0],
        'b': best_params[1],
        'a_rs': best_params[2],
        't0': best_params[3],
        'period': period,
        'planet_idx': planet_idx,
        'success': success,
    }])
    res_df.to_csv(FIT_DIR / f"TIC_{tic}_fit_params.csv", index=False)

    return best_params, success


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fit transit parameters for a TESS star.")
    ap.add_argument("--tic", type=int, required=True, help="TIC ID of the star")
    ap.add_argument("--planet", type=int, default=0, help="Planet index (default 0)")
    args = ap.parse_args()

    try:
        params, success = run_fitting_pipeline(args.tic, planet_idx=args.planet)
        print(f"Fitting for TIC {args.tic} Planet {args.planet+1}:")
        print(f"  Success: {success}")
        print(f"  Params: Rp/Rs={params[0]:.5f}, b={params[1]:.3f}, a/Rs={params[2]:.3f}, T0={params[3]:.5f}")
    except Exception as e:
        print(f"Error fitting TIC {args.tic}: {e}")


if __name__ == "__main__":
    main()


