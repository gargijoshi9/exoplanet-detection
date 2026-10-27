"""
Build data/labels.csv: one row per star, with a class label and a train/test split.

Classes
-------
Transit          : TOI dispositions CP / KP / PC       (NASA Exoplanet Archive, "toi" table)
Eclipsing Binary : TESS-EBs catalog (Prsa et al. 2022) (VizieR J/ApJS/258/16, or your own CSV)
Other            : TOI dispositions FP / FA            (looked like a candidate, then rejected)

Rules that keep the labels clean
--------------------------------
* A star that has ANY planet-type TOI (CP/KP/PC) AND is in TESS-EBs is ambiguous -> dropped.
* A star in TESS-EBs (and not a planet TOI) is labelled Eclipsing Binary, even if it is also a TOI false positive.
* "Other" only keeps stars that are not planets and not EBs.
* ERA MATCHING: TESS-EBs only covers sectors 1-26, so Transit / Other stars are restricted to low TOI
  numbers (--max-toi, default 2000), which were alerted from the early sectors. Without this, the sector
  number alone predicts the class. preprocess.py then enforces it with --max-sector 26.
* The train/test split is done per STAR (about 80/20) and depends only on the star id and the seed,
  so you can raise --n-per-class later without moving any star between train and test.

Usage (run from the repo root):
    python src/build_labels.py                       # defaults: 150 per class
    python src/build_labels.py --n-per-class 100
    python src/build_labels.py --eb-csv my_ebs.csv   # if the VizieR download fails (needs a TIC column)
"""

import argparse
import hashlib
import io
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "labels.csv"

PLANET_DISP = {"CP", "KP", "PC"}
FALSE_POS_DISP = {"FP", "FA"}
TIC_COLUMN_NAMES = {"tic", "ticid", "tic_id", "tess_id", "tessid", "tid"}

TOI_QUERY = "select tid,toi,tfopwg_disp from toi"
MAX_TOI_DEFAULT = 2000
TOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    + urllib.parse.quote(TOI_QUERY)
    + "&format=csv"
)


# ----------------------------------------------------------------------------
# Downloading the two catalogs (needs internet)
# ----------------------------------------------------------------------------
def fetch_toi() -> pd.DataFrame:
    print("Downloading TOI table from the NASA Exoplanet Archive ...")
    with urllib.request.urlopen(TOI_URL, timeout=120) as resp:
        text = resp.read().decode("utf-8")
    toi = pd.read_csv(io.StringIO(text))
    print(f"  {len(toi)} TOI rows, columns: {list(toi.columns)}")
    return toi


def find_tic_column(columns):
    for c in columns:
        if str(c).lower() in TIC_COLUMN_NAMES:
            return c
    return None


def fetch_ebs(eb_csv=None) -> set:
    if eb_csv:
        print(f"Reading eclipsing binaries from {eb_csv} ...")
        df = pd.read_csv(eb_csv)
        col = find_tic_column(df.columns)
        if col is None:
            raise SystemExit(f"No TIC column found in {eb_csv}. Columns: {list(df.columns)}")
        return set(df[col].dropna().astype("int64"))

    print("Downloading TESS-EBs catalog from VizieR (J/ApJS/258/16) ...")
    from astroquery.vizier import Vizier

    tables = Vizier(columns=["*"], row_limit=-1).get_catalogs("J/ApJS/258/16")
    for t in tables:
        col = find_tic_column(t.colnames)
        if col is not None:
            # Go through text -> numeric so masked / "N/A" / blank entries become NaN instead of crashing
            raw = pd.Series([str(x).strip() for x in t[col]])
            tics = pd.to_numeric(raw, errors="coerce").dropna().astype("int64")
            print(f"  table '{t.meta.get('name', '?')}': {len(tics)} rows, TIC column '{col}'")
            return set(tics)

    names = [(t.meta.get("name", "?"), t.colnames) for t in tables]
    raise SystemExit(
        "Could not find a TIC column in the VizieR tables.\n"
        f"Tables and columns: {names}\n"
        "Download the catalog from tessEBs.villanova.edu or MAST, save it as a CSV with a TIC column, "
        "and re-run with --eb-csv yourfile.csv"
    )


# ----------------------------------------------------------------------------
# Pure logic (no internet) - easy to test
# ----------------------------------------------------------------------------
def assign_labels(toi: pd.DataFrame, eb_tics: set, max_toi=None):
    """Return (labels DataFrame with columns tic_id,label,source) and a dict of counts.

    max_toi: only keep Transit / Other stars whose lowest TOI number is below this (era matching).
    """
    toi = toi.dropna(subset=["tid"]).copy()
    toi["tid"] = toi["tid"].astype("int64")
    toi["toi_num"] = np.floor(pd.to_numeric(toi["toi"], errors="coerce")).fillna(10**9)
    disp = toi["tfopwg_disp"].fillna("").astype(str).str.upper().str.strip()

    def tics(mask):
        sub = toi.loc[mask]
        if max_toi:
            lowest = sub.groupby("tid")["toi_num"].min()     # earliest-alerted TOI of each star
            return set(lowest[lowest < max_toi].index)
        return set(sub["tid"])

    planet_all = set(toi.loc[disp.isin(PLANET_DISP), "tid"])
    false_all = set(toi.loc[disp.isin(FALSE_POS_DISP), "tid"])
    planet = tics(disp.isin(PLANET_DISP))
    false_pos = tics(disp.isin(FALSE_POS_DISP))
    eb = {int(x) for x in eb_tics}

    ambiguous = planet_all & eb
    transit = planet - eb
    ebs = eb - planet_all
    other = false_pos - planet_all - eb       # never a star that has any planet-type TOI

    rows = (
        [(t, "Transit", "TOI CP/KP/PC") for t in sorted(transit)]
        + [(t, "Eclipsing Binary", "TESS-EBs") for t in sorted(ebs)]
        + [(t, "Other", "TOI FP/FA") for t in sorted(other)]
    )
    labels = pd.DataFrame(rows, columns=["tic_id", "label", "source"])
    counts = {
        "Transit": len(transit),
        "Eclipsing Binary": len(ebs),
        "Other": len(other),
        "dropped_ambiguous": len(ambiguous),
    }
    return labels, counts


def star_hash(tic_id, seed) -> float:
    """Deterministic number in [0, 1) for a star, used for the train/test split."""
    h = hashlib.md5(f"{seed}-{int(tic_id)}".encode()).hexdigest()
    return int(h[:8], 16) / 16**8


def sample_and_split(labels: pd.DataFrame, n_per_class: int, oversample: float,
                     test_frac: float, seed: int, existing_path: Path | None = None) -> pd.DataFrame:
    """Sample stars per class. If existing_path exists, keeps existing stars and adds new candidates to reach target."""
    rng = np.random.default_rng(seed)
    n_take = int(round(n_per_class * oversample))
    
    existing_df = None
    if existing_path and Path(existing_path).exists():
        try:
            existing_df = pd.read_csv(existing_path)
            print(f"Preserving {len(existing_df)} existing stars from {existing_path} ...")
        except Exception:
            existing_df = None

    parts = []
    for label, grp in labels.groupby("label"):
        existing_class_df = None
        existing_tics = set()
        if existing_df is not None and "label" in existing_df.columns:
            existing_class_df = existing_df[existing_df["label"] == label].copy()
            existing_tics = set(existing_class_df["tic_id"].astype("int64"))

        # Available candidates not already chosen
        grp_new = grp[~grp["tic_id"].isin(existing_tics)].copy()
        needed = max(0, n_take - len(existing_tics))

        if needed > 0 and len(grp_new) > 0:
            idx = rng.permutation(len(grp_new))[:needed]
            sampled_new = grp_new.iloc[idx].copy()
            sampled_new["split"] = [
                "test" if star_hash(t, seed) < test_frac else "train" for t in sampled_new["tic_id"]
            ]
        else:
            sampled_new = pd.DataFrame(columns=grp.columns.tolist() + ["split"])

        if existing_class_df is not None and len(existing_class_df) > 0:
            combined = pd.concat([existing_class_df, sampled_new], ignore_index=True)
        else:
            combined = sampled_new

        parts.append(combined)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["label", "tic_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=50, help="target stars per class (default 50)")
    ap.add_argument("--oversample", type=float, default=1.6,
                    help="take this factor more stars, since some have no 2-min data (default 1.6)")
    ap.add_argument("--max-toi", type=int, default=MAX_TOI_DEFAULT,
                    help="only Transit/Other stars with a TOI number below this, to match the EB era "
                         "(default 2000; 0 = no filter)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eb-csv", default=None, help="CSV with a TIC column, instead of downloading from VizieR")
    ap.add_argument("--fresh", action="store_true", help="ignore existing labels.csv and resample from scratch")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    toi = fetch_toi()
    eb_tics = fetch_ebs(args.eb_csv)

    labels, counts = assign_labels(toi, eb_tics, max_toi=args.max_toi or None)
    print(f"\nStars available per class (before sampling, TOI number < {args.max_toi or 'any'}):")
    for k, v in counts.items():
        print(f"  {k:<20} {v}")

    existing_to_check = None if args.fresh else Path(args.out)
    chosen = sample_and_split(labels, args.n_per_class, args.oversample, args.test_frac, args.seed, existing_to_check)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(args.out, index=False)

    print(f"\nWrote {len(chosen)} stars to {args.out}")
    print(chosen.groupby(["label", "split"]).size().unstack(fill_value=0))
    short = [k for k in ("Transit", "Eclipsing Binary", "Other")
             if counts[k] < args.n_per_class]
    if short:
        print(f"\nNote: fewer than {args.n_per_class} stars exist for: {', '.join(short)}")


if __name__ == "__main__":
    main()