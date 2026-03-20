from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _parse_inval(text: str) -> list[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError("inval must contain exactly 4 comma-separated integers.")
    return vals


def _spectral_columns(df: pd.DataFrame, wl_start: int, wl_end: int) -> list[str]:
    cols = [str(w) for w in range(wl_start, wl_end + 1) if str(w) in df.columns]
    if not cols:
        raise ValueError(f"No spectral columns found in range {wl_start}-{wl_end}.")
    return cols


def _safe_savgol(arr: np.ndarray, window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    n_bands = arr.shape[1]
    wl = min(window_length, n_bands if n_bands % 2 == 1 else n_bands - 1)
    if wl < 3:
        return arr
    po = min(polyorder, wl - 1)
    return savgol_filter(arr, wl, po, deriv=deriv, axis=1)


def _filter_segment(df_seg: pd.DataFrame, polyorder: int = 1, deriv: bool = False, window: int = 65) -> pd.DataFrame:
    arr = df_seg.to_numpy(dtype=np.float32)
    deriv_n = 1 if deriv else 0
    out = _safe_savgol(arr, window_length=window, polyorder=polyorder, deriv=deriv_n)
    return pd.DataFrame(out, columns=df_seg.columns, index=df_seg.index)


def _feature_preparation_like_original(
    spectra_df: pd.DataFrame,
    inval: list[int],
    frmax: int = 2451,
    polyorder: int = 1,
    deriv: bool = False,
    window: int = 65,
) -> pd.DataFrame:
    """Mirror multi-traitretrieval feature_preparation() behavior."""
    other = spectra_df.copy()
    other.columns = other.columns.astype(int)

    # Original behavior: mark invalid reflectance then fill.
    other[other < 0] = np.nan
    other[other > 1] = np.nan

    # Original behavior: (ffill + bfill) / 2 then interpolation across wavelength axis.
    other = (other.ffill() + other.bfill()) / 2.0
    other = other.interpolate(method="linear", axis=1, limit_direction="both")

    # Water absorption ranges to remove before filtering.
    wt_ab = (
        list(range(inval[0], inval[1]))
        + list(range(inval[2], inval[3]))
        + list(range(2451, 2501))
    )
    wt_ab = [w for w in wt_ab if w in other.columns]
    features_no_wtab = other.drop(columns=wt_ab)

    # Segment-wise filtering as in original code.
    seg1_cols = [c for c in features_no_wtab.columns if c <= inval[0] - 1]
    seg2_cols = [c for c in features_no_wtab.columns if inval[1] <= c <= inval[2] - 1]
    seg3_cols = [c for c in features_no_wtab.columns if inval[3] <= c <= frmax]

    fr1 = _filter_segment(features_no_wtab.loc[:, seg1_cols], polyorder=polyorder, deriv=deriv, window=window)
    fr2 = _filter_segment(features_no_wtab.loc[:, seg2_cols], polyorder=polyorder, deriv=deriv, window=window)
    fr3 = _filter_segment(features_no_wtab.loc[:, seg3_cols], polyorder=polyorder, deriv=deriv, window=window)

    inter = pd.concat([fr1, fr2, fr3], axis=1, join="inner")
    inter[inter < 0] = 0
    inter = inter.reindex(sorted(inter.columns), axis=1)
    inter.columns = inter.columns.astype(str)
    return inter


def process_raw_to_processed(
    raw_csv: Path,
    output_1721: Path,
    output_1522: Path,
    summary_csv: Path,
    wl_start: int = 400,
    wl_end: int = 2450,
    inval_1721: list[int] | None = None,
    inval_1522: list[int] | None = None,
    polyorder: int = 1,
    deriv: bool = False,
    window: int = 65,
) -> None:
    if inval_1721 is None:
        inval_1721 = [1351, 1431, 1801, 2051]
    if inval_1522 is None:
        inval_1522 = [1251, 1530, 1801, 2051]

    df = pd.read_csv(raw_csv, low_memory=False)
    wl_cols = _spectral_columns(df, wl_start=wl_start, wl_end=wl_end)
    meta_cols = [c for c in df.columns if c not in wl_cols]

    spectra = df[wl_cols].copy()
    spectra.columns = spectra.columns.astype(str)

    inter_1721 = _feature_preparation_like_original(
        spectra_df=spectra,
        inval=inval_1721,
        frmax=wl_end + 1,
        polyorder=polyorder,
        deriv=deriv,
        window=window,
    )
    inter_1522 = _feature_preparation_like_original(
        spectra_df=spectra,
        inval=inval_1522,
        frmax=wl_end + 1,
        polyorder=polyorder,
        deriv=deriv,
        window=window,
    )

    out_1721_df = pd.concat([df[meta_cols].reset_index(drop=True), inter_1721.reset_index(drop=True)], axis=1)
    out_1522_df = pd.concat([df[meta_cols].reset_index(drop=True), inter_1522.reset_index(drop=True)], axis=1)

    output_1721.parent.mkdir(parents=True, exist_ok=True)
    output_1522.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    out_1721_df.to_csv(output_1721, index=False)
    out_1522_df.to_csv(output_1522, index=False)

    summary = pd.DataFrame(
        [
            {"dataset": "1721", "n_rows": len(out_1721_df), "n_spectral_bands": inter_1721.shape[1], "output_csv": str(output_1721)},
            {"dataset": "1522", "n_rows": len(out_1522_df), "n_spectral_bands": inter_1522.shape[1], "output_csv": str(output_1522)},
        ]
    )
    summary.to_csv(summary_csv, index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw DB_50 CSV to processed 1721/1522 spectra using original SG + water-band removal logic.")
    parser.add_argument("--raw_csv", type=Path, default=Path("data/raw/DB_50_Meta_EC_teja_with_locations.csv"))
    parser.add_argument("--output_1721", type=Path, default=Path("data/processed/DB_50_Meta_EC_with_locations_clean1721_from_raw.csv"))
    parser.add_argument("--output_1522", type=Path, default=Path("data/processed/DB_50_Meta_EC_with_locations_clean1522_from_raw.csv"))
    parser.add_argument("--summary_csv", type=Path, default=Path("results/processing/raw_to_processed_summary.csv"))
    parser.add_argument("--wl_start", type=int, default=400)
    parser.add_argument("--wl_end", type=int, default=2450)
    parser.add_argument("--inval_1721", type=str, default="1351,1431,1801,2051")
    parser.add_argument("--inval_1522", type=str, default="1251,1530,1801,2051")
    parser.add_argument("--polyorder", type=int, default=1)
    parser.add_argument("--window", type=int, default=65)
    parser.add_argument("--deriv", action="store_true", help="Use first derivative SG filter (deriv=1).")
    args = parser.parse_args()

    process_raw_to_processed(
        raw_csv=args.raw_csv,
        output_1721=args.output_1721,
        output_1522=args.output_1522,
        summary_csv=args.summary_csv,
        wl_start=args.wl_start,
        wl_end=args.wl_end,
        inval_1721=_parse_inval(args.inval_1721),
        inval_1522=_parse_inval(args.inval_1522),
        polyorder=args.polyorder,
        deriv=bool(args.deriv),
        window=args.window,
    )


if __name__ == "__main__":
    main()
