from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class SceneTifBundle:
    spectra_1522: np.ndarray
    meta_df: pd.DataFrame
    n_rows: int
    n_cols: int
    transform: object | None
    crs: object | None


def _load_sensor_bands(bands_csv: Path) -> np.ndarray:
    bdf = pd.read_csv(bands_csv)
    cols_lower = {c.lower(): c for c in bdf.columns}
    pick = None
    for key in ("bands", "band", "wavelength", "wavelengths"):
        if key in cols_lower:
            pick = cols_lower[key]
            break
    if pick is None:
        # Fallback: first column that can be parsed numerically.
        for c in bdf.columns:
            vals = pd.to_numeric(bdf[c], errors="coerce")
            if vals.notna().sum() > 0:
                pick = c
                break
    if pick is None:
        raise ValueError(
            f"Could not find a sensor-band column in {bands_csv}. "
            "Expected one of: bands, band, wavelength, wavelengths."
        )
    bands = pd.to_numeric(bdf[pick], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if bands.size == 0:
        raise ValueError(f"No valid numeric band values found in column '{pick}' of {bands_csv}.")
    return bands


def _safe_savgol(arr: np.ndarray, window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    n_bands = int(arr.shape[1])
    wl = min(int(window_length), n_bands if n_bands % 2 == 1 else n_bands - 1)
    if wl < 3:
        return arr
    po = min(int(polyorder), wl - 1)
    return savgol_filter(arr, wl, po, deriv=deriv, axis=1)


def _filter_segment(df_seg: pd.DataFrame, polyorder: int = 1, deriv: bool = False, window: int = 65) -> pd.DataFrame:
    arr = df_seg.to_numpy(dtype=np.float32)
    deriv_n = 1 if deriv else 0
    out = _safe_savgol(arr, window_length=window, polyorder=polyorder, deriv=deriv_n)
    return pd.DataFrame(out, columns=df_seg.columns, index=df_seg.index)


def feature_preparation_like_original(
    features: pd.DataFrame,
    inval: list[int] | tuple[int, int, int, int] = (1251, 1530, 1801, 2051),
    frmax: int = 2451,
    polyorder: int = 1,
    deriv: bool = False,
    window: int = 65,
) -> pd.DataFrame:
    """Mirror original feature_preparation() from legacy uncertainty project."""
    other = features.copy()
    other.columns = other.columns.astype(int)
    other[other < 0] = np.nan
    other[other > 1] = np.nan
    other = (other.ffill() + other.bfill()) / 2.0
    other = other.interpolate(method="linear", axis=1, limit_direction="both")

    wt_ab = list(range(inval[0], inval[1])) + list(range(inval[2], inval[3])) + list(range(2451, 2501))
    wt_ab = [w for w in wt_ab if w in other.columns]
    features_no_wtab = other.drop(columns=wt_ab)

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


def _to_400_2500_grid(df_sensor: pd.DataFrame) -> pd.DataFrame:
    target = np.arange(400, 2501, dtype=np.int32)
    cur = np.array([float(c) for c in df_sensor.columns], dtype=np.float32)
    union = np.unique(np.concatenate([cur, target.astype(np.float32)]))
    interp = df_sensor.reindex(columns=union).interpolate(method="linear", axis=1, limit_direction="both")
    out = interp.reindex(columns=target.astype(np.float32))
    out.columns = target.astype(int)
    return out


def preprocess_scene_tif_to_1522(
    tif_path: Path,
    bands_csv: Path,
    divide_by: float = 10000.0,
    corrupted_pixel_offset: float = 10.0,
    inval_1522: list[int] | tuple[int, int, int, int] = (1251, 1530, 1801, 2051),
    polyorder: int = 1,
    deriv: bool = False,
    window: int = 65,
) -> SceneTifBundle:
    try:
        import rasterio
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "rasterio is required for TIFF scene inference. Install with one of:\n"
            "  1) poetry add rasterio\n"
            "  2) conda install -n multi-trait -c conda-forge rasterio"
        ) from exc

    bands = _load_sensor_bands(bands_csv)
    with rasterio.open(tif_path) as src:
        array = src.read()  # (bands, H, W)
        h, w = int(array.shape[1]), int(array.shape[2])
        px = array.reshape(array.shape[0], -1).T  # (Npix, bands)
        if px.shape[1] != len(bands):
            raise ValueError(
                f"Band count mismatch: TIFF has {px.shape[1]} bands, "
                f"bands CSV has {len(bands)}."
            )

        df = pd.DataFrame(px, columns=bands.tolist())
        if corrupted_pixel_offset is not None:
            thr = float(df.quantile(0.01).min()) + float(corrupted_pixel_offset)
            df[df < thr] = np.nan
        idx_null = df.isna().all(axis=1).to_numpy()

        veg = df.loc[~idx_null].copy()
        veg_grid = _to_400_2500_grid(veg)
        veg_grid = veg_grid / float(divide_by)

        inter = feature_preparation_like_original(
            features=veg_grid,
            inval=inval_1522,
            frmax=2451,
            polyorder=polyorder,
            deriv=deriv,
            window=window,
        )
        inter = inter.loc[:, ~inter.columns.duplicated()].copy()
        spectra_1522 = inter.loc[:, "400":"2450"].to_numpy(dtype=np.float32)

        all_pix = pd.DataFrame(
            {
                "pixel_index": np.arange(h * w, dtype=np.int64),
                "row": np.repeat(np.arange(h, dtype=np.int32), w),
                "col": np.tile(np.arange(w, dtype=np.int32), h),
                "is_valid_pixel": ~idx_null,
            }
        )
        if src.transform is not None:
            rows = all_pix["row"].to_numpy()
            cols = all_pix["col"].to_numpy()
            xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
            all_pix["x"] = np.asarray(xs, dtype=np.float64)
            all_pix["y"] = np.asarray(ys, dtype=np.float64)
            if src.crs is not None and getattr(src.crs, "is_geographic", False):
                all_pix["lon"] = all_pix["x"]
                all_pix["lat"] = all_pix["y"]

    return SceneTifBundle(
        spectra_1522=spectra_1522,
        meta_df=all_pix,
        n_rows=h,
        n_cols=w,
        transform=src.transform,
        crs=src.crs,
    )
