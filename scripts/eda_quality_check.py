from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _infer_columns(df: pd.DataFrame, n_traits: int) -> tuple[list[str], list[str]]:
    wl_cols = [c for c in df.columns if c.isdigit() and 350 <= int(c) <= 2600]
    if not wl_cols:
        raise ValueError("No numeric wavelength columns found. Expected columns like '400'...'2450'.")
    wl_cols = sorted(wl_cols, key=lambda x: int(x))

    trait_cols = [c for c in df.columns if c not in wl_cols and c.lower() not in {"dataset", "lat", "lon", "id"}]
    trait_cols = trait_cols[:n_traits]
    return wl_cols, trait_cols


def _region_mean(df: pd.DataFrame, wl_cols: list[str], start: int, end: int) -> np.ndarray:
    sub = [c for c in wl_cols if start <= int(c) <= end]
    if not sub:
        return np.full((len(df),), np.nan, dtype=np.float32)
    return df[sub].to_numpy(dtype=np.float32).mean(axis=1)


def _custom_red_nir_filter(df_spectra: pd.DataFrame) -> pd.Series:
    """Implements user-specified filter:
    drop when reflectance(1300) > reflectance(1000) and reflectance(750) > reflectance(1000).
    """
    required = ["750", "1000", "1300"]
    if not all(c in df_spectra.columns for c in required):
        return pd.Series(False, index=df_spectra.index)
    red_ed = df_spectra.loc[:, "750"]
    red_end = df_spectra.loc[:, "1300"]
    red1000_ = df_spectra.loc[:, "1000"]
    return (red_end > red1000_) & (red_ed > red1000_)


def run_eda(data_path: Path, out_dir: Path, n_traits: int, apply_red_nir_filter: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path, low_memory=False)

    wl_cols, trait_cols = _infer_columns(df, n_traits)
    x = df[wl_cols]
    y = df[trait_cols]

    all_traits_nan = y.isna().all(axis=1)
    all_spectra_nan = x.isna().all(axis=1)
    dup_spectra = x.duplicated(keep=False)
    dup_traits = y.duplicated(keep=False)
    dup_spectra_and_traits = dup_spectra & dup_traits

    vis = _region_mean(df, wl_cols, 400, 700)
    nir = _region_mean(df, wl_cols, 760, 1300)
    swir1 = _region_mean(df, wl_cols, 1550, 1750)
    swir2 = _region_mean(df, wl_cols, 2080, 2350)
    bad_shape = ~((vis < nir) & (swir1 > swir2))
    custom_bad_red = _custom_red_nir_filter(x)

    report = pd.DataFrame(
        {
            "row_index": np.arange(len(df)),
            "all_traits_nan": all_traits_nan,
            "all_spectra_nan": all_spectra_nan,
            "dup_spectra": dup_spectra,
            "dup_traits": dup_traits,
            "dup_spectra_and_traits": dup_spectra_and_traits,
            "bad_spectral_shape": bad_shape,
            "bad_red_nir_shape": custom_bad_red,
        }
    )
    # Requested strict drop policy:
    # 1) all trait values are NaN
    # 2) both spectra and traits are duplicates
    report["drop_sample"] = report["all_traits_nan"] | report["dup_spectra_and_traits"]
    if apply_red_nir_filter:
        report["drop_sample"] = report["drop_sample"] | report["bad_red_nir_shape"]
    report.to_csv(out_dir / "quality_flags.csv", index=False)

    kept_idx = ~report["drop_sample"]
    cleaned = df.loc[kept_idx].reset_index(drop=True)
    cleaned.to_csv(out_dir / "dataset_cleaned.csv", index=False)
    dropped = report.loc[report["drop_sample"]].copy()
    dropped.to_csv(out_dir / "dropped_rows.csv", index=False)

    # Keep bad spectra for manual review, do not drop them automatically.
    bad_candidates = report.loc[report["bad_spectral_shape"]].copy()
    bad_candidates.to_csv(out_dir / "bad_spectra_candidates.csv", index=False)
    red_nir_candidates = report.loc[report["bad_red_nir_shape"]].copy()
    red_nir_candidates.to_csv(out_dir / "bad_red_nir_candidates.csv", index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_rows_after_filter": int(len(cleaned)),
        "n_rows_dropped": int(dropped.shape[0]),
        "all_traits_nan": int(all_traits_nan.sum()),
        "all_spectra_nan": int(all_spectra_nan.sum()),
        "duplicate_spectra": int(dup_spectra.sum()),
        "duplicate_traits": int(dup_traits.sum()),
        "duplicate_spectra_and_traits": int(dup_spectra_and_traits.sum()),
        "bad_spectral_shape": int(bad_shape.sum()),
        "bad_red_nir_shape": int(custom_bad_red.sum()),
        "drop_uses_red_nir_filter": int(bool(apply_red_nir_filter)),
        "dropped_total": int(dropped.shape[0]),
    }
    pd.Series(summary).to_csv(out_dir / "summary.csv", header=["value"])

    wl = np.array([int(c) for c in wl_cols], dtype=np.int32)

    if plt is not None:
        # Plot random good vs bad spectra (visual inspection only).
        plt.figure(figsize=(11, 5))
        good_ix = np.where(~bad_shape.to_numpy())[0][:20]
        bad_ix = np.where(bad_shape.to_numpy())[0][:20]
        for i in good_ix:
            plt.plot(wl, x.iloc[i].to_numpy(dtype=np.float32), color="tab:green", alpha=0.2)
        for i in bad_ix:
            plt.plot(wl, x.iloc[i].to_numpy(dtype=np.float32), color="tab:red", alpha=0.2)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title("Spectral plausibility check (green=plausible, red=flagged)")
        plt.tight_layout()
        plt.savefig(out_dir / "spectral_plausibility.png", dpi=150)
        plt.close()

        # Region means scatter for easier review of flagged spectra.
        plt.figure(figsize=(8, 6))
        plt.scatter(vis[~bad_shape], nir[~bad_shape], s=8, alpha=0.3, label="plausible")
        plt.scatter(vis[bad_shape], nir[bad_shape], s=10, alpha=0.4, label="flagged")
        plt.xlabel("VIS mean (400-700)")
        plt.ylabel("NIR mean (760-1300)")
        plt.title("VIS vs NIR region means")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "vis_nir_scatter_flagged.png", dpi=150)
        plt.close()

        # All spectra plausibility map (full dataset), color by filter decisions.
        status = np.where(report["drop_sample"].to_numpy(), "dropped", "kept")
        status = np.where(report["bad_spectral_shape"].to_numpy() & (status == "kept"), "flagged_review", status)
        plt.figure(figsize=(9, 7))
        for label, alpha, size in [("kept", 0.15, 8), ("flagged_review", 0.25, 10), ("dropped", 0.35, 12)]:
            m = status == label
            if np.any(m):
                plt.scatter(vis[m], nir[m], s=size, alpha=alpha, label=label)
        plt.xlabel("VIS mean (400-700)")
        plt.ylabel("NIR mean (760-1300)")
        plt.title("All spectra plausibility overview")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "all_spectra_plausibility_overview.png", dpi=170)
        plt.close()

        # Spectra where trait rows contain at least one NaN
        trait_nan_ix = np.where(y.isna().any(axis=1).to_numpy())[0][:30]
        if len(trait_nan_ix) > 0:
            plt.figure(figsize=(11, 5))
            for i in trait_nan_ix:
                plt.plot(wl, x.iloc[i].to_numpy(dtype=np.float32), alpha=0.3)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance")
            plt.title("Spectra with at least one NaN trait")
            plt.tight_layout()
            plt.savefig(out_dir / "spectra_with_nan_traits.png", dpi=150)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("results/eda"))
    parser.add_argument("--n_traits", type=int, default=20)
    parser.add_argument("--apply_red_nir_filter", action="store_true")
    args = parser.parse_args()

    run_eda(
        data_path=args.data_path,
        out_dir=args.out_dir,
        n_traits=args.n_traits,
        apply_red_nir_filter=bool(args.apply_red_nir_filter),
    )
