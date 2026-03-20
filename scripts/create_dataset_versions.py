from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRAIT_COLS_20 = [
    "Anth_area_ug_cm2",
    "Boron_area_mg_cm2",
    "C_area_mg_cm2",
    "Ca_area_mg_cm2",
    "Car_area_ug_cm2",
    "Cellulose_mg_cm2",
    "Chl_area_ug_cm2",
    "Cu_area_mg_cm2",
    "EWT_mg_cm2",
    "Fiber_mg_cm2",
    "LAI_m2_m2",
    "LMA_g_m2",
    "Lignin_mg_cm2",
    "Mg_area_mg_cm2",
    "Mn_area_mg_cm2",
    "NSC_mg_cm2",
    "N_area_mg_cm2",
    "P_area_mg_cm2",
    "Potassium_area_mg_cm2",
    "S_area_mg_cm2",
]


def _parse_manual_indices(values: Iterable[str]) -> list[int]:
    out: set[int] = set()
    for raw in values:
        token = str(raw).strip()
        if not token:
            continue
        if ":" in token:
            a, b = token.split(":", 1)
            start = int(a)
            end = int(b)
            lo, hi = sorted((start, end))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    return sorted(out)


def _wavelength_columns(df: pd.DataFrame) -> list[str]:
    wl = [c for c in df.columns if c.isdigit() and 350 <= int(c) <= 2600]
    if not wl:
        raise ValueError("No wavelength columns found.")
    return sorted(wl, key=lambda c: int(c))


def _present_trait_cols(df: pd.DataFrame, trait_cols: list[str]) -> list[str]:
    present = [c for c in trait_cols if c in df.columns]
    if not present:
        raise ValueError("None of the configured trait columns were found.")
    return present


def clean_dataset(
    input_csv: Path,
    output_csv: Path,
    report_dir: Path,
    trait_cols: list[str],
    manual_drop_idx: list[int],
    dup_keep: str = "last",
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    wl_cols = _wavelength_columns(df)
    present_traits = _present_trait_cols(df, trait_cols)

    x = df[wl_cols]
    y = df[present_traits]

    all_traits_nan = y.isna().all(axis=1)
    xy = pd.concat([x, y], axis=1)
    dup_all = xy.duplicated(keep=False)
    dup_drop_mask = xy.duplicated(keep=dup_keep)

    n_rows = len(df)
    manual_mask = np.zeros(n_rows, dtype=bool)
    ignored_manual = []
    for idx in manual_drop_idx:
        if 0 <= idx < n_rows:
            manual_mask[idx] = True
        else:
            ignored_manual.append(idx)

    drop_mask = all_traits_nan.to_numpy() | dup_drop_mask.to_numpy() | manual_mask
    cleaned = df.loc[~drop_mask].reset_index(drop=True)
    cleaned.to_csv(output_csv, index=False)

    report = pd.DataFrame(
        {
            "row_index": np.arange(n_rows, dtype=np.int64),
            "manual_drop": manual_mask,
            "all_traits_nan": all_traits_nan.to_numpy(),
            "dup_xy_all_instances": dup_all.to_numpy(),
            "dup_xy_drop_mask": dup_drop_mask.to_numpy(),
            "drop_sample": drop_mask,
        }
    )
    report.to_csv(report_dir / f"{input_csv.stem}_v2_drop_report.csv", index=False)
    report.loc[report["drop_sample"]].to_csv(
        report_dir / f"{input_csv.stem}_v2_dropped_rows.csv", index=False
    )
    report.loc[report["dup_xy_all_instances"]].to_csv(
        report_dir / f"{input_csv.stem}_v2_duplicate_groups.csv", index=False
    )

    summary = pd.Series(
        {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "n_rows_before": int(n_rows),
            "n_rows_after": int(len(cleaned)),
            "n_rows_dropped": int(drop_mask.sum()),
            "manual_drop_requested": int(len(manual_drop_idx)),
            "manual_drop_applied": int(manual_mask.sum()),
            "manual_drop_ignored_out_of_range": int(len(ignored_manual)),
            "all_traits_nan": int(all_traits_nan.sum()),
            "dup_xy_all_instances": int(dup_all.sum()),
            "dup_xy_drop_mask": int(dup_drop_mask.sum()),
            "dup_keep": dup_keep,
            "n_wavelength_cols": int(len(wl_cols)),
            "n_trait_cols_used": int(len(present_traits)),
        }
    )
    summary.to_csv(report_dir / f"{input_csv.stem}_v2_summary.csv", header=["value"])

    if ignored_manual:
        pd.Series(ignored_manual).to_csv(
            report_dir / f"{input_csv.stem}_v2_ignored_manual_indices.csv",
            index=False,
            header=["row_index"],
        )

    print(f"[OK] {input_csv.name} -> {output_csv.name}")
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create cleaned dataset versions with manual index drops, "
            "all-NaN trait drops, and exact duplicate (spectra+traits) handling."
        )
    )
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        default=[
            "data/processed/DB_50_Meta_EC_with_locations_clean1721.csv",
            "data/processed/DB_50_Meta_EC_with_locations_clean1522.csv",
        ],
    )
    parser.add_argument("--output_suffix", type=str, default="_v2")
    parser.add_argument("--report_dir", type=Path, default=Path("results/eda/versioning"))
    parser.add_argument(
        "--manual_drop",
        nargs="*",
        default=["5642:5688", "7210", "4431"],
        help="Indices or inclusive ranges start:end, e.g. 4431 5642:5688",
    )
    parser.add_argument(
        "--dup_keep",
        type=str,
        default="last",
        choices=["first", "last", "False", "none"],
        help="Duplicate keep policy for xy. 'none'/'False' drops all duplicate instances.",
    )
    args = parser.parse_args()

    keep = args.dup_keep
    if keep in {"False", "none"}:
        keep = False

    manual_indices = _parse_manual_indices(args.manual_drop)
    for in_path in args.input_csvs:
        input_csv = Path(in_path)
        if not input_csv.exists():
            raise FileNotFoundError(f"Input not found: {input_csv}")
        output_csv = input_csv.with_name(f"{input_csv.stem}{args.output_suffix}.csv")
        clean_dataset(
            input_csv=input_csv,
            output_csv=output_csv,
            report_dir=args.report_dir,
            trait_cols=TRAIT_COLS_20,
            manual_drop_idx=manual_indices,
            dup_keep=keep,  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    main()
