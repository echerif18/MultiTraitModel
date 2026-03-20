from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path("data/processed")
    report_root = Path("results/eda/versioning")
    report_root.mkdir(parents=True, exist_ok=True)

    src_1721 = root / "DB_50_Meta_EC_with_locations_clean1721.csv"
    src_1522 = root / "DB_50_Meta_EC_with_locations_clean1522.csv"
    drop_report_1721 = report_root / "DB_50_Meta_EC_with_locations_clean1721_v2_drop_report.csv"

    out_1721 = root / "DB_50_Meta_EC_with_locations_clean1721_v2.csv"
    out_1522 = root / "DB_50_Meta_EC_with_locations_clean1522_v2.csv"
    out_1522_aligned = root / "DB_50_Meta_EC_with_locations_clean1522_v2_aligned_to_1721.csv"

    if not src_1721.exists() or not src_1522.exists() or not drop_report_1721.exists():
        raise FileNotFoundError("Required input file(s) missing for alignment run.")

    rep = pd.read_csv(drop_report_1721)
    keep_idx = rep.loc[~rep["drop_sample"], "row_index"].to_numpy(dtype=np.int64)
    drop_idx = rep.loc[rep["drop_sample"], "row_index"].to_numpy(dtype=np.int64)

    df1721 = pd.read_csv(src_1721, low_memory=False)
    df1522 = pd.read_csv(src_1522, low_memory=False)

    if int(keep_idx.max()) >= len(df1522):
        raise ValueError("1522 dataset does not contain all 1721 reference indices.")

    clean1721 = df1721.loc[keep_idx].reset_index(drop=True)
    clean1522 = df1522.loc[keep_idx].reset_index(drop=True)

    clean1721.to_csv(out_1721, index=False)
    clean1522.to_csv(out_1522, index=False)
    clean1522.to_csv(out_1522_aligned, index=False)

    # Report what was dropped in 1522 to enforce 1721 alignment
    forced_drop_mask_1522 = np.ones(len(df1522), dtype=bool)
    forced_drop_mask_1522[keep_idx] = False
    forced_drop_1522 = pd.DataFrame(
        {
            "row_index": np.arange(len(df1522), dtype=np.int64),
            "drop_for_1721_alignment": forced_drop_mask_1522,
            "in_1721_drop_mask": np.isin(np.arange(len(df1522)), drop_idx),
            "beyond_1721_length": np.arange(len(df1522)) >= len(df1721),
        }
    )
    forced_drop_1522 = forced_drop_1522.loc[forced_drop_1522["drop_for_1721_alignment"]]
    forced_drop_1522.to_csv(report_root / "DB_50_Meta_EC_with_locations_clean1522_v2_alignment_drops.csv", index=False)

    summary = pd.Series(
        {
            "reference_keep_count": int(len(keep_idx)),
            "reference_drop_count": int(len(drop_idx)),
            "rows_1721_input": int(len(df1721)),
            "rows_1522_input": int(len(df1522)),
            "rows_1721_v2": int(len(clean1721)),
            "rows_1522_v2": int(len(clean1522)),
            "rows_1522_dropped_for_alignment": int(forced_drop_mask_1522.sum()),
            "rows_1522_dropped_beyond_1721_length": int((np.arange(len(df1522)) >= len(df1721)).sum()),
        }
    )
    summary.to_csv(report_root / "v2_alignment_summary.csv", header=["value"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
