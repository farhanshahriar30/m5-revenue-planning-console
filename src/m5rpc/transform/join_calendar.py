from __future__ import annotations

import pandas as pd

from m5rpc.config.settings import settings


def main() -> None:
    sales_long_path = settings.PROCESSED_DIR / "sales_long.parquet"
    cal_path = settings.RAW_DIR / "calendar.csv"
    out_path = settings.PROCESSED_DIR / "sales_long_cal.parquet"

    print(f"Reading sales long: {sales_long_path}")
    sales_long = pd.read_parquet(sales_long_path)

    print(f"Reading calendar: {cal_path}")
    cal = pd.read_csv(cal_path)

    # Keep only what we need (reduce memory)
    keep_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    keep_cols = [c for c in keep_cols if c in cal.columns]
    cal = cal[keep_cols].copy()

    # Parse date
    if "date" in cal.columns:
        cal["date"] = pd.to_datetime(cal["date"])

    print("Joining on column: d")
    merged = sales_long.merge(cal, on="d", how="left", validate="m:1")

    # Validation
    null_dates = merged["date"].isna().mean() if "date" in merged.columns else None
    print(f"Merged shape: {merged.shape}")
    if null_dates is not None:
        print(f"Null date rate: {null_dates:.6f}")

    print(f"Writing: {out_path}")
    merged.to_parquet(out_path, index=False)
    print("✅ Done. Created sales_long_cal.parquet")


if __name__ == "__main__":
    main()
