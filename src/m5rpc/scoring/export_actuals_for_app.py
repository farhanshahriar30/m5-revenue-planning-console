from __future__ import annotations

# Phase A: Purpose
# - Streamlit Cloud won't big local files in data/processed (they're gitignored).
# - We export a SMALL "actuals" file that we CAN commit:
#     total revenue across all stores for the last 90 days.
# - This does NOT change results, it only packages a small slice for the UI.

import pandas as pd

from m5rpc.config.settings import settings


def main() -> None:
    # Phase B: Load full store-day actuals (local, big file)
    in_path = settings.PROCESSED_DIR / "store_day.parquet"
    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"])

    # Phase C: Aggregate across all stores (executive view)
    total = (
        df.groupby("date", as_index=False)["store_revenue"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Phase D: Keep only the last 90 days (small, deployable)
    total_90 = total.tail(90).copy()

    # Phase E: Write to data/outputs (small + commit-safe)
    out_dir = settings.DATA_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "actuals_allstores_last_90.parquet"
    total_90.to_parquet(out_path, index=False)

    print(f"✅ Wrote: {out_path} | shape={total_90.shape}")


if __name__ == "__main__":
    main()
