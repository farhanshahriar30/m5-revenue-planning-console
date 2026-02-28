from __future__ import annotations

import pandas as pd

from m5rpc.config.settings import settings


def main() -> None:
    in_path = settings.PROCESSED_DIR / "item_store_day.parquet"
    out_path = settings.PROCESSED_DIR / "item_store_day.parquet"  # overwrite in place

    print(f"Reading: {in_path}")
    df = pd.read_parquet(in_path)

    # Cases:
    # 1) units == 0 and sell_price is NaN -> revenue should be 0
    # 2) units > 0 and sell_price is NaN -> problematic, we will report count
    missing_price = df["sell_price"].isna()
    zero_units = df["units"] == 0

    case_ok = missing_price & zero_units
    case_bad = missing_price & (~zero_units)

    bad_count = int(case_bad.sum())
    ok_count = int(case_ok.sum())

    print(f"Missing price & zero units rows (safe -> revenue=0): {ok_count:,}")
    print(f"Missing price & units>0 rows (problem): {bad_count:,}")

    # Apply fix for safe rows
    df.loc[case_ok, "sell_price"] = 0.0
    df.loc[case_ok, "revenue"] = 0.0

    # Recompute null rates after fix
    null_price_rate = df["sell_price"].isna().mean()
    null_rev_rate = df["revenue"].isna().mean()
    print(f"Post-fix null sell_price rate: {null_price_rate:.6f}")
    print(f"Post-fix null revenue rate: {null_rev_rate:.6f}")

    print(f"Writing cleaned table: {out_path}")
    df.to_parquet(out_path, index=False)
    print("✅ Done. Cleaned item_store_day.parquet")


if __name__ == "__main__":
    main()
