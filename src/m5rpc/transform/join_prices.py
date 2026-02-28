from __future__ import annotations

import pandas as pd

from m5rpc.config.settings import settings


def main() -> None:
    in_path = settings.PROCESSED_DIR / "sales_long_cal.parquet"
    price_path = settings.RAW_DIR / "sell_prices.csv"
    out_path = settings.PROCESSED_DIR / "item_store_day.parquet"

    print(f"Reading sales+calendar: {in_path}")
    df = pd.read_parquet(in_path)

    print(f"Reading prices: {price_path}")
    prices = pd.read_csv(price_path)

    # Reduce columns (memory)
    prices = prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]].copy()

    # Ensure join keys match types
    df["wm_yr_wk"] = df["wm_yr_wk"].astype("int32")
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")

    print("Joining on (store_id, item_id, wm_yr_wk)")
    merged = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="m:1",
    )

    # Compute revenue
    merged["sell_price"] = merged["sell_price"].astype("float32")
    merged["revenue"] = (
        merged["units"].astype("float32") * merged["sell_price"]
    ).astype("float32")

    # Validations
    null_price_rate = merged["sell_price"].isna().mean()
    null_rev_rate = merged["revenue"].isna().mean()
    print(f"Merged shape: {merged.shape}")
    print(f"Null sell_price rate: {null_price_rate:.6f}")
    print(f"Null revenue rate: {null_rev_rate:.6f}")

    print(f"Writing: {out_path}")
    merged.to_parquet(out_path, index=False)
    print("✅ Done. Created item_store_day.parquet")


if __name__ == "__main__":
    main()
