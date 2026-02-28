from __future__ import annotations

import pandas as pd

from m5rpc.config.settings import settings


ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def main() -> None:
    sales_path = settings.RAW_DIR / "sales_train_evaluation.csv"
    out_path = settings.PROCESSED_DIR / "sales_long.parquet"

    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading sales: {sales_path}")
    sales = pd.read_csv(sales_path)

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if not day_cols:
        raise ValueError("No day columns found (expected columns like d_1, d_2, ...)")

    print(f"Rows (item-store): {len(sales):,}")
    print(f"Day columns: {len(day_cols):,}")

    # Wide -> long
    long_df = sales.melt(
        id_vars=ID_COLS,
        value_vars=day_cols,
        var_name="d",
        value_name="units",
    )

    # Basic cleanup / types
    long_df["units"] = long_df["units"].astype("int16")

    print(f"Long shape: {long_df.shape}")
    print(f"Writing: {out_path}")
    long_df.to_parquet(out_path, index=False)

    print("✅ Done. Created sales_long.parquet")


if __name__ == "__main__":
    main()
