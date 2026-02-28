from __future__ import annotations

import pandas as pd

from m5rpc.config.settings import settings


def _assert_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def main() -> None:
    sales_path = settings.RAW_DIR / "sales_train_evaluation.csv"
    cal_path = settings.RAW_DIR / "calendar.csv"
    price_path = settings.RAW_DIR / "sell_prices.csv"

    _assert_exists(sales_path)
    _assert_exists(cal_path)
    _assert_exists(price_path)

    # Load small samples first (fast sanity checks)
    sales_head = pd.read_csv(sales_path, nrows=5)
    cal_head = pd.read_csv(cal_path, nrows=5)
    price_head = pd.read_csv(price_path, nrows=5)

    print("✅ Raw files found.")
    print(f"sales_train_evaluation.csv sample shape: {sales_head.shape}")
    print(f"calendar.csv sample shape: {cal_head.shape}")
    print(f"sell_prices.csv sample shape: {price_head.shape}")

    # Print key columns we will join on
    print("\nKey columns present?")
    print(
        "sales:",
        [
            c
            for c in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
            if c in sales_head.columns
        ],
    )
    print(
        "calendar:",
        [
            c
            for c in [
                "d",
                "date",
                "wm_yr_wk",
                "event_name_1",
                "event_type_1",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ]
            if c in cal_head.columns
        ],
    )
    print(
        "prices:",
        [
            c
            for c in ["store_id", "item_id", "wm_yr_wk", "sell_price"]
            if c in price_head.columns
        ],
    )


if __name__ == "__main__":
    main()
