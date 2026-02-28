from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from m5rpc.config.settings import settings


def main() -> None:
    """
    Phase A: Load the "gold" item-store-day table
    - This table is already joined (sales + calendar + prices) and cleaned (no null revenue).
    - Columns include: store_id, dept_id, date, units, sell_price, revenue, plus event/SNAP features.
    """
    in_path = settings.PROCESSED_DIR / "item_store_day.parquet"
    df = pd.read_parquet(in_path)

    # Ensure date is datetime (parquet should preserve, but we enforce to be safe)
    df["date"] = pd.to_datetime(df["date"])

    """
    Phase B: Store-day revenue aggregation
    - Why: forecasting per item is too large; store-day is operationally meaningful and much smaller.
    - Output: one row per (date, store_id) with revenue summed across all items.
    """
    store_day = (
        df.groupby(["date", "store_id"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "store_revenue"})
    )

    store_day_out = settings.PROCESSED_DIR / "store_day.parquet"
    store_day.to_parquet(store_day_out, index=False)
    print(f"✅ Wrote: {store_day_out} | shape={store_day.shape}")

    """
    Phase C: Identify top N departments per store by historical revenue
    - Why: "Top 5 depts per store" gives a realistic drill-down without exploding series count.
    - Method:
        1) aggregate total revenue by (store_id, dept_id) over ALL history
        2) rank depts within each store by total revenue
        3) keep TOP_DEPTS_PER_STORE
    - Output: JSON mapping store_id -> list of dept_ids
    """
    dept_totals = (
        df.groupby(["store_id", "dept_id"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "dept_total_revenue"})
    )

    # Rank departments within each store
    dept_totals["rank_in_store"] = dept_totals.groupby("store_id")[
        "dept_total_revenue"
    ].rank(method="first", ascending=False)

    top_n = settings.TOP_DEPTS_PER_STORE
    top_depts = dept_totals[dept_totals["rank_in_store"] <= top_n].copy()

    # Build mapping store -> [dept1, dept2, ...]
    top_map = (
        top_depts.sort_values(["store_id", "rank_in_store"])
        .groupby("store_id")["dept_id"]
        .apply(list)
        .to_dict()
    )

    top_map_path = settings.ARTIFACTS_DIR / "metadata" / "top_depts_by_store.json"
    top_map_path.parent.mkdir(parents=True, exist_ok=True)
    top_map_path.write_text(json.dumps(top_map, indent=2))
    print(f"✅ Wrote: {top_map_path} | stores={len(top_map)} | top_n={top_n}")

    """
    Phase D: Store-dept-day aggregation (filtered to top N depts per store)
    - Why: we only want series that matter most for storytelling and planning.
    - Steps:
        1) filter raw rows to only (store_id, dept_id) pairs in the top_map
        2) group by (date, store_id, dept_id) and sum revenue
    """
    top_pairs = top_depts[["store_id", "dept_id"]].drop_duplicates()

    # Semi-join: keep only rows that match one of the top (store, dept) pairs
    df_top = df.merge(top_pairs, on=["store_id", "dept_id"], how="inner")

    store_dept_day = (
        df_top.groupby(["date", "store_id", "dept_id"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "dept_revenue"})
    )

    store_dept_out = settings.PROCESSED_DIR / "store_dept_day_top5.parquet"
    store_dept_day.to_parquet(store_dept_out, index=False)
    print(f"✅ Wrote: {store_dept_out} | shape={store_dept_day.shape}")

    """
    Phase E: Quick sanity prints
    - Confirm the number of stores, and average #depts per store included (should be TOP_DEPTS_PER_STORE).
    """
    stores = store_day["store_id"].nunique()
    avg_depts = top_pairs.groupby("store_id")["dept_id"].nunique().mean()
    print(f"Stores in store_day: {stores}")
    print(f"Avg depts per store in top set: {avg_depts:.2f}")


if __name__ == "__main__":
    main()
