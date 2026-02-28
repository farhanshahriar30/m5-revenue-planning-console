from __future__ import annotations

# Phase A: Overview page goal
# - Provide an executive summary of total forecasted revenue across all stores.
# - Show uncertainty bands (P10/P50/P90) and simple KPI totals for 7/14/28 days.

from pathlib import Path

import pandas as pd
import streamlit as st

from m5rpc.config.settings import settings


@st.cache_data
def load_store_forecast() -> pd.DataFrame:
    # Phase B: Load the latest precomputed forecast table
    # - This keeps the UI fast (we do NOT score models inside Streamlit).
    path = settings.FORECASTS_DIR / "store_latest.parquet"
    return pd.read_parquet(path)


def main() -> None:
    st.title("Executive Overview")

    df = load_store_forecast().copy()
    df["date"] = pd.to_datetime(df["date"])

    # Phase C: Aggregate across stores for a single executive time series
    total = (
        df.groupby("date", as_index=False)[["p10", "p50", "p90"]]
        .sum()
        .sort_values("date")
    )

    # Phase D: KPI cards (next 7/14/28 days totals using P50)
    def total_for_days(days: int) -> float:
        return float(total.head(days)["p50"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Next 7 days (P50)", f"{total_for_days(7):,.0f}")
    c2.metric("Next 14 days (P50)", f"{total_for_days(14):,.0f}")
    c3.metric("Next 28 days (P50)", f"{total_for_days(28):,.0f}")

    st.subheader("Total Revenue Forecast (All Stores)")
    st.line_chart(total.set_index("date")[["p10", "p50", "p90"]])

    st.caption(
        "Bands are quantile forecasts: P10 (conservative), P50 (expected), P90 (aggressive)."
    )


if __name__ == "__main__":
    main()
