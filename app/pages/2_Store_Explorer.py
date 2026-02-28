from __future__ import annotations

# Phase A: Store Explorer goal
# - Let users drill into one store's 28-day forecast with uncertainty bands.
# - Provide a forecast table and a download button for operational use.

import pandas as pd
import streamlit as st

from m5rpc.config.settings import settings


@st.cache_data
def load_store_forecast() -> pd.DataFrame:
    # Phase B: Load precomputed forecasts for fast UI
    path = settings.FORECASTS_DIR / "store_latest.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    st.title("Store Explorer")

    df = load_store_forecast()

    # Phase C: Store selector
    stores = sorted(df["store_id"].unique().tolist())
    store_id = st.selectbox("Select a store", stores)

    # Phase D: Filter to selected store
    s = df[df["store_id"] == store_id].sort_values("date").reset_index(drop=True)

    # Phase E: Chart
    st.subheader(f"Forecast for {store_id}")
    st.line_chart(s.set_index("date")[["p10", "p50", "p90"]])

    # Phase F: Table + download
    st.subheader("28-day forecast table")
    st.dataframe(s, use_container_width=True)

    csv = s.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{store_id}_28d_forecast.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
