from __future__ import annotations

# Phase A: Scenario Lab goal
# - Let users explore "what if prices change by X%" on the revenue forecast.
# - We keep the scenario logic explicit: this is a multiplicative projection on revenue,
#   not a full causal demand response model.

import json

import pandas as pd
import streamlit as st

from m5rpc.config.settings import settings
from m5rpc.scenarios.scenario_engine import apply_price_scenario


@st.cache_data
def load_top_depts_map() -> dict:
    # Phase B: Used to populate dept dropdown based on store selection
    path = settings.ARTIFACTS_DIR / "metadata" / "top_depts_by_store.json"
    return json.loads(path.read_text())


@st.cache_data
def load_store_forecast() -> pd.DataFrame:
    # Phase C: Store-level baseline forecasts
    path = settings.FORECASTS_DIR / "store_latest.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_store_dept_forecast() -> pd.DataFrame:
    # Phase D: Store+dept baseline forecasts
    path = settings.FORECASTS_DIR / "store_dept_top5_latest.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    st.title("Scenario Lab")

    top_map = load_top_depts_map()
    store_df = load_store_forecast()
    dept_df = load_store_dept_forecast()

    # Phase E: Scope selection
    scope = st.radio(
        "Scenario scope",
        ["All stores", "One store", "One store + dept"],
        horizontal=True,
    )

    store_id = None
    dept_id = None

    if scope in ("One store", "One store + dept"):
        store_id = st.selectbox(
            "Select store", sorted(store_df["store_id"].unique().tolist())
        )

    if scope == "One store + dept":
        # Dept list comes from top_map (top 5 for this store)
        depts = top_map[str(store_id)]
        dept_id = st.selectbox("Select department", depts)

    # Phase F: Price slider
    price_delta = st.slider(
        "Price change (%)", min_value=-10, max_value=10, value=0, step=1
    )

    # Phase G: Build the baseline series that we will adjust
    if scope == "All stores":
        base = store_df.groupby("date", as_index=False)[["p10", "p50", "p90"]].sum()
    elif scope == "One store":
        base = store_df[store_df["store_id"] == store_id][
            ["date", "p10", "p50", "p90"]
        ].copy()
    else:
        base = dept_df[
            (dept_df["store_id"] == store_id) & (dept_df["dept_id"] == dept_id)
        ][["date", "p10", "p50", "p90"]].copy()

    base = base.sort_values("date").reset_index(drop=True)

    # Phase H: Apply scenario projection
    result = apply_price_scenario(base, price_delta_pct=price_delta)
    scen = result.scenario

    # Phase I: KPI delta cards (P50 totals)
    s = result.summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline total (P50)", f"{s['baseline_total_p50']:,.0f}")
    c2.metric("Scenario total (P50)", f"{s['scenario_total_p50']:,.0f}")
    c3.metric("Delta (P50)", f"{s['delta_total_p50']:,.0f}")

    st.subheader("Baseline vs Scenario (P10/P50/P90)")

    # Phase J: Overlay chart
    chart_df = scen[
        ["date", "p10", "p50", "p90", "p10_scenario", "p50_scenario", "p90_scenario"]
    ].set_index("date")
    st.line_chart(chart_df)

    st.caption(
        "Scenario is a revenue multiplier on the baseline forecast (planning projection)."
    )


if __name__ == "__main__":
    main()
