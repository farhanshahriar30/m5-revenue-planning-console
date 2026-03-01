from __future__ import annotations

import bootstrap

bootstrap.add_src_to_path()
# Phase A: Risk & Planning goal
# - Convert uncertainty bands into actionable planning metrics.
# - "Revenue at risk" helps conservative planning (how much could we lose vs expected).
# - "Upside potential" helps aggressive planning (how much could we gain vs expected).

import pandas as pd
import streamlit as st

from m5rpc.config.settings import settings

st.set_page_config(page_title="M5 Revenue Planning Console", layout="wide")


@st.cache_data
def load_store_forecast() -> pd.DataFrame:
    # Phase B: Load store-level forecast outputs
    path = settings.FORECASTS_DIR / "store_latest.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    st.title("Risk & Planning")

    df = load_store_forecast()

    # Phase C: Aggregate 28-day totals per store (planning is usually done at horizon totals)
    totals = (
        df.groupby("store_id", as_index=False)[["p10", "p50", "p90"]]
        .sum()
        .rename(columns={"p10": "p10_28d", "p50": "p50_28d", "p90": "p90_28d"})
    )

    # Phase D: Risk metrics
    totals["revenue_at_risk"] = totals["p50_28d"] - totals["p10_28d"]
    totals["upside_potential"] = totals["p90_28d"] - totals["p50_28d"]

    # Phase E: Planner choice (what number do we "plan at"?)
    plan_level = st.radio(
        "Plan at", ["P50 (Expected)", "P90 (Buffered)"], horizontal=True
    )
    totals["plan_value"] = (
        totals["p50_28d"] if plan_level.startswith("P50") else totals["p90_28d"]
    )

    # Phase F: Overall summary cards
    total_plan = float(totals["plan_value"].sum())
    total_risk = float(totals["revenue_at_risk"].sum())
    total_upside = float(totals["upside_potential"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total planned revenue", f"{total_plan:,.0f}")
    c2.metric("Total revenue at risk (P50−P10)", f"{total_risk:,.0f}")
    c3.metric("Total upside potential (P90−P50)", f"{total_upside:,.0f}")

    # Phase G: Store ranking table
    rank_options = {
        "Planned revenue": "plan_value",
        "Revenue at risk": "revenue_at_risk",
        "Upside potential": "upside_potential",
    }

    rank_label = st.selectbox("Rank stores by", list(rank_options.keys()))
    sort_by = rank_options[rank_label]

    view = totals.sort_values(sort_by, ascending=False).reset_index(drop=True)

    st.subheader("Store planning table (28-day totals)")
    st.dataframe(view, width="stretch")

    st.caption(
        "This table helps you plan with uncertainty in mind. "
        "‘Planned revenue’ is what you plan for. "
        "‘Revenue at risk’ shows how much could be lost in a worse case. "
        "‘Upside potential’ shows how much could be gained in a best case."
    )


if __name__ == "__main__":
    main()
