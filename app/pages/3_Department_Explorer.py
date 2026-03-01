from __future__ import annotations

import bootstrap

bootstrap.add_src_to_path()
# Phase A: Department Explorer goal
# - Drill down from store -> top 5 departments (historically highest revenue).
# - Show 28-day forecast bands for a selected store+dept.
# - Provide contribution metrics and a clean export-ready table.
# - Use Altair for a locked date axis + consistent hover.

import json

import pandas as pd
import streamlit as st
import altair as alt

from m5rpc.config.settings import settings

st.set_page_config(page_title="M5 Revenue Planning Console", layout="wide")


@st.cache_data
def load_top_depts_map() -> dict:
    # Phase B: store -> [dept_id,...] mapping created during aggregation
    path = settings.ARTIFACTS_DIR / "metadata" / "top_depts_by_store.json"
    return json.loads(path.read_text())


@st.cache_data
def load_store_forecast() -> pd.DataFrame:
    # Phase C: Store-level forecasts (used for contribution calculations)
    path = settings.FORECASTS_DIR / "store_latest.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_store_dept_forecast() -> pd.DataFrame:
    # Phase D: Store+dept forecasts (top 5 per store)
    path = settings.FORECASTS_DIR / "store_dept_top5_latest.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    st.title("Department Explorer (Top 5 per Store)")

    top_map = load_top_depts_map()
    store_df = load_store_forecast()
    dept_df = load_store_dept_forecast()

    # Phase E: Store selector (only stores present in top_map)
    stores = sorted(top_map.keys())
    store_id = st.selectbox("Select a store", stores)

    # Phase F: Dept selector (top 5 for this store)
    depts = top_map[store_id]
    dept_id = st.selectbox("Select a department", depts)

    # Phase G: Filter to selected store+dept and store-only
    d = (
        dept_df[(dept_df["store_id"] == store_id) & (dept_df["dept_id"] == dept_id)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    s = (
        store_df[store_df["store_id"] == store_id]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Phase H: Contribution KPIs (next 28 days, using P50)
    dept_total = float(d["p50"].sum())
    store_total = float(s["p50"].sum())
    share = (dept_total / store_total) if store_total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Dept 28d total (P50)", f"{dept_total:,.0f}")
    c2.metric("Store 28d total (P50)", f"{store_total:,.0f}")
    c3.metric("Dept share of store (P50)", f"{share:.1%}")

    # Phase I: Altair chart (locked x-range + consistent hover)
    st.subheader(f"Forecast for {store_id} • {dept_id}")

    plot_df = d.melt(
        id_vars=["store_id", "dept_id", "date"],
        value_vars=["p10", "p50", "p90"],
        var_name="band",
        value_name="revenue",
    )
    plot_df["band"] = plot_df["band"].map({"p10": "P10", "p50": "P50", "p90": "P90"})

    x_min = plot_df["date"].min()
    x_max = plot_df["date"].max()

    nearest = alt.selection_point(
        fields=["date", "band"],
        nearest=True,
        on="mouseover",
        empty=False,
    )

    base = alt.Chart(plot_df).encode(
        x=alt.X(
            "date:T",
            title=None,
            axis=alt.Axis(format="%b %d", labelAngle=0),
            scale=alt.Scale(domain=[x_min, x_max]),
        ),
        y=alt.Y(
            "revenue:Q", title="Revenue", axis=alt.Axis(format=",.0f", tickCount=6)
        ),
        color=alt.Color("band:N", title=None, legend=alt.Legend(orient="top")),
    )

    lines = base.mark_line().encode(
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("band:N", title="Band"),
            alt.Tooltip("revenue:Q", title="Revenue", format=",.0f"),
        ]
    )

    hover_points = base.mark_point(opacity=0).add_params(nearest)

    hover_rule = (
        alt.Chart(plot_df)
        .mark_rule(opacity=0.25)
        .encode(x=alt.X("date:T", scale=alt.Scale(domain=[x_min, x_max])))
        .transform_filter(nearest)
    )

    chart = (lines + hover_rule + hover_points).properties(height=360)
    st.altair_chart(chart, width="stretch")

    st.caption(
        "This chart shows the selected department’s expected revenue for the next 28 days. "
        "P50 is the most likely forecast, and P10/P90 show a lower and higher range. "
        "The ‘Dept share’ tells you how much this department contributes to the store’s total forecast."
    )

    # Phase J: Table (clean date, clean index)
    st.subheader("28-day forecast table")
    table_df = d.copy()
    table_df["date"] = table_df["date"].dt.date
    st.dataframe(table_df, width="stretch")


if __name__ == "__main__":
    main()
