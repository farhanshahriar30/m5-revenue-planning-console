from __future__ import annotations

# Phase A: Scenario Lab goal
# - Explore "what if prices change by X%" on the revenue forecast.
# - Scenario logic is explicit: a multiplicative projection on revenue
#   (not a causal demand response model).
# - Use Altair for:
#     1) locked x-axis to forecast window
#     2) clean legend labels
#     3) baseline (solid) vs scenario (dashed)
#     4) consistent hover tooltips

import json

import pandas as pd
import streamlit as st
import altair as alt

from m5rpc.config.settings import settings
from m5rpc.scenarios.scenario_engine import apply_price_scenario

st.set_page_config(page_title="M5 Revenue Planning Console", layout="wide")


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
        depts = top_map[str(store_id)]
        dept_id = st.selectbox("Select department", depts)

    # Phase F: Price slider
    price_delta = st.slider(
        "Price change (%)", min_value=-10, max_value=10, value=0, step=1
    )

    # Phase G: Build baseline series for the selected scope
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
    scen = result.scenario.sort_values("date").reset_index(drop=True)

    # Phase I: KPI delta cards (P50 totals)
    s = result.summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline total (P50)", f"{s['baseline_total_p50']:,.0f}")
    c2.metric("Scenario total (P50)", f"{s['scenario_total_p50']:,.0f}")
    c3.metric("Delta (P50)", f"{s['delta_total_p50']:,.0f}")

    st.subheader("Baseline vs Scenario (P10/P50/P90)")

    # Phase J: Build plotting dataframe (long format)
    # - We create six series:
    #     P10/P50/P90 (Baseline) and P10/P50/P90 (Scenario)
    plot_base = base.melt(
        id_vars=["date"],
        value_vars=["p10", "p50", "p90"],
        var_name="q",
        value_name="revenue",
    )
    plot_base["series"] = (
        plot_base["q"].map({"p10": "P10", "p50": "P50", "p90": "P90"}) + " (Baseline)"
    )
    plot_base["style"] = "Baseline"

    plot_scen = scen.melt(
        id_vars=["date"],
        value_vars=["p10_scenario", "p50_scenario", "p90_scenario"],
        var_name="q",
        value_name="revenue",
    )
    plot_scen["series"] = (
        plot_scen["q"].map(
            {"p10_scenario": "P10", "p50_scenario": "P50", "p90_scenario": "P90"}
        )
        + " (Scenario)"
    )
    plot_scen["style"] = "Scenario"

    plot_df = pd.concat(
        [
            plot_base[["date", "series", "style", "revenue"]],
            plot_scen[["date", "series", "style", "revenue"]],
        ],
        ignore_index=True,
    )

    x_min = plot_df["date"].min()
    x_max = plot_df["date"].max()

    # Phase K: Consistent hover
    nearest = alt.selection_point(
        fields=["date", "series"], nearest=True, on="mouseover", empty=False
    )

    base_chart = alt.Chart(plot_df).encode(
        x=alt.X(
            "date:T",
            title=None,
            axis=alt.Axis(format="%b %d", labelAngle=0),
            scale=alt.Scale(domain=[x_min, x_max]),
        ),
        y=alt.Y(
            "revenue:Q", title="Revenue", axis=alt.Axis(format=",.0f", tickCount=6)
        ),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
    )

    # Baseline solid lines
    baseline_lines = (
        base_chart.transform_filter(alt.datum.style == "Baseline")
        .mark_line()
        .encode(
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("revenue:Q", title="Revenue", format=",.0f"),
            ]
        )
    )

    # Scenario dashed lines
    scenario_lines = (
        base_chart.transform_filter(alt.datum.style == "Scenario")
        .mark_line(strokeDash=[6, 4])
        .encode(
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("revenue:Q", title="Revenue", format=",.0f"),
            ]
        )
    )

    hover_points = base_chart.mark_point(opacity=0).add_params(nearest)
    hover_rule = (
        alt.Chart(plot_df)
        .mark_rule(opacity=0.25)
        .encode(x=alt.X("date:T", scale=alt.Scale(domain=[x_min, x_max])))
        .transform_filter(nearest)
    )

    chart = (baseline_lines + scenario_lines + hover_rule + hover_points).properties(
        height=360
    )

    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Scenario is a revenue multiplier on the baseline forecast (planning projection)."
    )


if __name__ == "__main__":
    main()
