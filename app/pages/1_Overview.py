from __future__ import annotations

# Phase A: Overview page goal
# - Provide an executive summary of total revenue for:
#     1) recent actual history (last 90 days)
#     2) next 28-day forecast with uncertainty bands (P10/P50/P90)
# - Make the timeline readable and "executive-ready":
#     - clear date axis
#     - tooltips
#     - a cutoff marker between actuals and forecast

import pandas as pd
import streamlit as st
import altair as alt

from m5rpc.config.settings import settings

st.set_page_config(page_title="M5 Revenue Planning Console", layout="wide")


@st.cache_data
def load_store_forecast() -> pd.DataFrame:
    """
    Phase B: Load the latest precomputed forecast table (28 days)
    - Source: forecasts/store_latest.parquet
    - Contains store-level P10/P50/P90 per day.
    """
    path = settings.FORECASTS_DIR / "store_latest.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_store_actuals() -> pd.DataFrame:
    """
    Phase C: Load historical actual revenue (store-day)
    - Source: data/processed/store_day.parquet
    - Contains store_revenue per store per date.
    """
    path = settings.PROCESSED_DIR / "store_day.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    st.title("Executive Overview")

    # Phase D: Aggregate forecast across stores to create a single executive series
    f = load_store_forecast()
    forecast_total = (
        f.groupby("date", as_index=False)[["p10", "p50", "p90"]]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Phase E: KPI cards (next 7/14/28 days totals using forecast P50)
    def total_for_days(days: int) -> float:
        return float(forecast_total.head(days)["p50"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Next 7 days (P50)", f"{total_for_days(7):,.0f}")
    c2.metric("Next 14 days (P50)", f"{total_for_days(14):,.0f}")
    c3.metric("Next 28 days (P50)", f"{total_for_days(28):,.0f}")

    st.subheader("Revenue: Last 90 Days Actuals + Next 28 Days Forecast")

    # Phase F: Aggregate actuals across stores, then take last 90 days ending at forecast start - 1
    a = load_store_actuals()
    actual_total = (
        a.groupby("date", as_index=False)["store_revenue"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    forecast_start = forecast_total["date"].min()
    actual_end = forecast_start - pd.Timedelta(days=1)

    # Keep a clean 90-day window immediately before the forecast starts
    actual_90 = actual_total[actual_total["date"] <= actual_end].tail(90).copy()

    # Phase G: Prepare plotting data
    # - Actuals: one line
    # - Forecast: three lines (P10/P50/P90)
    actual_plot = actual_90.rename(columns={"store_revenue": "revenue"})
    actual_plot["series"] = "Actual"

    forecast_plot = forecast_total.melt(
        id_vars=["date"],
        value_vars=["p10", "p50", "p90"],
        var_name="series",
        value_name="revenue",
    )
    forecast_plot["series"] = forecast_plot["series"].map(
        {"p10": "P10", "p50": "P50", "p90": "P90"}
    )

    plot_df = pd.concat(
        [
            actual_plot[["date", "series", "revenue"]],
            forecast_plot[["date", "series", "revenue"]],
        ],
        ignore_index=True,
    )

    # Lock x-axis domain to the visible window (actuals + forecast)
    x_min = plot_df["date"].min()
    x_max = plot_df["date"].max()

    # Phase H: Build Altair chart layers
    # - We want one legend that includes BOTH:
    #     Actual + (P10/P50/P90)
    # - We still render Actual with a thicker stroke for emphasis.

    base = alt.Chart(plot_df).encode(
        x=alt.X(
            "date:T",
            title=None,
            axis=alt.Axis(format="%b %d", labelAngle=0),
            scale=alt.Scale(domain=[x_min, x_max]),
        )
    )

    # Shared legend/color encoding for all series (Actual, P10, P50, P90)
    color_enc = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"))

    # Forecast lines (P10/P50/P90)
    forecast_lines = (
        base.transform_filter(alt.datum.series != "Actual")
        .mark_line()
        .encode(
            y=alt.Y(
                "revenue:Q",
                title="Revenue",
                axis=alt.Axis(format=",.0f", tickCount=6),
            ),
            color=color_enc,
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("revenue:Q", title="Revenue", format=",.0f"),
            ],
        )
    )

    # Actual line (thicker, but included in the SAME legend)
    actual_line = (
        base.transform_filter(alt.datum.series == "Actual")
        .mark_line(strokeWidth=3)
        .encode(
            y=alt.Y(
                "revenue:Q",
                title="Revenue",
                axis=alt.Axis(format=",.0f", tickCount=6),
            ),
            color=color_enc,
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("revenue:Q", title="Revenue", format=",.0f"),
            ],
        )
    )

    # Cutoff marker where forecast begins
    cutoff_df = pd.DataFrame({"date": [forecast_start]})
    cutoff_line = alt.Chart(cutoff_df).mark_rule(strokeDash=[6, 6]).encode(x="date:T")

    # Phase H2: Make tooltips consistent
    # - Altair tooltips can feel "spotty" if you must hover exactly on the line.
    # - We add an invisible points layer + nearest selection, so hover snaps reliably.

    nearest = alt.selection_point(
        fields=["date", "series"],
        nearest=True,
        on="mouseover",
        empty=False,
    )

    hover_points = (
        alt.Chart(plot_df)
        .mark_point(opacity=0)
        .encode(
            x=alt.X(
                "date:T",
                title=None,
                axis=alt.Axis(format="%b %d", labelAngle=0),
                scale=alt.Scale(domain=[x_min, x_max]),
            ),
            y=alt.Y(
                "revenue:Q", title="Revenue", axis=alt.Axis(format=",.0f", tickCount=6)
            ),
            color=color_enc,
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("revenue:Q", title="Revenue", format=",.0f"),
            ],
        )
        .add_params(nearest)
    )

    hover_rule = (
        alt.Chart(plot_df)
        .mark_rule(opacity=0.25)
        .encode(
            x=alt.X("date:T", scale=alt.Scale(domain=[x_min, x_max])),
        )
        .transform_filter(nearest)
    )
    chart = (
        forecast_lines + actual_line + cutoff_line + hover_rule + hover_points
    ).properties(height=380)

    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "The light-blue line shows what actually happened over the last 90 days. "
        "The colored lines show the next 28 days. "
        "P50 is the most likely/typical forecast, P10 is a cautious low estimate, and P90 is an optimistic high estimate. "
        "The dashed vertical line marks where the forecast begins."
    )


if __name__ == "__main__":
    main()
