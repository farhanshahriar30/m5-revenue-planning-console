from __future__ import annotations

# Phase A: Store Explorer goal
# - Let users drill into one store's 28-day forecast with uncertainty bands.
# - Improve chart quality vs st.line_chart:
#     1) lock x-axis to exactly the forecast window
#     2) consistent hover tooltips (nearest point)
# - Clean up table date formatting and keep download.

import pandas as pd
import streamlit as st
import altair as alt

from m5rpc.config.settings import settings

st.set_page_config(page_title="M5 Revenue Planning Console", layout="wide")


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

    st.subheader(f"Forecast for {store_id}")

    # Phase E: Prepare long-form data for Altair
    # - Altair works best with one value column + a label column for the band.

    plot_df = s.melt(
        id_vars=["store_id", "date"],
        value_vars=["p10", "p50", "p90"],
        var_name="band",
        value_name="revenue",
    )
    plot_df["band"] = plot_df["band"].map({"p10": "P10", "p50": "P50", "p90": "P90"})

    x_min = plot_df["date"].min()
    x_max = plot_df["date"].max()

    # Phase F: Consistent hover tooltips
    # - We add an invisible point layer + nearest selection so hover feels reliable.

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
            "revenue:Q",
            title="Revenue",
            axis=alt.Axis(format=",.0f", tickCount=6),
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
    st.altair_chart(chart, use_container_width=True)

    # Phase G: Table + download (clean date display)
    st.subheader("28-day forecast table")
    table_df = s.copy()
    table_df["date"] = table_df["date"].dt.date
    st.dataframe(table_df, width="stretch")

    csv = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{store_id}_28d_forecast.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
