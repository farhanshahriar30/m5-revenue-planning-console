from __future__ import annotations

# Phase A: Streamlit entrypoint
# - Keeps routing simple: Streamlit will auto-load files in app/pages for multi-page apps.
# - This file is mainly for global config and a small landing page.

import streamlit as st


def main() -> None:
    # Phase B: App-wide settings (title, layout)
    st.set_page_config(
        page_title="M5 Revenue Planning Console",
        layout="wide",
    )

    st.title("M5 Revenue Planning Console")

    st.write(
        "This app turns raw retail sales data into a simple planning dashboard. "
        "We first combined daily units sold, weekly prices, and the calendar of events to calculate revenue. "
        "Then we forecast revenue for the next 28 days and show a realistic range (low / expected / high) instead of a single number."
    )

    st.write("What you can do here:")
    st.markdown(
        "- **Executive Overview:** See total revenue trends, recent actuals, and the next 28-day forecast.\n"
        "- **Store Explorer:** Drill into one store’s forecast and download the numbers.\n"
        "- **Department Explorer:** See the top departments in each store and how much they contribute.\n"
        "- **Scenario Lab:** Move a slider to simulate price changes and compare against the baseline forecast.\n"
        "- **Risk & Planning:** Plan using conservative vs buffered assumptions and see revenue risk and upside.\n"
        "- **Model Health:** View basic accuracy checks to build trust in the predictions."
    )


st.info("Use the sidebar to navigate through the pages.")


if __name__ == "__main__":
    main()
