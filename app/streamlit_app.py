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

    # Phase C: Landing page content
    st.title("M5 Revenue Planning Console")
    st.write(
        "Use the pages in the sidebar to explore forecasts, drill into stores/departments, "
        "run price scenarios, and review model health."
    )
    st.info("Next: Open the **Overview** page from the sidebar.")


if __name__ == "__main__":
    main()
