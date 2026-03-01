from __future__ import annotations

# Phase A: Model Health goal
# - Build trust by showing performance metrics and basic training metadata.
# - This page is the "why should I believe this forecast?" layer.

import json

import streamlit as st

from m5rpc.config.settings import settings

st.set_page_config(page_title="M5 Revenue Planning Console", layout="wide")


def _load_json(path):
    # Phase B: Small helper to read JSON metadata safely
    return json.loads(path.read_text())


def main() -> None:
    st.title("Model Health")

    store_metrics_path = (
        settings.ARTIFACTS_DIR / "metadata" / "store_train_metrics.json"
    )
    dept_metrics_path = (
        settings.ARTIFACTS_DIR / "metadata" / "store_dept_train_metrics.json"
    )

    st.subheader("Store-level model (P50 sanity check)")
    store_m = _load_json(store_metrics_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (P50)", f"{store_m['mae_p50']:.2f}")
    c2.metric("sMAPE (P50)", f"{store_m['smape_p50']:.2f}%")
    c3.metric("Test start date", store_m["split_cutoff_date"])
    st.json(store_m)

    st.subheader("Store–Dept model (Top 5 per store, P50 sanity check)")
    dept_m = _load_json(dept_metrics_path)
    c4, c5, c6 = st.columns(3)
    c4.metric("MAE (P50)", f"{dept_m['mae_p50']:.2f}")
    c5.metric("sMAPE (P50)", f"{dept_m['smape_p50']:.2f}%")
    c6.metric("Test start date", dept_m["split_cutoff_date"])
    st.json(dept_m)

    st.caption(
        "Next upgrade: rolling backtests + interval calibration (P10–P90 coverage)."
    )


if __name__ == "__main__":
    main()
