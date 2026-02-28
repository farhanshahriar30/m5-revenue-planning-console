from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScenarioResult:
    """
    Phase A: Lightweight container for scenario outputs
    - baseline: original forecast table
    - scenario: adjusted forecast table
    - summary: delta KPIs (e.g., total delta revenue)
    """

    baseline: pd.DataFrame
    scenario: pd.DataFrame
    summary: dict


def apply_price_scenario(
    forecast_df: pd.DataFrame,
    price_delta_pct: float,
    value_cols: tuple[str, str, str] = ("p10", "p50", "p90"),
) -> ScenarioResult:
    """
    Phase B: Apply a price scenario to a forecast table

    Inputs
    - forecast_df: must include date + quantile columns (p10/p50/p90).
      It can be store-level or store+dept level; we leave IDs untouched.
    - price_delta_pct: e.g., +5 means +5% price; -10 means -10% price.
    - value_cols: which columns represent uncertainty bands.

    Core assumption (explicit):
    - We treat price change as a multiplicative adjustment to revenue forecasts:
        scenario = baseline * (1 + delta)
      This is a planning "projection", not a causal demand model.
    """
    df = forecast_df.copy()
    multiplier = 1.0 + (price_delta_pct / 100.0)

    """
    Phase C: Create scenario-adjusted columns
    - Keep baseline columns intact.
    - Add scenario columns so the UI can overlay both.
    """
    scen = df.copy()
    for c in value_cols:
        scen[f"{c}_scenario"] = scen[c] * multiplier

    """
    Phase D: Build a small KPI summary for dashboard delta cards
    - Total baseline vs scenario (using p50 as default "expected" trajectory)
    """
    baseline_total = float(df[value_cols[1]].sum())
    scenario_total = float(scen[f"{value_cols[1]}_scenario"].sum())
    delta_total = scenario_total - baseline_total

    summary = {
        "price_delta_pct": float(price_delta_pct),
        "multiplier": float(multiplier),
        "baseline_total_p50": baseline_total,
        "scenario_total_p50": scenario_total,
        "delta_total_p50": delta_total,
    }

    return ScenarioResult(baseline=df, scenario=scen, summary=summary)
