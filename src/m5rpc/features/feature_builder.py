from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureBuilder:
    """
    Phase A: What this class is for
    - We want one reusable place to create time-series features consistently for:
        1) store_day revenue
        2) store_dept_day_top5 revenue
    - We build "tabular" features from a time series:
        - time features (calendar position)
        - lag features (recent history)
        - rolling stats (recent level + volatility)
    - This is ideal for LightGBM global models.
    """

    target_col: str
    group_cols: list[str]
    date_col: str = "date"

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase B: Input expectations
        - df must contain:
            - date_col (datetime)
            - group_cols (e.g., ["store_id"] or ["store_id","dept_id"])
            - target_col (revenue)
        - Output:
            - same rows, plus feature columns, plus y = target_col
        """
        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col])

        # Phase C: Sort so lags/rollings are correct
        out = out.sort_values(self.group_cols + [self.date_col]).reset_index(drop=True)

        # Phase D: Standardize target name to y (common ML convention)
        out["y"] = out[self.target_col].astype("float32")

        # Phase E: Time features (known in advance)
        # These are "future-known" signals useful for seasonality patterns.
        out["dow"] = out[self.date_col].dt.dayofweek.astype("int8")  # 0=Mon
        out["week"] = out[self.date_col].dt.isocalendar().week.astype("int16")
        out["month"] = out[self.date_col].dt.month.astype("int8")
        out["year"] = out[self.date_col].dt.year.astype("int16")

        # Phase F: Lag + rolling features (history-dependent)
        # We compute within each group (store or store+dept).
        g = out.groupby(self.group_cols, sort=False)["y"]

        # Phase F1: Lags (simple history lookbacks)
        out["lag_1"] = g.shift(1)
        out["lag_7"] = g.shift(7)
        out["lag_28"] = g.shift(28)

        # Phase F2: Rolling stats (level + volatility)
        # Important: shift(1) prevents target leakage (today's y shouldn't influence today's features).
        out["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
        out["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())

        out["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
        out["roll_std_28"] = g.transform(lambda s: s.shift(1).rolling(28).std())

        # Phase G: Final cleanup
        # Early rows don't have enough history -> they will contain NaNs.
        # We keep them for now; the training script will drop rows with NaNs.
        return out
