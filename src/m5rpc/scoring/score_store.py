from __future__ import annotations

import pandas as pd
import numpy as np
import joblib

from m5rpc.config.settings import settings


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Phase A: Define the exact feature columns used at training time
    - We stored this list in metadata during training, but to keep scoring simple,
      we use the known set created by FeatureBuilder.
    - If we later change features, we should read them from artifacts/metadata.
    """
    return [
        "dow",
        "week",
        "month",
        "year",
        "lag_1",
        "lag_7",
        "lag_28",
        "roll_mean_7",
        "roll_std_7",
        "roll_mean_28",
        "roll_std_28",
    ]


def _add_time_features(date: pd.Timestamp) -> dict[str, int]:
    """
    Phase B: Future-known time features for a single date.
    """
    return {
        "dow": int(date.dayofweek),
        "week": int(date.isocalendar().week),
        "month": int(date.month),
        "year": int(date.year),
    }


def main() -> None:
    """
    Phase C: Load history features (already engineered for past dates)
    - We take the most recent 28 days of history per store to seed lags/rollings.
    """
    hist_path = settings.PROCESSED_DIR / "store_day_features.parquet"
    df = pd.read_parquet(hist_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values(["store_id", "date"]).reset_index(drop=True)

    last_date = df["date"].max()
    horizon = settings.HORIZON_DAYS
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    """
    Phase D: Load trained quantile models
    """
    model_dir = settings.ARTIFACTS_DIR / "models" / "store"
    m10 = joblib.load(model_dir / "lgb_q10.joblib")
    m50 = joblib.load(model_dir / "lgb_q50.joblib")
    m90 = joblib.load(model_dir / "lgb_q90.joblib")

    feat_cols = _feature_cols(df)

    """
    Phase E: Recursive forecasting per store
    - Why recursion? Our features depend on lag/rolling values that require prior days.
    - We keep a rolling window of the most recent "y" values (history + predictions).
    - We generate P10/P50/P90 for each day, but we update history using P50
      (typical choice for expected trajectory).
    """
    forecasts = []

    for store_id, g in df.groupby("store_id", sort=False):
        g = g.sort_values("date")

        # Seed history with the full y series so we can access lags and rolling windows
        y_hist = g["y"].tolist()

        for dt in future_dates:
            # Build lag features from current history
            lag_1 = y_hist[-1]
            lag_7 = y_hist[-7] if len(y_hist) >= 7 else np.nan
            lag_28 = y_hist[-28] if len(y_hist) >= 28 else np.nan

            # Rolling features (based on last observed/predicted values)
            last_7 = y_hist[-7:] if len(y_hist) >= 7 else y_hist
            last_28 = y_hist[-28:] if len(y_hist) >= 28 else y_hist

            roll_mean_7 = float(np.mean(last_7))
            roll_std_7 = float(np.std(last_7, ddof=1)) if len(last_7) > 1 else 0.0

            roll_mean_28 = float(np.mean(last_28))
            roll_std_28 = float(np.std(last_28, ddof=1)) if len(last_28) > 1 else 0.0

            tfeat = _add_time_features(dt)

            row = {
                "store_id": store_id,
                "date": dt,
                **tfeat,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_28": lag_28,
                "roll_mean_7": roll_mean_7,
                "roll_std_7": roll_std_7,
                "roll_mean_28": roll_mean_28,
                "roll_std_28": roll_std_28,
            }

            X = pd.DataFrame([row])[feat_cols]

            p10 = float(m10.predict(X)[0])
            p50 = float(m50.predict(X)[0])
            p90 = float(m90.predict(X)[0])

            forecasts.append(
                {
                    "store_id": store_id,
                    "date": dt,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                }
            )

            # Update history with expected trajectory
            y_hist.append(p50)

    out_df = (
        pd.DataFrame(forecasts).sort_values(["store_id", "date"]).reset_index(drop=True)
    )

    settings.FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = settings.FORECASTS_DIR / "store_latest.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"✅ Wrote: {out_path} | shape={out_df.shape}")


if __name__ == "__main__":
    main()
