from __future__ import annotations

import pandas as pd
import numpy as np
import joblib

from m5rpc.config.settings import settings


def _feature_cols() -> list[str]:
    """
    Phase A: Feature columns expected by the models
    - Must match what FeatureBuilder produced during training.
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
    Phase C: Load historical feature table (store+dept)
    - We use past engineered features to seed the recursive forecast.
    """
    hist_path = settings.PROCESSED_DIR / "store_dept_day_top5_features.parquet"
    df = pd.read_parquet(hist_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values(["store_id", "dept_id", "date"]).reset_index(drop=True)

    last_date = df["date"].max()
    horizon = settings.HORIZON_DAYS
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    """
    Phase D: Load trained quantile models
    """
    model_dir = settings.ARTIFACTS_DIR / "models" / "store_dept_top5"
    m10 = joblib.load(model_dir / "lgb_q10.joblib")
    m50 = joblib.load(model_dir / "lgb_q50.joblib")
    m90 = joblib.load(model_dir / "lgb_q90.joblib")

    feat_cols = _feature_cols()

    """
    Phase E: Recursive forecasting per (store, dept)
    - Same logic as store scoring, but each series is smaller/noisier.
    - We update history using P50 to maintain a single expected trajectory.
    """
    forecasts = []

    for (store_id, dept_id), g in df.groupby(["store_id", "dept_id"], sort=False):
        g = g.sort_values("date")
        y_hist = g["y"].tolist()

        for dt in future_dates:
            lag_1 = y_hist[-1]
            lag_7 = y_hist[-7] if len(y_hist) >= 7 else np.nan
            lag_28 = y_hist[-28] if len(y_hist) >= 28 else np.nan

            last_7 = y_hist[-7:] if len(y_hist) >= 7 else y_hist
            last_28 = y_hist[-28:] if len(y_hist) >= 28 else y_hist

            roll_mean_7 = float(np.mean(last_7))
            roll_std_7 = float(np.std(last_7, ddof=1)) if len(last_7) > 1 else 0.0

            roll_mean_28 = float(np.mean(last_28))
            roll_std_28 = float(np.std(last_28, ddof=1)) if len(last_28) > 1 else 0.0

            tfeat = _add_time_features(dt)

            row = {
                "store_id": store_id,
                "dept_id": dept_id,
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
                    "dept_id": dept_id,
                    "date": dt,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                }
            )

            y_hist.append(p50)

    out_df = (
        pd.DataFrame(forecasts)
        .sort_values(["store_id", "dept_id", "date"])
        .reset_index(drop=True)
    )

    settings.FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = settings.FORECASTS_DIR / "store_dept_top5_latest.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"✅ Wrote: {out_path} | shape={out_df.shape}")


if __name__ == "__main__":
    main()
