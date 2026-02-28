from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from m5rpc.config.settings import settings


def _time_split(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase A: Time-based split (no leakage)
    - We hold out the last `horizon` days as a test window.
    - Because this dataset contains many store-dept series, we split by date globally.
    """
    df = df.sort_values(["store_id", "dept_id", "date"]).reset_index(drop=True)
    last_date = df["date"].max()
    cutoff = last_date - pd.Timedelta(days=horizon)

    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Phase B: Feature selection
    - Drop identifiers and target columns.
    - Keep engineered lag/rolling features and known future calendar features.
    """
    drop = {"date", "store_id", "dept_id", "dept_revenue", "y"}
    return [c for c in df.columns if c not in drop]


def _train_quantile_model(
    X: pd.DataFrame, y: pd.Series, alpha: float
) -> lgb.LGBMRegressor:
    """
    Phase C: Quantile regression (P10/P50/P90)
    - Separate model per quantile using LightGBM's quantile objective.
    """
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Phase D: Metric
    - sMAPE is common for retail-style forecasting and behaves well across scales.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def main() -> None:
    """
    Phase E: Load features + drop warm-up NaNs
    - Early rows per series lack enough history for lag/rolling features.
    """
    in_path = settings.PROCESSED_DIR / "store_dept_day_top5_features.parquet"
    df = pd.read_parquet(in_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().reset_index(drop=True)

    """
    Phase F: Split train/test
    """
    horizon = settings.HORIZON_DAYS
    train_df, test_df = _time_split(df, horizon=horizon)

    feat_cols = _get_feature_cols(df)
    X_train, y_train = train_df[feat_cols], train_df["y"]
    X_test, y_test = test_df[feat_cols], test_df["y"]

    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Num features: {len(feat_cols)}")

    """
    Phase G: Train quantile models and save artifacts
    """
    out_dir = settings.ARTIFACTS_DIR / "models" / "store_dept_top5"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for alpha in (0.1, 0.5, 0.9):
        print(f"Training quantile model alpha={alpha} ...")
        m = _train_quantile_model(X_train, y_train, alpha=alpha)
        models[alpha] = m
        joblib.dump(m, out_dir / f"lgb_q{int(alpha * 100):02d}.joblib")

    """
    Phase H: Quick evaluation on P50
    - This gives us a sanity check before we invest in rolling backtests.
    """
    p50 = models[0.5].predict(X_test)
    mae = float(np.mean(np.abs(y_test.values - p50)))
    smape = _smape(y_test.values, p50)

    metrics = {
        "horizon_days": horizon,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "mae_p50": mae,
        "smape_p50": smape,
        "feature_cols": feat_cols,
        "split_cutoff_date": str(test_df["date"].min().date()),
    }

    meta_dir = settings.ARTIFACTS_DIR / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "store_dept_train_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("✅ Saved models + metrics")
    print(metrics)


if __name__ == "__main__":
    main()
