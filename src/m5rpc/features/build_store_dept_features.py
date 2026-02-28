from __future__ import annotations

import pandas as pd

from m5rpc.config.settings import settings
from m5rpc.features.feature_builder import FeatureBuilder


def main() -> None:
    """
    Phase A: Load the store-dept-day revenue table (top 5 depts per store)
    - This is our drill-down dataset for the dashboard.
    """
    in_path = settings.PROCESSED_DIR / "store_dept_day_top5.parquet"
    df = pd.read_parquet(in_path)

    """
    Phase B: Build features
    - group_cols=["store_id","dept_id"] so lags/rollings are computed per store-dept series.
    - target_col="dept_revenue"
    """
    fb = FeatureBuilder(target_col="dept_revenue", group_cols=["store_id", "dept_id"])
    feat = fb.build(df)

    """
    Phase C: Write output for modeling
    """
    out_path = settings.PROCESSED_DIR / "store_dept_day_top5_features.parquet"
    feat.to_parquet(out_path, index=False)
    print(f"✅ Wrote: {out_path} | shape={feat.shape}")


if __name__ == "__main__":
    main()
