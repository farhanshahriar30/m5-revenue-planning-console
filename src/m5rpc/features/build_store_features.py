from __future__ import annotations

import pandas as pd

from m5rpc.config.settings import settings
from m5rpc.features.feature_builder import FeatureBuilder


def main() -> None:
    """
    Phase A: Load the aggregated store-day revenue table
    - This is small and fast compared to the raw item-level table.
    """
    in_path = settings.PROCESSED_DIR / "store_day.parquet"
    df = pd.read_parquet(in_path)

    """
    Phase B: Build features
    - group_cols=["store_id"] so lags/rollings are computed per store.
    - target_col="store_revenue"
    """
    fb = FeatureBuilder(target_col="store_revenue", group_cols=["store_id"])
    feat = fb.build(df)

    """
    Phase C: Write output for modeling
    """
    out_path = settings.PROCESSED_DIR / "store_day_features.parquet"
    feat.to_parquet(out_path, index=False)
    print(f"✅ Wrote: {out_path} | shape={feat.shape}")


if __name__ == "__main__":
    main()
