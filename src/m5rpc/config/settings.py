from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # ---------- Core project parameters ----------
    HORIZON_DAYS: int = 28
    TOP_DEPTS_PER_STORE: int = 5

    # ---------- Project paths ----------
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    OUTPUTS_DIR: Path = DATA_DIR / "outputs"

    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    FORECASTS_DIR: Path = PROJECT_ROOT / "forecasts"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"


settings = Settings()
