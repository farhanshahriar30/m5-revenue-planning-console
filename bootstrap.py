# bootstrap.py
from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    """
    Adds <repo_root>/src to PYTHONPATH so `import m5rpc` works on Streamlit Cloud.
    Safe to call multiple times.
    """
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"

    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
