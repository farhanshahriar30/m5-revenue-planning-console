from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    """
    Streamlit Cloud runs the app with `app/` on sys.path.
    This function adds `<repo_root>/src` so `import m5rpc` works.
    """
    repo_root = Path(__file__).resolve().parent.parent  # .../app -> repo root
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
