from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NESTED_UPSTREAM_ROOT = PROJECT_ROOT / "external" / "pushworld"
LOCAL_UPSTREAM_ROOT = PROJECT_ROOT
UPSTREAM_ROOT = NESTED_UPSTREAM_ROOT if (NESTED_UPSTREAM_ROOT / "python3" / "src").exists() else LOCAL_UPSTREAM_ROOT
UPSTREAM_PYTHON_SRC = UPSTREAM_ROOT / "python3" / "src"
BENCHMARK_PUZZLES = UPSTREAM_ROOT / "benchmark" / "puzzles"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))


def ensure_upstream_pushworld_on_path() -> None:
    """Expose the upstream PushWorld Python package without editing the submodule."""
    src_path = str(UPSTREAM_PYTHON_SRC)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def default_smoke_puzzle() -> Path:
    return BENCHMARK_PUZZLES / "level1" / "Simple Tool.pwp"
