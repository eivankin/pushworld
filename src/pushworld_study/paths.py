from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_upstream_root() -> Path:
    candidates = (
        PROJECT_ROOT / "external" / "pushworld" / "external" / "pushworld",
        PROJECT_ROOT / "external" / "pushworld",
        PROJECT_ROOT,
    )
    for candidate in candidates:
        if (candidate / "python3" / "src" / "pushworld" / "puzzle.py").exists():
            return candidate
    return PROJECT_ROOT / "external" / "pushworld"


UPSTREAM_ROOT = _find_upstream_root()
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
