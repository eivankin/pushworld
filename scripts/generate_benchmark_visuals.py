from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pushworld_study.envs import make_pushworld_env


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def _render_rgb(puzzle_path: Path) -> np.ndarray:
    env = make_pushworld_env(
        puzzle_path=puzzle_path,
        observation_mode="rgb",
        channel_first=False,
        uint8_observation=False,
    )
    observation, _ = env.reset(seed=0)
    env.close()
    return observation


def _render_planes(puzzle_path: Path) -> np.ndarray:
    env = make_pushworld_env(
        puzzle_path=puzzle_path,
        observation_mode="planes",
    )
    observation, _ = env.reset(seed=0)
    env.close()
    return observation


def make_benchmark_overview(output: Path) -> None:
    samples = [
        ("Level 0", ROOT / "data" / "debug" / "base_train_5" / "level_0_base_train_0.pwp"),
        (
            "Level 1",
            ROOT / "external" / "pushworld" / "benchmark" / "puzzles" / "level1" / "A Worthy Sacrifice.pwp",
        ),
        (
            "Level 3",
            ROOT / "external" / "pushworld" / "benchmark" / "puzzles" / "level3" / "Interlock.pwp",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    fig.suptitle("PushWorld benchmark examples", fontsize=16, fontweight="bold")

    for ax, (title, puzzle_path) in zip(axes, samples, strict=True):
        rgb = _render_rgb(puzzle_path)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def make_observation_modes(output: Path) -> None:
    puzzle_path = ROOT / "data" / "debug" / "base_train_5" / "level_0_base_train_0.pwp"
    rgb = _render_rgb(puzzle_path)
    planes = _render_planes(puzzle_path)

    plane_titles = [
        "Walls",
        "Agent blockers",
        "Agent",
        "Goal objects",
        "Other objects",
        "Goals",
    ]

    fig = plt.figure(figsize=(14, 6))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.4, 1, 1, 1])

    ax_rgb = fig.add_subplot(grid[:, 0])
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB observation", fontsize=13)
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])

    for idx, title in enumerate(plane_titles):
        row = idx // 3
        col = idx % 3 + 1
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(planes[idx], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Structured plane observations", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    make_benchmark_overview(ASSETS / "benchmark_overview.png")
    make_observation_modes(ASSETS / "observation_modes.png")
