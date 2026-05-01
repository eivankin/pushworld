from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ASSETS = ROOT / "docs" / "assets"


def load_last_jsonl(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"No rows found in {path}")
    return json.loads(lines[-1])


def load_matching_jsonl(path: Path, **criteria) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise ValueError(f"No matching row in {path} for {criteria}")


def _set_headroom(ax: plt.Axes, top: float) -> None:
    ax.set_ylim(0, top * 1.18)


def _annotate(ax: plt.Axes, bars, labels: list[str], offset: float) -> None:
    for bar, label in zip(bars, labels, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )


def make_figure(output: Path) -> None:
    ppo_rgb = load_last_jsonl(REPORTS / "profile_ppo_rgb_5.jsonl")
    ppo_planes = load_last_jsonl(REPORTS / "profile_ppo_planes_5.jsonl")
    ppo_vec16 = load_matching_jsonl(
        REPORTS / "profile_ppo_planes_timed.jsonl",
        n_envs=16,
        vec_env="dummy",
    )
    dqn_planes = load_last_jsonl(REPORTS / "profile_dqn_planes_5.jsonl")

    env_share_labels = [
        "PPO RGB\n1 env",
        "PPO planes\n1 env",
        "PPO planes\n16 envs",
        "DQN planes\n1 env",
    ]
    env_share_values = [
        100.0 * ppo_rgb["env_measured_seconds"] / ppo_rgb["train_elapsed_seconds"],
        100.0 * ppo_planes["env_measured_seconds"] / ppo_planes["train_elapsed_seconds"],
        100.0 * ppo_vec16["env_measured_seconds"] / ppo_vec16["train_elapsed_seconds"],
        100.0 * dqn_planes["env_measured_seconds"] / dqn_planes["train_elapsed_seconds"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("PushWorld Bottleneck Summary", fontsize=15, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(
        env_share_labels,
        env_share_values,
        color=["#6b7280", "#60a5fa", "#1d4ed8", "#10b981"],
        edgecolor="#111827",
        linewidth=0.8,
    )
    ax.set_title("Measured env work as share of train time", fontsize=12)
    ax.set_ylabel("Percent of train wall time")
    ax.grid(axis="y", alpha=0.2)
    _set_headroom(ax, max(env_share_values))
    _annotate(ax, bars, [f"{value:.2f}%" for value in env_share_values], max(env_share_values) * 0.04)

    ax = axes[1]
    env_counts = ["4 envs", "16 envs"]
    rollout_pct = [
        100.0 * load_matching_jsonl(REPORTS / "profile_ppo_planes_timed.jsonl", n_envs=4, vec_env="dummy")[
            "ppo_rollout_fraction"
        ],
        100.0 * ppo_vec16["ppo_rollout_fraction"],
    ]
    update_pct = [
        100.0 * load_matching_jsonl(REPORTS / "profile_ppo_planes_timed.jsonl", n_envs=4, vec_env="dummy")[
            "ppo_update_fraction"
        ],
        100.0 * ppo_vec16["ppo_update_fraction"],
    ]
    x = range(len(env_counts))
    ax.bar(x, rollout_pct, label="Rollout", color="#93c5fd", edgecolor="#111827")
    ax.bar(
        x,
        update_pct,
        bottom=rollout_pct,
        label="Update",
        color="#1d4ed8",
        edgecolor="#111827",
    )
    ax.set_xticks(list(x), env_counts)
    ax.set_ylim(0, 100)
    ax.set_title("PPO rollout vs update time", fontsize=12)
    ax.set_ylabel("Share of train wall time")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
    )
    for idx, (rollout, update) in enumerate(zip(rollout_pct, update_pct, strict=True)):
        ax.text(idx, rollout / 2, f"{rollout:.0f}%", ha="center", va="center", fontsize=9)
        ax.text(idx, rollout + update / 2, f"{update:.0f}%", ha="center", va="center", fontsize=9, color="white")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    make_figure(ASSETS / "bottleneck_summary.png")
