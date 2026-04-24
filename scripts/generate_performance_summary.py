from __future__ import annotations

import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
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


def load_tb_mean_fps(run_dir: Path) -> float:
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalars = accumulator.Scalars("time/fps")
    if not scalars:
        raise ValueError(f"No time/fps scalars in {run_dir}")
    return statistics.fmean(item.value for item in scalars)


def annotate_bars(
    ax: plt.Axes,
    bars,
    labels: list[str],
    y_offset: float = 0.0,
) -> None:
    for bar, label in zip(bars, labels, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )


def set_headroom(ax: plt.Axes, values: list[float], fraction: float = 0.18) -> None:
    upper = max(values) * (1 + fraction)
    ax.set_ylim(0, upper)


def make_figure(output: Path) -> None:
    ppo_rgb = load_last_jsonl(REPORTS / "profile_ppo_rgb_5.jsonl")
    ppo_planes = load_last_jsonl(REPORTS / "profile_ppo_planes_5.jsonl")
    ppo_vec = [
        load_matching_jsonl(REPORTS / "profile_ppo_planes_vec.jsonl", n_envs=4, vec_env="dummy"),
        load_matching_jsonl(REPORTS / "profile_ppo_planes_vec.jsonl", n_envs=16, vec_env="dummy"),
    ]
    dqn_rgb_fps = load_tb_mean_fps(RUNS / "DQN_4")
    dqn_planes = load_last_jsonl(REPORTS / "profile_dqn_planes_5.jsonl")
    ppo_timed = [
        load_matching_jsonl(REPORTS / "profile_ppo_planes_timed.jsonl", n_envs=4, vec_env="dummy"),
        load_matching_jsonl(REPORTS / "profile_ppo_planes_timed.jsonl", n_envs=16, vec_env="dummy"),
    ]
    level_profiles = [
        ("Level 0", ppo_planes["env_measured_steps_per_second"]),
        ("Level 1", load_last_jsonl(REPORTS / "profile_ppo_planes_level1.jsonl")["env_measured_steps_per_second"]),
        ("Level 2", load_last_jsonl(REPORTS / "profile_ppo_planes_level2.jsonl")["env_measured_steps_per_second"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PushWorld Performance Summary", fontsize=16, fontweight="bold")

    # PPO throughput
    ax = axes[0, 0]
    ppo_labels = ["RGB\n1 env", "Planes\n1 env", "Planes\n4 envs", "Planes\n16 envs"]
    ppo_values = [
        ppo_rgb["train_steps_per_second"],
        ppo_planes["train_steps_per_second"],
        ppo_vec[0]["train_steps_per_second"],
        ppo_vec[1]["train_steps_per_second"],
    ]
    ppo_speedups = [value / ppo_values[0] for value in ppo_values]
    bars = ax.bar(
        ppo_labels,
        ppo_values,
        color=["#6b7280", "#60a5fa", "#2563eb", "#1d4ed8"],
        edgecolor="#111827",
        linewidth=0.8,
    )
    ax.set_title("PPO throughput", fontsize=12)
    ax.set_ylabel("Training FPS")
    ax.grid(axis="y", alpha=0.2)
    set_headroom(ax, ppo_values)
    annotate_bars(
        ax,
        bars,
        [f"{value:.0f}\n{speedup:.1f}x" for value, speedup in zip(ppo_values, ppo_speedups, strict=True)],
        y_offset=max(ppo_values) * 0.02,
    )

    # DQN throughput
    ax = axes[0, 1]
    dqn_labels = ["RGB\n1 env", "Planes\n1 env"]
    dqn_values = [dqn_rgb_fps, dqn_planes["train_steps_per_second"]]
    dqn_speedups = [value / dqn_values[0] for value in dqn_values]
    bars = ax.bar(
        dqn_labels,
        dqn_values,
        color=["#6b7280", "#10b981"],
        edgecolor="#111827",
        linewidth=0.8,
    )
    ax.set_title("DQN throughput", fontsize=12)
    ax.set_ylabel("Training FPS")
    ax.grid(axis="y", alpha=0.2)
    set_headroom(ax, dqn_values)
    annotate_bars(
        ax,
        bars,
        [f"{value:.1f}\n{speedup:.1f}x" for value, speedup in zip(dqn_values, dqn_speedups, strict=True)],
        y_offset=max(dqn_values) * 0.05,
    )

    # PPO rollout/update split
    ax = axes[1, 0]
    env_counts = ["4 envs", "16 envs"]
    rollout_pct = [row["ppo_rollout_fraction"] * 100 for row in ppo_timed]
    update_pct = [row["ppo_update_fraction"] * 100 for row in ppo_timed]
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
    ax.set_title("PPO time split", fontsize=12)
    ax.set_ylabel("Share of train time")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
    )
    for idx, (r_pct, u_pct) in enumerate(zip(rollout_pct, update_pct, strict=True)):
        ax.text(
            idx,
            r_pct / 2,
            f"{r_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
            clip_on=False,
        )
        ax.text(
            idx,
            r_pct + u_pct / 2,
            f"{u_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            clip_on=False,
        )

    # Level complexity env throughput
    ax = axes[1, 1]
    level_labels = [item[0] for item in level_profiles]
    level_values = [item[1] for item in level_profiles]
    bars = ax.bar(
        level_labels,
        level_values,
        color=["#34d399", "#f59e0b", "#ef4444"],
        edgecolor="#111827",
        linewidth=0.8,
    )
    ax.set_title("Plane env throughput by level", fontsize=12)
    ax.set_ylabel("Env steps/sec")
    ax.grid(axis="y", alpha=0.2)
    set_headroom(ax, level_values)
    annotate_bars(
        ax,
        bars,
        [f"{value:.0f}" for value in level_values],
        y_offset=max(level_values) * 0.03,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    make_figure(ASSETS / "performance_summary.png")
