from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO

from pushworld_study.envs import make_pushworld_env


def export_rollout_frames(
    algorithm: str,
    model_path: Path,
    puzzle_path: Path,
    output_dir: Path,
    deterministic: bool,
    max_steps: int,
    seed: int,
    title: str,
) -> None:
    model_cls = PPO if algorithm == "ppo" else DQN
    model = model_cls.load(str(model_path), device="cpu")

    env = make_pushworld_env(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
        observation_mode="planes",
    )
    observation, _ = env.reset(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_reward = 0.0
    frame_idx = 0

    def save_frame(rendered: np.ndarray, done_text: str = "") -> None:
        nonlocal frame_idx
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.imshow(rendered)
        ax.set_xticks([])
        ax.set_yticks([])
        subtitle = f"step={frame_idx} reward={total_reward:.2f}"
        if done_text:
            subtitle = f"{subtitle} {done_text}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / f"frame_{frame_idx:03d}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        frame_idx += 1

    save_frame(env.render())

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        done_text = ""
        if terminated:
            done_text = "[solved]"
        elif truncated:
            done_text = "[truncated]"
        save_frame(env.render(), done_text=done_text)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=["ppo", "dqn"])
    parser.add_argument("model_path", type=Path)
    parser.add_argument("puzzle_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", type=str, default="PushWorld rollout")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    export_rollout_frames(
        algorithm=args.algorithm,
        model_path=args.model_path,
        puzzle_path=args.puzzle_path,
        output_dir=args.output_dir,
        deterministic=not args.stochastic,
        max_steps=args.max_steps,
        seed=args.seed,
        title=args.title,
    )


if __name__ == "__main__":
    main()
