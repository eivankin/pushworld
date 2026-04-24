from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal
import warnings

import gymnasium as gym
import numpy as np

from pushworld_study.envs import make_pushworld_env


Algorithm = Literal["ppo", "dqn"]


@dataclass(frozen=True)
class BaselineResult:
    algorithm: Algorithm
    total_timesteps: int
    model_path: Path


def _require_sb3() -> None:
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "RL dependencies are not installed. Run `uv sync --group rl` first."
        ) from exc


def _configure_cpu_runtime() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    warnings.filterwarnings(
        "ignore",
        message="CUDA initialization:.*",
        category=UserWarning,
    )


def make_training_env(
    puzzle_path: str | Path | None = None,
    max_steps: int = 100,
    seed: int = 0,
) -> gym.Env:
    _require_sb3()

    from stable_baselines3.common.monitor import Monitor

    env = make_pushworld_env(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
        channel_first=True,
        uint8_observation=True,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def policy_kwargs(observation_space: gym.spaces.Box) -> dict:
    _require_sb3()

    from pushworld_study.models import PushWorldCNN

    return {
        "features_extractor_class": PushWorldCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": True,
    }


def make_model(
    algorithm: Algorithm,
    env: gym.Env,
    seed: int = 0,
    tensorboard_log: Path | None = None,
    device: str = "auto",
):
    _require_sb3()

    from stable_baselines3 import DQN, PPO

    kwargs = policy_kwargs(env.observation_space)

    if algorithm == "ppo":
        kwargs["net_arch"] = {"pi": [128], "vf": [128]}
        return PPO(
            "CnnPolicy",
            env,
            learning_rate=2e-4,
            ent_coef=0.01,
            n_epochs=2,
            n_steps=128,
            batch_size=32,
            seed=seed,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            policy_kwargs=kwargs,
            device=device,
            verbose=1,
        )

    if algorithm == "dqn":
        kwargs["net_arch"] = [128]
        return DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.05,
            exploration_fraction=0.0,
            buffer_size=2_000,
            learning_starts=100,
            batch_size=256,
            gamma=1.0,
            train_freq=1,
            gradient_steps=1,
            seed=seed,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            policy_kwargs=kwargs,
            device=device,
            verbose=1,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def train_baseline(
    algorithm: Algorithm,
    puzzle_path: str | Path | None = None,
    total_timesteps: int = 1_000,
    seed: int = 0,
    log_dir: Path = Path("runs"),
    model_dir: Path = Path("models"),
    device: str = "auto",
) -> BaselineResult:
    if device == "cpu":
        _configure_cpu_runtime()

    env = make_training_env(puzzle_path=puzzle_path, seed=seed)
    model = make_model(
        algorithm=algorithm,
        env=env,
        seed=seed,
        tensorboard_log=log_dir,
        device=device,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{algorithm}_smoke_seed{seed}_{total_timesteps}.zip"
    model.save(model_path)
    env.close()

    return BaselineResult(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        model_path=model_path,
    )


def profile_env_steps(
    puzzle_path: str | Path | None = None,
    episodes: int = 10,
    max_steps: int = 100,
    seed: int = 0,
) -> dict[str, float | int]:
    env = make_pushworld_env(puzzle_path=puzzle_path, max_steps=max_steps)
    rng = np.random.default_rng(seed)

    import time

    total_steps = 0
    total_reward = 0.0
    started_at = time.perf_counter()

    for episode_idx in range(episodes):
        _, _ = env.reset(seed=seed + episode_idx)
        for _ in range(max_steps):
            _, reward, terminated, truncated, _ = env.step(
                int(rng.integers(env.action_space.n))
            )
            total_steps += 1
            total_reward += reward
            if terminated or truncated:
                break

    elapsed_seconds = time.perf_counter() - started_at
    env.close()
    return {
        "episodes": episodes,
        "steps": total_steps,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": total_steps / elapsed_seconds,
        "mean_reward_per_episode": total_reward / episodes,
    }
