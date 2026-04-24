from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal
import warnings

import gymnasium as gym
import numpy as np

from pushworld_study.envs import make_pushworld_env
from pushworld_study.paths import ensure_upstream_pushworld_on_path


Algorithm = Literal["ppo", "dqn"]


@dataclass(frozen=True)
class BaselineResult:
    algorithm: Algorithm
    total_timesteps: int
    model_path: Path


@dataclass(frozen=True)
class EvalResult:
    algorithm: Algorithm
    episodes: int
    successes: int
    mean_reward: float
    mean_length: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


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
    learning_rate: float | None = None,
    ent_coef: float | None = None,
    n_steps: int = 128,
    batch_size: int | None = None,
    n_epochs: int = 2,
):
    _require_sb3()

    from stable_baselines3 import DQN, PPO

    kwargs = policy_kwargs(env.observation_space)

    if algorithm == "ppo":
        kwargs["net_arch"] = {"pi": [128], "vf": [128]}
        return PPO(
            "CnnPolicy",
            env,
            learning_rate=2e-4 if learning_rate is None else learning_rate,
            ent_coef=0.01 if ent_coef is None else ent_coef,
            n_epochs=n_epochs,
            n_steps=n_steps,
            batch_size=32 if batch_size is None else batch_size,
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
            learning_rate=1e-4 if learning_rate is None else learning_rate,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.05,
            exploration_fraction=0.0,
            buffer_size=2_000,
            learning_starts=100,
            batch_size=256 if batch_size is None else batch_size,
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
    eval_puzzle_path: str | Path | None = None,
    eval_freq: int = 0,
    n_eval_episodes: int = 20,
    eval_deterministic: bool = True,
    learning_rate: float | None = None,
    ent_coef: float | None = None,
    n_steps: int = 128,
    batch_size: int | None = None,
    n_epochs: int = 2,
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
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    callback = None
    if eval_puzzle_path is not None and eval_freq > 0:
        from stable_baselines3.common.callbacks import EvalCallback

        eval_env = make_training_env(
            puzzle_path=eval_puzzle_path,
            seed=seed + 10_000,
        )
        best_model_dir = model_dir / f"{algorithm}_best_seed{seed}_{total_timesteps}"
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(log_dir / "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=eval_deterministic,
            render=False,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{algorithm}_smoke_seed{seed}_{total_timesteps}.zip"
    model.save(model_path)
    env.close()

    return BaselineResult(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        model_path=model_path,
    )


def _iter_puzzle_files(puzzle_path: str | Path) -> list[Path]:
    ensure_upstream_pushworld_on_path()

    from pushworld.config import PUZZLE_EXTENSION
    from pushworld.utils.filesystem import iter_files_with_extension

    return sorted(
        Path(path) for path in iter_files_with_extension(str(puzzle_path), PUZZLE_EXTENSION)
    )


def evaluate_baseline(
    algorithm: Algorithm,
    model_path: str | Path,
    puzzle_path: str | Path,
    max_episodes: int | None = None,
    repeat_episodes: int = 1,
    max_steps: int = 100,
    deterministic: bool = True,
    device: str = "cpu",
) -> EvalResult:
    _require_sb3()
    if device == "cpu":
        _configure_cpu_runtime()

    from stable_baselines3 import DQN, PPO

    model_cls = PPO if algorithm == "ppo" else DQN
    model = model_cls.load(str(model_path), device=device)

    puzzle_files = _iter_puzzle_files(puzzle_path)
    if not puzzle_files:
        raise ValueError(f"No puzzle files found in: {puzzle_path}")
    if repeat_episodes < 1:
        raise ValueError("repeat_episodes must be >= 1")

    eval_puzzle_files = [
        puzzle_file for puzzle_file in puzzle_files for _ in range(repeat_episodes)
    ]
    if max_episodes is not None:
        eval_puzzle_files = eval_puzzle_files[:max_episodes]

    successes = 0
    total_reward = 0.0
    total_length = 0

    for episode_idx, puzzle_file in enumerate(eval_puzzle_files):
        env = make_pushworld_env(
            puzzle_path=puzzle_file,
            max_steps=max_steps,
            channel_first=True,
            uint8_observation=True,
        )
        observation, _ = env.reset(seed=episode_idx)
        episode_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            episode_reward += reward
            episode_length += 1

        successes += int(terminated)
        total_reward += episode_reward
        total_length += episode_length
        env.close()

    episodes = len(eval_puzzle_files)
    return EvalResult(
        algorithm=algorithm,
        episodes=episodes,
        successes=successes,
        mean_reward=total_reward / episodes,
        mean_length=total_length / episodes,
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
