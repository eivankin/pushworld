from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Literal
import warnings

import gymnasium as gym
import numpy as np

from pushworld_study.envs import make_pushworld_env
from pushworld_study.envs import ObservationMode
from pushworld_study.paths import ensure_upstream_pushworld_on_path


Algorithm = Literal["ppo", "dqn"]
VecEnvType = Literal["dummy", "subproc"]


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
    observation_mode: ObservationMode = "rgb",
) -> gym.Env:
    _require_sb3()

    from stable_baselines3.common.monitor import Monitor

    if observation_mode == "rgb":
        env = make_pushworld_env(
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            channel_first=True,
            uint8_observation=True,
            observation_mode=observation_mode,
        )
    else:
        env = make_pushworld_env(
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            observation_mode=observation_mode,
        )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_training_env_fn(
    puzzle_path: str | Path | None = None,
    max_steps: int = 100,
    seed: int = 0,
    observation_mode: ObservationMode = "rgb",
):
    def init_env() -> gym.Env:
        return make_training_env(
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            seed=seed,
            observation_mode=observation_mode,
        )

    return init_env


def make_vector_training_env(
    puzzle_path: str | Path | None = None,
    max_steps: int = 100,
    seed: int = 0,
    observation_mode: ObservationMode = "rgb",
    n_envs: int = 1,
    vec_env: VecEnvType = "dummy",
):
    _require_sb3()
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    if n_envs == 1:
        return make_training_env(
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            seed=seed,
            observation_mode=observation_mode,
        )

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [
        make_training_env_fn(
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            seed=seed + env_idx,
            observation_mode=observation_mode,
        )
        for env_idx in range(n_envs)
    ]
    if vec_env == "dummy":
        return DummyVecEnv(env_fns)
    if vec_env == "subproc":
        return SubprocVecEnv(env_fns, start_method="fork")
    raise ValueError(f"Unknown vector env type: {vec_env}")


def policy_kwargs(observation_space: gym.spaces.Box, features_dim: int = 256) -> dict:
    _require_sb3()

    from pushworld_study.models import PushWorldCNN

    normalize_images = observation_space.dtype == np.uint8
    return {
        "features_extractor_class": PushWorldCNN,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "normalize_images": normalize_images,
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
    features_dim: int = 256,
    vf_coef: float | None = None,
    net_arch_pi: tuple[int, ...] = (128,),
    net_arch_vf: tuple[int, ...] = (128,),
    ppo_cls=None,
):
    _require_sb3()

    from stable_baselines3 import DQN, PPO

    kwargs = policy_kwargs(env.observation_space, features_dim=features_dim)

    if algorithm == "ppo":
        ppo_model_cls = PPO if ppo_cls is None else ppo_cls
        kwargs["net_arch"] = {"pi": list(net_arch_pi), "vf": list(net_arch_vf)}
        return ppo_model_cls(
            "CnnPolicy",
            env,
            learning_rate=2e-4 if learning_rate is None else learning_rate,
            ent_coef=0.01 if ent_coef is None else ent_coef,
            n_epochs=n_epochs,
            n_steps=n_steps,
            batch_size=32 if batch_size is None else batch_size,
            vf_coef=0.5 if vf_coef is None else vf_coef,
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
    observation_mode: ObservationMode = "rgb",
    features_dim: int = 256,
    vf_coef: float | None = None,
    net_arch_pi: tuple[int, ...] = (128,),
    net_arch_vf: tuple[int, ...] = (128,),
    n_envs: int = 1,
    vec_env: VecEnvType = "dummy",
) -> BaselineResult:
    if device == "cpu":
        _configure_cpu_runtime()

    env = make_vector_training_env(
        puzzle_path=puzzle_path,
        seed=seed,
        observation_mode=observation_mode,
        n_envs=n_envs,
        vec_env=vec_env,
    )
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
        features_dim=features_dim,
        vf_coef=vf_coef,
        net_arch_pi=net_arch_pi,
        net_arch_vf=net_arch_vf,
    )
    callback = None
    if eval_puzzle_path is not None and eval_freq > 0:
        from stable_baselines3.common.callbacks import EvalCallback

        eval_env = make_training_env(
            puzzle_path=eval_puzzle_path,
            seed=seed + 10_000,
            observation_mode=observation_mode,
        )
        best_model_dir = model_dir / f"{algorithm}_best_seed{seed}_{total_timesteps}"
        effective_eval_freq = max(eval_freq // n_envs, 1)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(log_dir / "eval"),
            eval_freq=effective_eval_freq,
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
    observation_mode: ObservationMode = "rgb",
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
            channel_first=observation_mode == "rgb",
            uint8_observation=observation_mode == "rgb",
            observation_mode=observation_mode,
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
    observation_mode: ObservationMode = "rgb",
) -> dict[str, float | int]:
    env = make_pushworld_env(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
        observation_mode=observation_mode,
    )
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


def _profile_unwrapped_env_components(
    puzzle_path: str | Path | None,
    max_steps: int,
    seed: int,
    observation_mode: ObservationMode,
    steps: int,
) -> dict[str, float | int | str | tuple[int, ...]]:
    from pushworld_study.envs import PushWorldGymnasiumEnv

    env = PushWorldGymnasiumEnv(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
        observation_mode=observation_mode,
    )
    rng = np.random.default_rng(seed)

    reset_seconds = 0.0
    transition_seconds = 0.0
    observe_seconds = 0.0
    total_reward = 0.0
    episodes = 0
    observation_shape: tuple[int, ...] | None = None

    def timed_reset(reset_seed: int) -> None:
        nonlocal reset_seconds, episodes, observation_shape
        started_at = time.perf_counter()
        observation, _ = env.reset(seed=reset_seed)
        reset_seconds += time.perf_counter() - started_at
        observation_shape = observation.shape
        episodes += 1

    timed_reset(seed)

    for step_idx in range(steps):
        if env._current_puzzle is None or env._current_state is None:  # noqa: SLF001
            raise RuntimeError("Profiler expected an initialized environment.")

        action = int(rng.integers(env.action_space.n))
        previous_state = env._current_state  # noqa: SLF001

        started_at = time.perf_counter()
        env._steps += 1  # noqa: SLF001
        next_state = env._current_puzzle.get_next_state(previous_state, action)  # noqa: SLF001
        terminated = env._current_puzzle.is_goal_state(next_state)  # noqa: SLF001
        if terminated:
            reward = 10.0
        else:
            previous_achieved_goals = env._current_puzzle.count_achieved_goals(  # noqa: SLF001
                previous_state
            )
            current_achieved_goals = env._current_puzzle.count_achieved_goals(  # noqa: SLF001
                next_state
            )
            reward = float(current_achieved_goals - previous_achieved_goals - 0.01)
        truncated = env._max_steps is not None and env._steps >= env._max_steps  # noqa: SLF001
        env._current_state = next_state  # noqa: SLF001
        transition_seconds += time.perf_counter() - started_at

        started_at = time.perf_counter()
        observation = env._observe(next_state)  # noqa: SLF001
        observe_seconds += time.perf_counter() - started_at
        observation_shape = observation.shape
        total_reward += reward

        if terminated or truncated:
            timed_reset(seed + step_idx + 1)

    env.close()
    measured_seconds = reset_seconds + transition_seconds + observe_seconds
    return {
        "env_steps": steps,
        "env_episodes": episodes,
        "env_observation_shape": observation_shape or (),
        "env_reset_seconds": reset_seconds,
        "env_transition_seconds": transition_seconds,
        "env_observe_seconds": observe_seconds,
        "env_measured_seconds": measured_seconds,
        "env_measured_steps_per_second": steps / measured_seconds,
        "env_mean_reward_per_step": total_reward / steps,
    }


def _profile_observation_conversion(
    puzzle_path: str | Path | None,
    max_steps: int,
    observation_mode: ObservationMode,
    iterations: int,
) -> dict[str, float | int | str]:
    env = make_pushworld_env(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
        observation_mode=observation_mode,
    )
    observation, _ = env.reset(seed=0)
    env.close()

    started_at = time.perf_counter()
    if observation_mode == "rgb":
        for _ in range(iterations):
            converted = np.rint(np.transpose(observation, (2, 0, 1)) * 255).astype(
                np.uint8
            )
    else:
        for _ in range(iterations):
            converted = np.asarray(observation, dtype=np.float32)
    elapsed_seconds = time.perf_counter() - started_at

    return {
        "conversion_iterations": iterations,
        "conversion_seconds": elapsed_seconds,
        "conversion_per_observation_seconds": elapsed_seconds / iterations,
        "conversion_output_dtype": str(converted.dtype),
        "conversion_output_nbytes": int(converted.nbytes),
    }


def _profile_model_predict(
    algorithm: Algorithm,
    puzzle_path: str | Path | None,
    max_steps: int,
    seed: int,
    observation_mode: ObservationMode,
    device: str,
    iterations: int,
) -> dict[str, float | int]:
    if device == "cpu":
        _configure_cpu_runtime()

    env = make_training_env(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
        seed=seed,
        observation_mode=observation_mode,
    )
    model = make_model(
        algorithm=algorithm,
        env=env,
        seed=seed,
        tensorboard_log=None,
        device=device,
    )
    observation, _ = env.reset(seed=seed)

    started_at = time.perf_counter()
    for _ in range(iterations):
        model.predict(observation, deterministic=True)
    elapsed_seconds = time.perf_counter() - started_at
    env.close()

    return {
        "predict_iterations": iterations,
        "predict_seconds": elapsed_seconds,
        "predict_per_call_seconds": elapsed_seconds / iterations,
        "predict_calls_per_second": iterations / elapsed_seconds,
    }


def _profile_short_training(
    algorithm: Algorithm,
    puzzle_path: str | Path | None,
    max_steps: int,
    seed: int,
    observation_mode: ObservationMode,
    device: str,
    train_steps: int,
    n_envs: int,
    vec_env: VecEnvType,
) -> dict[str, float | int]:
    if device == "cpu":
        _configure_cpu_runtime()

    ppo_cls = None
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        class TimedPPO(PPO):
            rollout_seconds = 0.0
            update_seconds = 0.0
            rollout_calls = 0
            update_calls = 0

            def collect_rollouts(self, *args, **kwargs):
                started_at = time.perf_counter()
                try:
                    return super().collect_rollouts(*args, **kwargs)
                finally:
                    self.rollout_seconds += time.perf_counter() - started_at
                    self.rollout_calls += 1

            def train(self) -> None:
                started_at = time.perf_counter()
                try:
                    return super().train()
                finally:
                    self.update_seconds += time.perf_counter() - started_at
                    self.update_calls += 1

        ppo_cls = TimedPPO

    env = make_vector_training_env(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
        seed=seed,
        observation_mode=observation_mode,
        n_envs=n_envs,
        vec_env=vec_env,
    )
    model = make_model(
        algorithm=algorithm,
        env=env,
        seed=seed,
        tensorboard_log=None,
        device=device,
        n_steps=min(128, train_steps) if algorithm == "ppo" else 128,
        batch_size=32 if algorithm == "ppo" else None,
        ppo_cls=ppo_cls,
    )

    started_at = time.perf_counter()
    model.learn(total_timesteps=train_steps, progress_bar=False)
    elapsed_seconds = time.perf_counter() - started_at
    actual_timesteps = int(model.num_timesteps)
    env.close()

    metrics = {
        "train_requested_steps": train_steps,
        "train_elapsed_seconds": elapsed_seconds,
        "train_steps_per_second": actual_timesteps / elapsed_seconds,
        "train_model_timesteps": actual_timesteps,
        "train_n_envs": n_envs,
        "train_vec_env": vec_env,
    }
    if algorithm == "ppo":
        rollout_seconds = float(getattr(model, "rollout_seconds", 0.0))
        update_seconds = float(getattr(model, "update_seconds", 0.0))
        metrics.update(
            {
                "ppo_rollout_seconds": rollout_seconds,
                "ppo_update_seconds": update_seconds,
                "ppo_rollout_calls": int(getattr(model, "rollout_calls", 0)),
                "ppo_update_calls": int(getattr(model, "update_calls", 0)),
                "ppo_rollout_fraction": rollout_seconds / elapsed_seconds,
                "ppo_update_fraction": update_seconds / elapsed_seconds,
            }
        )
    return metrics


def profile_pipeline(
    algorithm: Algorithm,
    puzzle_path: str | Path | None = None,
    steps: int = 1_000,
    max_steps: int = 100,
    seed: int = 0,
    observation_mode: ObservationMode = "rgb",
    device: str = "cpu",
    predict_iterations: int = 200,
    conversion_iterations: int = 1_000,
    n_envs: int = 1,
    vec_env: VecEnvType = "dummy",
    output: Path | None = None,
) -> dict[str, float | int | str | tuple[int, ...]]:
    _require_sb3()

    metrics: dict[str, float | int | str | tuple[int, ...]] = {
        "algorithm": algorithm,
        "puzzle_path": str(puzzle_path) if puzzle_path is not None else "default",
        "observation_mode": observation_mode,
        "device": device,
        "seed": seed,
        "max_steps": max_steps,
        "n_envs": n_envs,
        "vec_env": vec_env,
    }
    started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    metrics.update(
        _profile_unwrapped_env_components(
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            seed=seed,
            observation_mode=observation_mode,
            steps=steps,
        )
    )
    metrics["env_profile_phase_seconds"] = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    metrics.update(
        _profile_observation_conversion(
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            observation_mode=observation_mode,
            iterations=conversion_iterations,
        )
    )
    metrics["conversion_profile_phase_seconds"] = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    metrics.update(
        _profile_model_predict(
            algorithm=algorithm,
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            seed=seed,
            observation_mode=observation_mode,
            device=device,
            iterations=predict_iterations,
        )
    )
    metrics["predict_profile_phase_seconds"] = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    metrics.update(
        _profile_short_training(
            algorithm=algorithm,
            puzzle_path=puzzle_path,
            max_steps=max_steps,
            seed=seed,
            observation_mode=observation_mode,
            device=device,
            train_steps=steps,
            n_envs=n_envs,
            vec_env=vec_env,
        )
    )
    metrics["train_profile_phase_seconds"] = time.perf_counter() - phase_started_at
    metrics["profile_elapsed_seconds"] = time.perf_counter() - started_at

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")

    return metrics
