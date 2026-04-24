from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pushworld_study.baselines import (
    evaluate_baseline,
    profile_env_steps,
    profile_pipeline,
    train_baseline,
)
from pushworld_study.envs import PushWorldGymnasiumEnv


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def smoke_env(puzzle_path: Path | None = None, steps: int = 8, seed: int = 0) -> None:
    env = PushWorldGymnasiumEnv(puzzle_path=puzzle_path, max_steps=100)
    observation, info = env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    total_reward = 0.0
    terminated = False
    truncated = False

    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(
            int(rng.integers(env.action_space.n))
        )
        total_reward += float(reward)
        if terminated or truncated:
            break

    print(f"puzzle={env.puzzle_path}")
    print("api=gymnasium")
    print(f"observation_shape={observation.shape}")
    print(f"actions_taken={env.steps_taken}")
    print(f"total_reward={total_reward:.2f}")
    print(f"terminated={terminated}")
    print(f"truncated={truncated}")
    print(f"info_keys={sorted(info.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pushworld-study")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser(
        "smoke-env",
        help="Import the upstream PushWorld env and step through one puzzle.",
    )
    smoke_parser.add_argument("--puzzle-path", type=Path, default=None)
    smoke_parser.add_argument("--steps", type=int, default=8)
    smoke_parser.add_argument("--seed", type=int, default=0)

    profile_parser = subparsers.add_parser(
        "profile-env",
        help="Measure random-action throughput for the native Gymnasium env.",
    )
    profile_parser.add_argument("--puzzle-path", type=Path, default=None)
    profile_parser.add_argument("--episodes", type=int, default=10)
    profile_parser.add_argument("--max-steps", type=int, default=100)
    profile_parser.add_argument("--seed", type=int, default=0)
    profile_parser.add_argument(
        "--observation-mode",
        choices=["rgb", "planes"],
        default="rgb",
    )

    pipeline_parser = subparsers.add_parser(
        "profile-pipeline",
        help="Profile env, observation, prediction, and short training costs.",
    )
    pipeline_parser.add_argument("algorithm", choices=["ppo", "dqn"])
    pipeline_parser.add_argument("--puzzle-path", type=Path, default=None)
    pipeline_parser.add_argument("--steps", type=int, default=1_000)
    pipeline_parser.add_argument("--max-steps", type=int, default=100)
    pipeline_parser.add_argument("--seed", type=int, default=0)
    pipeline_parser.add_argument("--device", default="cpu")
    pipeline_parser.add_argument("--predict-iterations", type=int, default=200)
    pipeline_parser.add_argument("--conversion-iterations", type=int, default=1_000)
    pipeline_parser.add_argument("--n-envs", type=int, default=1)
    pipeline_parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    pipeline_parser.add_argument("--output", type=Path, default=None)
    pipeline_parser.add_argument(
        "--observation-mode",
        choices=["rgb", "planes"],
        default="rgb",
    )

    train_parser = subparsers.add_parser(
        "train-baseline",
        help="Run a short SB3 PPO/DQN smoke-training baseline.",
    )
    train_parser.add_argument("algorithm", choices=["ppo", "dqn"])
    train_parser.add_argument("--puzzle-path", type=Path, default=None)
    train_parser.add_argument("--total-timesteps", type=int, default=1_000)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    train_parser.add_argument("--model-dir", type=Path, default=Path("models"))
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--eval-puzzle-path", type=Path, default=None)
    train_parser.add_argument("--eval-freq", type=int, default=0)
    train_parser.add_argument("--n-eval-episodes", type=int, default=20)
    train_parser.add_argument("--eval-stochastic", action="store_true")
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--ent-coef", type=float, default=None)
    train_parser.add_argument("--n-steps", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--n-epochs", type=int, default=2)
    train_parser.add_argument("--features-dim", type=int, default=256)
    train_parser.add_argument("--vf-coef", type=float, default=None)
    train_parser.add_argument("--net-arch-pi", type=parse_int_list, default=(128,))
    train_parser.add_argument("--net-arch-vf", type=parse_int_list, default=(128,))
    train_parser.add_argument("--n-envs", type=int, default=1)
    train_parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    train_parser.add_argument(
        "--observation-mode",
        choices=["rgb", "planes"],
        default="rgb",
    )

    eval_parser = subparsers.add_parser(
        "eval-baseline",
        help="Evaluate a saved PPO/DQN model on each puzzle in a path.",
    )
    eval_parser.add_argument("algorithm", choices=["ppo", "dqn"])
    eval_parser.add_argument("model_path", type=Path)
    eval_parser.add_argument("--puzzle-path", type=Path, required=True)
    eval_parser.add_argument("--max-episodes", type=int, default=None)
    eval_parser.add_argument("--repeat-episodes", type=int, default=1)
    eval_parser.add_argument("--max-steps", type=int, default=100)
    eval_parser.add_argument("--stochastic", action="store_true")
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument(
        "--observation-mode",
        choices=["rgb", "planes"],
        default="rgb",
    )

    args = parser.parse_args()

    if args.command == "smoke-env":
        smoke_env(args.puzzle_path, args.steps, args.seed)
    elif args.command == "profile-env":
        metrics = profile_env_steps(
            puzzle_path=args.puzzle_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            observation_mode=args.observation_mode,
        )
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}={value:.4f}")
            else:
                print(f"{key}={value}")
    elif args.command == "profile-pipeline":
        metrics = profile_pipeline(
            algorithm=args.algorithm,
            puzzle_path=args.puzzle_path,
            steps=args.steps,
            max_steps=args.max_steps,
            seed=args.seed,
            observation_mode=args.observation_mode,
            device=args.device,
            predict_iterations=args.predict_iterations,
            conversion_iterations=args.conversion_iterations,
            n_envs=args.n_envs,
            vec_env=args.vec_env,
            output=args.output,
        )
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}={value:.6f}")
            else:
                print(f"{key}={value}")
    elif args.command == "train-baseline":
        result = train_baseline(
            algorithm=args.algorithm,
            puzzle_path=args.puzzle_path,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            device=args.device,
            eval_puzzle_path=args.eval_puzzle_path,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            eval_deterministic=not args.eval_stochastic,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            observation_mode=args.observation_mode,
            features_dim=args.features_dim,
            vf_coef=args.vf_coef,
            net_arch_pi=args.net_arch_pi,
            net_arch_vf=args.net_arch_vf,
            n_envs=args.n_envs,
            vec_env=args.vec_env,
        )
        print(f"algorithm={result.algorithm}")
        print(f"total_timesteps={result.total_timesteps}")
        print(f"model_path={result.model_path}")
    elif args.command == "eval-baseline":
        result = evaluate_baseline(
            algorithm=args.algorithm,
            model_path=args.model_path,
            puzzle_path=args.puzzle_path,
            max_episodes=args.max_episodes,
            repeat_episodes=args.repeat_episodes,
            max_steps=args.max_steps,
            deterministic=not args.stochastic,
            device=args.device,
            observation_mode=args.observation_mode,
        )
        print(f"algorithm={result.algorithm}")
        print(f"episodes={result.episodes}")
        print(f"successes={result.successes}")
        print(f"success_rate={result.success_rate:.4f}")
        print(f"mean_reward={result.mean_reward:.4f}")
        print(f"mean_length={result.mean_length:.4f}")
