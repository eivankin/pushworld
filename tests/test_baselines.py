from __future__ import annotations

from pushworld_study.baselines import (
    make_vector_training_env,
    profile_env_steps,
    profile_pipeline,
)
from pushworld_study.envs import make_pushworld_env
from pushworld_study.model_benchmarks import benchmark_model_compile


def test_channel_first_env_shape() -> None:
    env = make_pushworld_env(channel_first=True)
    observation, _ = env.reset(seed=0)
    assert observation.shape[0] == 3
    assert observation in env.observation_space


def test_channel_first_uint8_env_shape() -> None:
    env = make_pushworld_env(channel_first=True, uint8_observation=True)
    observation, _ = env.reset(seed=0)
    assert observation.shape[0] == 3
    assert observation.dtype.name == "uint8"
    assert observation in env.observation_space


def test_plane_env_shape() -> None:
    env = make_pushworld_env(observation_mode="planes")
    observation, _ = env.reset(seed=0)
    assert observation.shape[0] == 6
    assert observation.dtype.name == "float32"
    assert observation in env.observation_space


def test_profile_env_steps_smoke() -> None:
    metrics = profile_env_steps(episodes=1, max_steps=3, seed=0)
    assert metrics["episodes"] == 1
    assert metrics["steps"] == 3
    assert metrics["steps_per_second"] > 0


def test_profile_plane_env_steps_smoke() -> None:
    metrics = profile_env_steps(
        episodes=1,
        max_steps=3,
        seed=0,
        observation_mode="planes",
    )
    assert metrics["episodes"] == 1
    assert metrics["steps"] == 3
    assert metrics["steps_per_second"] > 0


def test_profile_pipeline_smoke(tmp_path) -> None:
    output = tmp_path / "profile.jsonl"
    metrics = profile_pipeline(
        algorithm="dqn",
        steps=4,
        predict_iterations=2,
        conversion_iterations=2,
        observation_mode="planes",
        output=output,
    )
    assert metrics["algorithm"] == "dqn"
    assert metrics["env_steps"] == 4
    assert metrics["predict_iterations"] == 2
    assert metrics["train_steps_per_second"] > 0
    assert output.read_text(encoding="utf-8").strip()


def test_profile_pipeline_ppo_timing_smoke() -> None:
    metrics = profile_pipeline(
        algorithm="ppo",
        steps=32,
        predict_iterations=1,
        conversion_iterations=1,
        observation_mode="planes",
    )
    assert metrics["ppo_rollout_calls"] >= 1
    assert metrics["ppo_update_calls"] >= 1
    assert metrics["ppo_rollout_seconds"] > 0
    assert metrics["ppo_update_seconds"] > 0


def test_profile_pipeline_dqn_timing_smoke() -> None:
    metrics = profile_pipeline(
        algorithm="dqn",
        steps=128,
        predict_iterations=1,
        conversion_iterations=1,
        observation_mode="planes",
    )
    assert metrics["dqn_rollout_calls"] >= 1
    assert metrics["dqn_update_calls"] >= 1
    assert metrics["dqn_rollout_seconds"] > 0
    assert metrics["dqn_update_seconds"] > 0


def test_dummy_vector_training_env_smoke() -> None:
    env = make_vector_training_env(
        observation_mode="planes",
        n_envs=2,
        vec_env="dummy",
    )
    observation = env.reset()
    assert observation.shape[0] == 2
    env.close()


def test_benchmark_model_compile_smoke() -> None:
    metrics = benchmark_model_compile(
        observation_mode="planes",
        device="cpu",
        batch_size=4,
        iterations=2,
        warmup=1,
    )
    assert metrics["observation_mode"] == "planes"
    assert metrics["batch_size"] == 4
    assert metrics["eager_infer_steps_per_second"] > 0
    assert metrics["eager_train_steps_per_second"] > 0
    if metrics["compile_success"] == 1:
        assert metrics["compiled_infer_steps_per_second"] > 0
        assert metrics["compiled_train_steps_per_second"] > 0
    else:
        assert metrics["compile_error"]
