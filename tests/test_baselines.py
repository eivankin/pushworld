from __future__ import annotations

from pushworld_study.baselines import (
    make_vector_training_env,
    profile_env_steps,
    profile_pipeline,
)
from pushworld_study.envs import make_pushworld_env


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


def test_dummy_vector_training_env_smoke() -> None:
    env = make_vector_training_env(
        observation_mode="planes",
        n_envs=2,
        vec_env="dummy",
    )
    observation = env.reset()
    assert observation.shape[0] == 2
    env.close()
