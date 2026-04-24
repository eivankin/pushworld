from __future__ import annotations

from pushworld_study.baselines import profile_env_steps
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


def test_profile_env_steps_smoke() -> None:
    metrics = profile_env_steps(episodes=1, max_steps=3, seed=0)
    assert metrics["episodes"] == 1
    assert metrics["steps"] == 3
    assert metrics["steps_per_second"] > 0
