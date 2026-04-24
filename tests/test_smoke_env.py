from __future__ import annotations

import numpy as np

from pushworld_study.envs import PushWorldGymnasiumEnv


def test_native_gymnasium_env_steps() -> None:
    env = PushWorldGymnasiumEnv(max_steps=100)
    observation, info = env.reset(seed=0)
    assert observation.ndim == 3
    assert observation.shape[-1] == 3
    assert "puzzle_state" in info

    rng = np.random.default_rng(0)
    observation, reward, terminated, truncated, info = env.step(
        int(rng.integers(env.action_space.n))
    )

    assert observation in env.observation_space
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "puzzle_state" in info
