from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from pushworld_study.paths import default_smoke_puzzle, ensure_upstream_pushworld_on_path


class PushWorldGymnasiumEnv(gym.Env):
    """Native Gymnasium PushWorld environment.

    This intentionally does not import the official legacy Gym environment
    (`external/pushworld/python3/src/pushworld/gym_env.py`) so modern training
    code does not inherit Gym's deprecation warning or NumPy constraints. The
    transition and rendering logic still come from the official benchmark's
    `PushWorldPuzzle` implementation to preserve puzzle semantics.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        puzzle_path: str | Path | None = None,
        max_steps: int | None = 100,
        render_mode: str | None = "rgb_array",
        standard_padding: bool = False,
    ) -> None:
        ensure_upstream_pushworld_on_path()

        from pushworld.config import PUZZLE_EXTENSION
        from pushworld.puzzle import (
            DEFAULT_BORDER_WIDTH,
            DEFAULT_PIXELS_PER_CELL,
            NUM_ACTIONS,
            PushWorldPuzzle,
        )
        from pushworld.utils.env_utils import (
            get_max_puzzle_dimensions,
            render_observation_padded,
        )
        from pushworld.utils.filesystem import iter_files_with_extension

        if render_mode not in (None, "rgb_array"):
            raise ValueError("PushWorld only supports render_mode='rgb_array'.")

        selected_puzzle = Path(puzzle_path) if puzzle_path else default_smoke_puzzle()
        self._puzzle_path = selected_puzzle
        self._max_steps = max_steps
        self._pixels_per_cell = DEFAULT_PIXELS_PER_CELL
        self._border_width = DEFAULT_BORDER_WIDTH
        self._render_observation_padded = render_observation_padded
        self.render_mode = render_mode

        self._puzzles = [
            PushWorldPuzzle(puzzle_file_path)
            for puzzle_file_path in iter_files_with_extension(
                str(selected_puzzle), PUZZLE_EXTENSION
            )
        ]
        if not self._puzzles:
            raise ValueError(f"No PushWorld puzzles found in: {selected_puzzle}")

        widths, heights = zip(*[puzzle.dimensions for puzzle in self._puzzles])
        self._max_cell_width = max(widths)
        self._max_cell_height = max(heights)

        if standard_padding:
            standard_cell_height, standard_cell_width = get_max_puzzle_dimensions()
            if standard_cell_height < self._max_cell_height:
                raise ValueError(
                    "`standard_padding` is True, but the benchmark maximum puzzle "
                    "height is less than this puzzle set's maximum height."
                )
            if standard_cell_width < self._max_cell_width:
                raise ValueError(
                    "`standard_padding` is True, but the benchmark maximum puzzle "
                    "width is less than this puzzle set's maximum width."
                )
            self._max_cell_height = standard_cell_height
            self._max_cell_width = standard_cell_width

        self._current_puzzle = None
        self._current_state = None
        self._current_achieved_goals = 0
        self._steps = 0

        initial_observation = self._render_observation(self._puzzles[0].initial_state)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=initial_observation.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

    @property
    def puzzle_path(self) -> Path:
        return self._puzzle_path

    @property
    def steps_taken(self) -> int:
        return self._steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        puzzle_idx = int(self.np_random.integers(len(self._puzzles)))
        self._current_puzzle = self._puzzles[puzzle_idx]
        self._current_state = self._current_puzzle.initial_state
        self._current_achieved_goals = self._current_puzzle.count_achieved_goals(
            self._current_state
        )
        self._steps = 0

        return self._render_observation(self._current_state), {
            "puzzle_state": self._current_state,
            "puzzle_index": puzzle_idx,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("The provided action is not in the action space.")
        if self._current_puzzle is None or self._current_state is None:
            raise RuntimeError("reset() must be called before step() can be called.")

        previous_state = self._current_state
        self._steps += 1
        self._current_state = self._current_puzzle.get_next_state(
            self._current_state, int(action)
        )

        terminated = self._current_puzzle.is_goal_state(self._current_state)
        if terminated:
            reward = 10.0
        else:
            previous_achieved_goals = self._current_puzzle.count_achieved_goals(
                previous_state
            )
            current_achieved_goals = self._current_puzzle.count_achieved_goals(
                self._current_state
            )
            reward = float(current_achieved_goals - previous_achieved_goals - 0.01)

        truncated = self._max_steps is not None and self._steps >= self._max_steps
        observation = self._render_observation(self._current_state)
        info = {"puzzle_state": self._current_state}

        return observation, reward, bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray:
        if self._current_puzzle is None or self._current_state is None:
            raise RuntimeError("reset() must be called before render() can be called.")
        return self._current_puzzle.render(
            self._current_state,
            border_width=self._border_width,
            pixels_per_cell=self._pixels_per_cell,
        )

    def close(self) -> None:
        return None

    def _render_observation(self, state: Any) -> np.ndarray:
        if self._current_puzzle is None:
            puzzle = self._puzzles[0]
        else:
            puzzle = self._current_puzzle
        return self._render_observation_padded(
            puzzle,
            state,
            self._max_cell_height,
            self._max_cell_width,
            self._pixels_per_cell,
            self._border_width,
        )


class ChannelFirstObservation(gym.ObservationWrapper):
    """Convert image observations from HWC to CHW for PyTorch policies."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        height, width, channels = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, height, width),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(observation, (2, 0, 1))


class ChannelFirstUint8Observation(gym.ObservationWrapper):
    """Convert HWC float observations to CHW uint8 for image replay buffers."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        height, width, channels = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels, height, width),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        chw = np.transpose(observation, (2, 0, 1))
        return np.rint(chw * 255).astype(np.uint8)


def make_pushworld_env(
    puzzle_path: str | Path | None = None,
    max_steps: int | None = 100,
    channel_first: bool = False,
    uint8_observation: bool = False,
    standard_padding: bool = False,
) -> gym.Env:
    env: gym.Env = PushWorldGymnasiumEnv(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
        standard_padding=standard_padding,
    )
    if uint8_observation and not channel_first:
        raise ValueError("uint8_observation=True requires channel_first=True.")
    if uint8_observation:
        env = ChannelFirstUint8Observation(env)
    elif channel_first:
        env = ChannelFirstObservation(env)
    return env
