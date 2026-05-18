from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional optimization dependency
    triton = None
    tl = None

from pushworld_study.paths import ensure_upstream_pushworld_on_path


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402


State = tuple[tuple[int, int], ...]
EncodeCache = dict[tuple[str, State], torch.Tensor]
PredictionCache = dict[tuple[str, State], tuple[torch.Tensor, torch.Tensor]]
ACTION_COUNT = 4
DISTANCE_TARGETS = ("linear", "log")
BEAM_SCORE_MODES = ("policy", "policy_distance", "distance")
SEARCH_MODES = ("beam", "best_first", "best_first_fallback", "gpu_particles", "cem_sampling")
RANKER_MODES = ("model_distance", "model_goal_distance")


if triton is not None:
    @triton.jit
    def _pushworld_step_kernel(
        states_ptr,
        actions_ptr,
        out_ptr,
        agent_collision_ptr,
        wall_collision_ptr,
        movable_collision_ptr,
        displacements_ptr,
        batch_size,
        num_movables: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        rel_height: tl.constexpr,
        rel_width: tl.constexpr,
        rel_x_offset: tl.constexpr,
        rel_y_offset: tl.constexpr,
        max_movables: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= batch_size:
            return

        action = tl.load(actions_ptr + pid)
        agent_x = tl.load(states_ptr + (pid * max_movables * 2) + 0)
        agent_y = tl.load(states_ptr + (pid * max_movables * 2) + 1)
        valid_agent = (agent_x >= 0) & (agent_x < width) & (agent_y >= 0) & (agent_y < height)
        agent_blocked = tl.load(
            agent_collision_ptr + action * height * width + agent_y * width + agent_x,
            mask=valid_agent,
            other=1,
        ) != 0
        blocked = (~valid_agent) | agent_blocked

        pushed_bits = 1
        for _pass in tl.static_range(0, 16):
            if _pass < num_movables:
                for pusher_idx in tl.static_range(0, 16):
                    if pusher_idx < num_movables:
                        pusher_active = ((pushed_bits & (1 << pusher_idx)) != 0) & (~blocked)
                        pusher_x = tl.load(states_ptr + (pid * max_movables * 2) + pusher_idx * 2)
                        pusher_y = tl.load(states_ptr + (pid * max_movables * 2) + pusher_idx * 2 + 1)
                        for pushee_idx in tl.static_range(1, 16):
                            if pushee_idx < num_movables:
                                pushee_unpushed = (pushed_bits & (1 << pushee_idx)) == 0
                                active = pusher_active & pushee_unpushed
                                pushee_x = tl.load(states_ptr + (pid * max_movables * 2) + pushee_idx * 2)
                                pushee_y = tl.load(states_ptr + (pid * max_movables * 2) + pushee_idx * 2 + 1)
                                rel_x = pusher_x - pushee_x + rel_x_offset
                                rel_y = pusher_y - pushee_y + rel_y_offset
                                valid_rel = active & (rel_x >= 0) & (rel_x < rel_width) & (rel_y >= 0) & (rel_y < rel_height)
                                collision_offset = (
                                    (((action * max_movables + pusher_idx) * max_movables + pushee_idx) * rel_height + rel_y)
                                    * rel_width
                                    + rel_x
                                )
                                pushed_now = valid_rel & (
                                    tl.load(movable_collision_ptr + collision_offset, mask=valid_rel, other=0) != 0
                                )
                                wall_valid = pushed_now & (pushee_x >= 0) & (pushee_x < width) & (pushee_y >= 0) & (pushee_y < height)
                                wall_offset = ((action * max_movables + pushee_idx) * height + pushee_y) * width + pushee_x
                                wall_blocked = wall_valid & (
                                    tl.load(wall_collision_ptr + wall_offset, mask=wall_valid, other=0) != 0
                                )
                                blocked = blocked | wall_blocked
                                should_push = pushed_now & (~wall_blocked) & (~blocked)
                                pushed_bits = tl.where(should_push, pushed_bits | (1 << pushee_idx), pushed_bits)

        dx = tl.load(displacements_ptr + action * 2)
        dy = tl.load(displacements_ptr + action * 2 + 1)
        for movable_idx in tl.static_range(0, 16):
            if movable_idx < num_movables:
                x = tl.load(states_ptr + (pid * max_movables * 2) + movable_idx * 2)
                y = tl.load(states_ptr + (pid * max_movables * 2) + movable_idx * 2 + 1)
                should_move = ((pushed_bits & (1 << movable_idx)) != 0) & (~blocked)
                out_x = tl.where(should_move, x + dx, x)
                out_y = tl.where(should_move, y + dy, y)
                tl.store(out_ptr + (pid * max_movables * 2) + movable_idx * 2, out_x)
                tl.store(out_ptr + (pid * max_movables * 2) + movable_idx * 2 + 1, out_y)


@dataclass
class RolloutProfile:
    puzzle_parse_time_s: float = 0.0
    eval_loop_time_s: float = 0.0
    encode_time_s: float = 0.0
    model_forward_time_s: float = 0.0
    env_step_time_s: float = 0.0
    beam_expand_time_s: float = 0.0
    beam_rank_time_s: float = 0.0
    encode_cache_hits: int = 0
    encode_cache_misses: int = 0
    prediction_cache_hits: int = 0
    prediction_cache_misses: int = 0
    model_forward_batches: int = 0
    model_forward_states: int = 0
    predict_batch_calls: int = 0
    predict_batch_requested_states: int = 0
    predict_batch_unique_forward_states: int = 0
    beam_candidate_count: int = 0
    beam_closed_list_prunes: int = 0
    best_first_expand_time_s: float = 0.0
    best_first_rank_time_s: float = 0.0
    best_first_nodes_expanded: int = 0
    best_first_nodes_generated: int = 0
    best_first_closed_prunes: int = 0
    best_first_queue_max: int = 0
    best_first_batches: int = 0
    particle_step_time_s: float = 0.0
    particle_encode_time_s: float = 0.0
    particle_sample_time_s: float = 0.0
    particle_rank_time_s: float = 0.0
    particle_steps: int = 0
    particle_transitions: int = 0
    particle_resamples: int = 0
    cem_rounds: int = 0
    cem_rollouts: int = 0
    cem_transitions: int = 0
    cem_elite_updates: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "puzzle_parse_time_s": self.puzzle_parse_time_s,
            "eval_loop_time_s": self.eval_loop_time_s,
            "encode_time_s": self.encode_time_s,
            "model_forward_time_s": self.model_forward_time_s,
            "env_step_time_s": self.env_step_time_s,
            "beam_expand_time_s": self.beam_expand_time_s,
            "beam_rank_time_s": self.beam_rank_time_s,
            "encode_cache_hits": self.encode_cache_hits,
            "encode_cache_misses": self.encode_cache_misses,
            "prediction_cache_hits": self.prediction_cache_hits,
            "prediction_cache_misses": self.prediction_cache_misses,
            "model_forward_batches": self.model_forward_batches,
            "model_forward_states": self.model_forward_states,
            "predict_batch_calls": self.predict_batch_calls,
            "predict_batch_requested_states": self.predict_batch_requested_states,
            "predict_batch_unique_forward_states": self.predict_batch_unique_forward_states,
            "beam_candidate_count": self.beam_candidate_count,
            "beam_closed_list_prunes": self.beam_closed_list_prunes,
            "best_first_expand_time_s": self.best_first_expand_time_s,
            "best_first_rank_time_s": self.best_first_rank_time_s,
            "best_first_nodes_expanded": self.best_first_nodes_expanded,
            "best_first_nodes_generated": self.best_first_nodes_generated,
            "best_first_closed_prunes": self.best_first_closed_prunes,
            "best_first_queue_max": self.best_first_queue_max,
            "best_first_batches": self.best_first_batches,
            "particle_step_time_s": self.particle_step_time_s,
            "particle_encode_time_s": self.particle_encode_time_s,
            "particle_sample_time_s": self.particle_sample_time_s,
            "particle_rank_time_s": self.particle_rank_time_s,
            "particle_steps": self.particle_steps,
            "particle_transitions": self.particle_transitions,
            "particle_resamples": self.particle_resamples,
            "cem_rounds": self.cem_rounds,
            "cem_rollouts": self.cem_rollouts,
            "cem_transitions": self.cem_transitions,
            "cem_elite_updates": self.cem_elite_updates,
        }


@dataclass(frozen=True)
class BestFirstSearchResult:
    solved: bool
    path: tuple[int, ...]
    expanded: int
    generated: int
    closed: int
    frontier: int


@dataclass(frozen=True)
class ParticleSearchResult:
    solved: bool
    path: tuple[int, ...]
    steps: int
    particles: int
    transitions: int


@dataclass(frozen=True)
class CEMSearchResult:
    solved: bool
    path: tuple[int, ...]
    rounds: int
    rollouts: int
    transitions: int
    best_score: float


def set_cells(
    planes: np.ndarray,
    channel: int,
    origin: tuple[int, int],
    cells: set[tuple[int, int]],
) -> None:
    origin_x, origin_y = origin
    _, height, width = planes.shape
    for cell_x, cell_y in cells:
        x = origin_x + cell_x
        y = origin_y + cell_y
        if 0 <= x < width and 0 <= y < height:
            planes[channel, y, x] = 1.0


def encode_state(
    puzzle: PushWorldPuzzle,
    state: State,
    height: int,
    width: int,
) -> np.ndarray:
    planes = np.zeros((7, height, width), dtype=np.float32)

    for x, y in puzzle.wall_positions:
        if 0 <= x < width and 0 <= y < height:
            planes[0, y, x] = 1.0
    for x, y in puzzle.agent_wall_positions:
        if 0 <= x < width and 0 <= y < height:
            planes[1, y, x] = 1.0

    goal_count = len(puzzle.goal_state)
    for movable_idx, movable in enumerate(puzzle.movable_objects):
        if movable_idx == 0:
            channel = 2
        elif movable_idx <= goal_count:
            channel = 3
        else:
            channel = 4
        set_cells(planes, channel, state[movable_idx], movable.cells)

    for goal_idx, goal in enumerate(puzzle.goal_state, start=1):
        if goal_idx < len(puzzle.movable_objects):
            set_cells(planes, 5, goal, puzzle.movable_objects[goal_idx].cells)
            if state[goal_idx] == goal:
                set_cells(planes, 6, goal, puzzle.movable_objects[goal_idx].cells)

    return planes


def encode_cached(
    puzzle: PushWorldPuzzle,
    puzzle_key: str,
    state: State,
    height: int,
    width: int,
    cache: EncodeCache,
    max_cache_entries: int,
    profile: RolloutProfile | None = None,
) -> torch.Tensor:
    key = (puzzle_key, state)
    cached = cache.get(key)
    if cached is not None:
        if profile is not None:
            profile.encode_cache_hits += 1
        return cached
    if profile is not None:
        profile.encode_cache_misses += 1
    start = time.perf_counter()
    encoded = torch.from_numpy(encode_state(puzzle, state, height, width))
    if profile is not None:
        profile.encode_time_s += time.perf_counter() - start
    if max_cache_entries > 0 and len(cache) < max_cache_entries:
        cache[key] = encoded
    return encoded


class TensorPuzzleDynamics:
    def __init__(
        self,
        puzzle: PushWorldPuzzle,
        height: int,
        width: int,
        device: torch.device,
        *,
        use_triton: bool = False,
        use_approx: bool = False,
    ):
        self.puzzle = puzzle
        self.height = height
        self.width = width
        self.device = device
        self.num_movables = puzzle.num_movables
        self.max_movables = max(16, self.num_movables)
        self.use_triton = bool(use_triton and triton is not None and device.type == "cuda" and self.num_movables <= 16)
        self.use_approx = use_approx
        self.goal_count = len(puzzle.goal_state)

        displacements = getattr(Actions, "DISPLACEMENTS", ((-1, 0), (1, 0), (0, -1), (0, 1)))
        self.displacements = torch.as_tensor(displacements, device=device, dtype=torch.long)
        self.agent_collision = torch.zeros((ACTION_COUNT, height, width), device=device, dtype=torch.bool)
        for action in range(ACTION_COUNT):
            for x, y in puzzle._agent_collision_map[action]:  # noqa: SLF001
                if 0 <= x < width and 0 <= y < height:
                    self.agent_collision[action, y, x] = True

        self.wall_collision = torch.zeros(
            (ACTION_COUNT, self.max_movables, height, width),
            device=device,
            dtype=torch.bool,
        )
        for action in range(ACTION_COUNT):
            for movable_idx in range(self.num_movables):
                for x, y in puzzle._wall_collision_map[action][movable_idx]:  # noqa: SLF001
                    if 0 <= x < width and 0 <= y < height:
                        self.wall_collision[action, movable_idx, y, x] = True

        self.rel_x_offset = width
        self.rel_y_offset = height
        self.rel_width = 2 * width + 1
        self.rel_height = 2 * height + 1
        self.movable_collision = torch.zeros(
            (
                ACTION_COUNT,
                self.max_movables,
                self.max_movables,
                self.rel_height,
                self.rel_width,
            ),
            device=device,
            dtype=torch.bool,
        )
        for action in range(ACTION_COUNT):
            for pusher_idx in range(self.num_movables):
                for pushee_idx in range(self.num_movables):
                    rels = puzzle._movable_collision_map[action][pusher_idx][pushee_idx]  # noqa: SLF001
                    for rel_x, rel_y in rels:
                        idx_x = rel_x + self.rel_x_offset
                        idx_y = rel_y + self.rel_y_offset
                        if 0 <= idx_x < self.rel_width and 0 <= idx_y < self.rel_height:
                            self.movable_collision[action, pusher_idx, pushee_idx, idx_y, idx_x] = True

        self.wall_positions = torch.as_tensor(
            list(puzzle.wall_positions),
            device=device,
            dtype=torch.long,
        ) if puzzle.wall_positions else torch.empty((0, 2), device=device, dtype=torch.long)
        self.agent_wall_positions = torch.as_tensor(
            list(puzzle.agent_wall_positions),
            device=device,
            dtype=torch.long,
        ) if puzzle.agent_wall_positions else torch.empty((0, 2), device=device, dtype=torch.long)
        self.approx_blocked = torch.zeros((height, width), device=device, dtype=torch.bool)
        for x, y in set(puzzle.wall_positions) | set(puzzle.agent_wall_positions):
            if 0 <= x < width and 0 <= y < height:
                self.approx_blocked[y, x] = True
        self.object_cells = [
            torch.as_tensor(list(movable.cells), device=device, dtype=torch.long)
            if movable.cells
            else torch.empty((0, 2), device=device, dtype=torch.long)
            for movable in puzzle.movable_objects
        ]
        self.goal_state = torch.as_tensor(
            puzzle.goal_state,
            device=device,
            dtype=torch.long,
        ) if puzzle.goal_state else torch.empty((0, 2), device=device, dtype=torch.long)

    def states_to_tensor(self, states: list[State] | tuple[State, ...]) -> torch.Tensor:
        raw = torch.as_tensor(states, device=self.device, dtype=torch.long)
        if raw.shape[1] == self.max_movables:
            return raw.contiguous()
        padded = torch.zeros((raw.shape[0], self.max_movables, 2), device=self.device, dtype=torch.long)
        padded[:, : raw.shape[1]] = raw
        return padded

    def tensor_to_state(self, state: torch.Tensor) -> State:
        return tuple((int(x), int(y)) for x, y in state[: self.num_movables].detach().cpu().tolist())

    def is_goal(self, states: torch.Tensor) -> torch.Tensor:
        if self.goal_count == 0:
            return torch.ones((states.shape[0],), device=states.device, dtype=torch.bool)
        return torch.all(states[:, 1 : 1 + self.goal_count] == self.goal_state.unsqueeze(0), dim=(1, 2))

    def step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.use_approx:
            return self.step_approx(states, actions)
        if self.use_triton:
            return self.step_triton(states, actions)
        return self.step_torch(states, actions)

    def step_approx(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        displacement = self.displacements[actions]
        next_states = states.clone()
        pushed = torch.zeros((batch_size, self.max_movables), device=states.device, dtype=torch.bool)
        pushed[:, 0] = True
        blocked = torch.zeros((batch_size,), device=states.device, dtype=torch.bool)

        for _ in range(self.num_movables):
            for pusher_idx in range(self.num_movables):
                pusher_active = pushed[:, pusher_idx] & ~blocked
                target = states[:, pusher_idx] + displacement
                target_x = target[:, 0]
                target_y = target[:, 1]
                outside = (target_x < 0) | (target_x >= self.width) | (target_y < 0) | (target_y >= self.height)
                hits_static = torch.zeros((batch_size,), device=states.device, dtype=torch.bool)
                valid_target = pusher_active & ~outside
                hits_static[valid_target] = self.approx_blocked[target_y[valid_target], target_x[valid_target]]
                blocked |= pusher_active & (outside | hits_static)
                for pushee_idx in range(1, self.num_movables):
                    unpushed = ~pushed[:, pushee_idx]
                    target_matches = torch.all(states[:, pushee_idx] == target, dim=1)
                    pushed[:, pushee_idx] |= pusher_active & unpushed & target_matches & ~blocked

        movable_next = states + displacement[:, None, :]
        next_states = torch.where(pushed[:, :, None] & ~blocked[:, None, None], movable_next, next_states)
        return next_states

    def step_triton(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(states)
        _pushworld_step_kernel[(states.shape[0],)](
            states.contiguous(),
            actions.contiguous(),
            out,
            self.agent_collision,
            self.wall_collision,
            self.movable_collision,
            self.displacements,
            batch_size=states.shape[0],
            num_movables=self.num_movables,
            height=self.height,
            width=self.width,
            rel_height=self.rel_height,
            rel_width=self.rel_width,
            rel_x_offset=self.rel_x_offset,
            rel_y_offset=self.rel_y_offset,
            max_movables=self.max_movables,
            num_warps=1,
        )
        return out

    def step_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        next_states = states.clone()
        agent_pos = states[:, 0]
        valid_agent_pos = (
            (agent_pos[:, 0] >= 0)
            & (agent_pos[:, 0] < self.width)
            & (agent_pos[:, 1] >= 0)
            & (agent_pos[:, 1] < self.height)
        )
        blocked = torch.ones((batch_size,), device=states.device, dtype=torch.bool)
        blocked[valid_agent_pos] = self.agent_collision[
            actions[valid_agent_pos],
            agent_pos[valid_agent_pos, 1],
            agent_pos[valid_agent_pos, 0],
        ]

        pushed = torch.zeros((batch_size, self.max_movables), device=states.device, dtype=torch.bool)
        pushed[:, 0] = True

        for _ in range(self.num_movables):
            changed = torch.zeros((batch_size,), device=states.device, dtype=torch.bool)
            for pusher_idx in range(self.num_movables):
                pusher_active = pushed[:, pusher_idx] & ~blocked
                pusher_pos = states[:, pusher_idx]
                for pushee_idx in range(1, self.num_movables):
                    not_pushed = ~pushed[:, pushee_idx]
                    active = pusher_active & not_pushed
                    rel = pusher_pos - states[:, pushee_idx]
                    rel_x = rel[:, 0] + self.rel_x_offset
                    rel_y = rel[:, 1] + self.rel_y_offset
                    valid_rel = (
                        active
                        & (rel_x >= 0)
                        & (rel_x < self.rel_width)
                        & (rel_y >= 0)
                        & (rel_y < self.rel_height)
                    )
                    pushed_now = torch.zeros((batch_size,), device=states.device, dtype=torch.bool)
                    pushed_now[valid_rel] = self.movable_collision[
                        actions[valid_rel],
                        pusher_idx,
                        pushee_idx,
                        rel_y[valid_rel],
                        rel_x[valid_rel],
                    ]
                    obstacle_pos = states[:, pushee_idx]
                    valid_obstacle_pos = (
                        pushed_now
                        & (obstacle_pos[:, 0] >= 0)
                        & (obstacle_pos[:, 0] < self.width)
                        & (obstacle_pos[:, 1] >= 0)
                        & (obstacle_pos[:, 1] < self.height)
                    )
                    wall_blocked = torch.zeros((batch_size,), device=states.device, dtype=torch.bool)
                    wall_blocked[valid_obstacle_pos] = self.wall_collision[
                        actions[valid_obstacle_pos],
                        pushee_idx,
                        obstacle_pos[valid_obstacle_pos, 1],
                        obstacle_pos[valid_obstacle_pos, 0],
                    ]
                    blocked |= wall_blocked
                    newly_pushed = pushed_now & ~wall_blocked & ~blocked
                    pushed[:, pushee_idx] |= newly_pushed
                    changed |= newly_pushed

        displacement = self.displacements[actions]
        movable_next = states + displacement[:, None, :]
        next_states = torch.where(pushed[:, :, None], movable_next, next_states)
        return torch.where(blocked[:, None, None], states, next_states)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        planes = torch.zeros((batch_size, 7, self.height, self.width), device=states.device, dtype=torch.float32)
        if self.wall_positions.numel():
            planes[:, 0, self.wall_positions[:, 1], self.wall_positions[:, 0]] = 1.0
        if self.agent_wall_positions.numel():
            planes[:, 1, self.agent_wall_positions[:, 1], self.agent_wall_positions[:, 0]] = 1.0

        for movable_idx, cells in enumerate(self.object_cells):
            if cells.numel() == 0:
                continue
            if movable_idx == 0:
                channel = 2
            elif movable_idx <= self.goal_count:
                channel = 3
            else:
                channel = 4
            positions = states[:, movable_idx, None, :] + cells[None, :, :]
            x = positions[..., 0]
            y = positions[..., 1]
            valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            batch_idx = torch.arange(batch_size, device=states.device)[:, None].expand_as(x)
            if bool(torch.any(valid)):
                planes[batch_idx[valid], channel, y[valid], x[valid]] = 1.0

        for goal_idx, goal in enumerate(self.goal_state, start=1):
            if goal_idx >= self.num_movables:
                continue
            cells = self.object_cells[goal_idx]
            if cells.numel() == 0:
                continue
            positions = goal[None, :] + cells
            x = positions[:, 0]
            y = positions[:, 1]
            valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            if bool(torch.any(valid)):
                planes[:, 5, y[valid], x[valid]] = 1.0
                solved = torch.all(states[:, goal_idx] == goal, dim=1)
                if bool(torch.any(solved)):
                    solved_idx = torch.nonzero(solved, as_tuple=False).flatten()
                    planes[solved_idx[:, None], 6, y[valid], x[valid]] = 1.0
        return planes


def distance_targets(
    remaining: torch.Tensor,
    distance_bins: int,
    distance_target: str,
) -> torch.Tensor:
    if distance_target not in DISTANCE_TARGETS:
        raise ValueError(f"Unknown distance target {distance_target!r}; expected one of {DISTANCE_TARGETS}")
    if distance_bins <= 1:
        raise ValueError("distance_bins must be > 1")
    if distance_target == "log":
        targets = torch.round(torch.log(remaining.float() + 1.0)).long()
    else:
        targets = remaining.long()
    return targets.clamp_(min=0, max=distance_bins - 1)


def auto_distance_bins(max_steps: int, distance_target: str) -> int:
    if distance_target not in DISTANCE_TARGETS:
        raise ValueError(f"Unknown distance target {distance_target!r}; expected one of {DISTANCE_TARGETS}")
    if distance_target == "log":
        return max(2, int(math.ceil(math.log(max_steps + 1))) + 1)
    return max_steps + 1


def distance_bin_values(
    distance_bins: int,
    distance_target: str,
    max_steps: int | None,
    device: torch.device,
) -> torch.Tensor:
    if distance_target not in DISTANCE_TARGETS:
        raise ValueError(f"Unknown distance target {distance_target!r}; expected one of {DISTANCE_TARGETS}")
    bins = torch.arange(distance_bins, device=device, dtype=torch.float32)
    if distance_target == "log":
        values = torch.expm1(bins)
    else:
        values = bins
    if max_steps is not None:
        values = torch.clamp(values, max=float(max_steps))
    return values


def predict_batch(
    model: nn.Module,
    puzzle_states: list[tuple[PushWorldPuzzle, str, State]],
    height: int,
    width: int,
    device: torch.device,
    encode_cache: EncodeCache,
    max_cache_entries: int,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    prediction_cache: PredictionCache | None = None,
    profile: RolloutProfile | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if profile is not None:
        profile.predict_batch_calls += 1
        profile.predict_batch_requested_states += len(puzzle_states)

    cached_results: dict[tuple[str, State], tuple[torch.Tensor, torch.Tensor]] = {}
    uncached: list[tuple[PushWorldPuzzle, str, State]] = []
    uncached_keys: set[tuple[str, State]] = set()
    for puzzle, puzzle_key, state in puzzle_states:
        key = (puzzle_key, state)
        cached = prediction_cache.get(key) if prediction_cache is not None else None
        if cached is not None:
            if profile is not None:
                profile.prediction_cache_hits += 1
            cached_results[key] = cached
            continue
        if key not in uncached_keys:
            uncached.append((puzzle, puzzle_key, state))
            uncached_keys.add(key)
        if profile is not None:
            profile.prediction_cache_misses += 1

    if uncached:
        encoded = [
            encode_cached(
                puzzle,
                puzzle_key,
                state,
                height,
                width,
                encode_cache,
                max_cache_entries,
                profile,
            )
            for puzzle, puzzle_key, state in uncached
        ]
        batch = torch.stack(encoded).to(device)
        start = time.perf_counter()
        action_logits, distance_logits = model(batch)
        action_log_probs = torch.log_softmax(action_logits, dim=-1).cpu()
        distance_probs = torch.softmax(distance_logits, dim=-1)
        distances = distance_bin_values(
            distance_logits.shape[-1],
            distance_target,
            distance_max_steps,
            device,
        )
        expected_distance = torch.sum(distance_probs * distances.unsqueeze(0), dim=-1).cpu()
        if profile is not None:
            profile.model_forward_time_s += time.perf_counter() - start
            profile.model_forward_batches += 1
            profile.model_forward_states += len(uncached)
            profile.predict_batch_unique_forward_states += len(uncached)
        for idx, (_, puzzle_key, state) in enumerate(uncached):
            key = (puzzle_key, state)
            result = (action_log_probs[idx], expected_distance[idx])
            cached_results[key] = result
            if (
                prediction_cache is not None
                and max_cache_entries > 0
                and len(prediction_cache) < max_cache_entries
            ):
                prediction_cache[key] = result

    action_rows = []
    distance_rows = []
    for _, puzzle_key, state in puzzle_states:
        action_log_probs, expected_distance = cached_results[(puzzle_key, state)]
        action_rows.append(action_log_probs)
        distance_rows.append(expected_distance)
    return torch.stack(action_rows), torch.stack(distance_rows)


def beam_rank_score(
    policy_cost: float,
    expected_distance: float,
    path_len: int,
    beam_score: str,
    distance_weight: float,
    beam_length_normalization: float,
) -> float:
    if beam_score not in BEAM_SCORE_MODES:
        raise ValueError(f"Unknown beam score mode {beam_score!r}; expected one of {BEAM_SCORE_MODES}")
    if beam_length_normalization < 0.0:
        raise ValueError("beam_length_normalization must be >= 0")
    if distance_weight < 0.0:
        raise ValueError("distance_weight must be >= 0")

    normalized_policy_cost = policy_cost / (max(1, path_len) ** beam_length_normalization)
    if beam_score == "policy":
        return normalized_policy_cost
    if beam_score == "distance":
        return expected_distance + distance_weight * normalized_policy_cost
    return normalized_policy_cost + distance_weight * expected_distance


def best_first_priority(
    policy_cost: float,
    expected_distance: float,
    path_len: int,
    distance_weight: float,
    step_penalty: float,
    goal_distance: float = 0.0,
    achieved_goals: int = 0,
    goal_distance_weight: float = 0.0,
    achieved_goal_bonus: float = 0.0,
) -> float:
    return (
        policy_cost
        + distance_weight * expected_distance
        + step_penalty * path_len
        + goal_distance_weight * goal_distance
        - achieved_goal_bonus * achieved_goals
    )


def goal_rank_features(puzzle: PushWorldPuzzle, state: State) -> tuple[float, int]:
    goal_distance = 0
    achieved = 0
    for object_state, goal_state in zip(state[1 : 1 + len(puzzle.goal_state)], puzzle.goal_state, strict=False):
        if object_state == goal_state:
            achieved += 1
        goal_distance += abs(object_state[0] - goal_state[0]) + abs(object_state[1] - goal_state[1])
    return float(goal_distance), achieved


def best_first_search(
    model: nn.Module,
    puzzle: PushWorldPuzzle,
    state: State,
    height: int,
    width: int,
    device: torch.device,
    puzzle_key: str,
    encode_cache: EncodeCache,
    max_cache_entries: int,
    node_budget: int,
    batch_size: int,
    top_k: int,
    max_depth: int,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    distance_weight: float = 0.15,
    step_penalty: float = 0.0,
    ranker_mode: str = "model_distance",
    goal_distance_weight: float = 0.0,
    achieved_goal_bonus: float = 0.0,
    prediction_cache: PredictionCache | None = None,
    profile: RolloutProfile | None = None,
) -> BestFirstSearchResult:
    if ranker_mode not in RANKER_MODES:
        raise ValueError(f"Unknown ranker mode {ranker_mode!r}; expected one of {RANKER_MODES}")
    if node_budget <= 0 or batch_size <= 0 or top_k <= 0 or max_depth <= 0:
        return BestFirstSearchResult(False, (), 0, 0, 0, 0)
    if puzzle.is_goal_state(state):
        return BestFirstSearchResult(True, (), 0, 0, 0, 0)

    counter = itertools.count()
    frontier: list[tuple[float, int, int, State, tuple[int, ...], float]] = [
        (0.0, 0, next(counter), state, (), 0.0)
    ]
    closed: set[State] = set()
    expanded = 0
    generated = 0

    while frontier and expanded < node_budget:
        nodes: list[tuple[State, tuple[int, ...], float]] = []
        while frontier and len(nodes) < batch_size and expanded + len(nodes) < node_budget:
            _, _, _, node_state, path, policy_cost = heapq.heappop(frontier)
            if node_state in closed:
                if profile is not None:
                    profile.best_first_closed_prunes += 1
                continue
            closed.add(node_state)
            nodes.append((node_state, path, policy_cost))
        if not nodes:
            continue

        if profile is not None:
            profile.best_first_batches += 1
            profile.best_first_nodes_expanded += len(nodes)

        action_log_probs, _ = predict_batch(
            model,
            [(puzzle, puzzle_key, node_state) for node_state, _, _ in nodes],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
            prediction_cache,
            profile,
        )

        candidates_by_state: dict[State, tuple[State, tuple[int, ...], float]] = {}
        expand_start = time.perf_counter()
        for node_idx, (node_state, path, policy_cost) in enumerate(nodes):
            expanded += 1
            if puzzle.is_goal_state(node_state):
                return BestFirstSearchResult(True, path, expanded, generated, len(closed), len(frontier))
            if len(path) >= max_depth:
                continue
            action_count = min(top_k, ACTION_COUNT)
            top_actions = torch.topk(action_log_probs[node_idx], k=action_count).indices.tolist()
            for action in top_actions:
                step_start = time.perf_counter()
                next_state = puzzle.get_next_state(node_state, int(action))
                if profile is not None:
                    profile.env_step_time_s += time.perf_counter() - step_start
                if next_state == node_state:
                    continue
                if next_state in closed:
                    if profile is not None:
                        profile.best_first_closed_prunes += 1
                    continue
                next_path = path + (int(action),)
                next_policy_cost = policy_cost - float(action_log_probs[node_idx, action])
                generated += 1
                if profile is not None:
                    profile.best_first_nodes_generated += 1
                if puzzle.is_goal_state(next_state):
                    if profile is not None:
                        profile.best_first_expand_time_s += time.perf_counter() - expand_start
                    return BestFirstSearchResult(
                        True,
                        next_path,
                        expanded,
                        generated,
                        len(closed),
                        len(frontier),
                    )
                previous = candidates_by_state.get(next_state)
                if previous is None or next_policy_cost < previous[2]:
                    candidates_by_state[next_state] = (next_state, next_path, next_policy_cost)
        if profile is not None:
            profile.best_first_expand_time_s += time.perf_counter() - expand_start

        candidates = list(candidates_by_state.values())
        if not candidates:
            continue
        rank_start = time.perf_counter()
        _, leaf_distances = predict_batch(
            model,
            [(puzzle, puzzle_key, candidate[0]) for candidate in candidates],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
            prediction_cache,
            profile,
        )
        for candidate, expected_distance in zip(candidates, leaf_distances.tolist(), strict=True):
            _, path, policy_cost = candidate
            if ranker_mode == "model_goal_distance":
                goal_distance, achieved_goals = goal_rank_features(puzzle, candidate[0])
            else:
                goal_distance, achieved_goals = 0.0, 0
            priority = best_first_priority(
                policy_cost=policy_cost,
                expected_distance=float(expected_distance),
                path_len=len(path),
                distance_weight=distance_weight,
                step_penalty=step_penalty,
                goal_distance=goal_distance,
                achieved_goals=achieved_goals,
                goal_distance_weight=goal_distance_weight,
                achieved_goal_bonus=achieved_goal_bonus,
            )
            heapq.heappush(frontier, (priority, len(path), next(counter), candidate[0], path, policy_cost))
        if profile is not None:
            profile.best_first_rank_time_s += time.perf_counter() - rank_start
            profile.best_first_queue_max = max(profile.best_first_queue_max, len(frontier))

    return BestFirstSearchResult(False, (), expanded, generated, len(closed), len(frontier))


def gpu_particle_search(
    model: nn.Module,
    puzzle: PushWorldPuzzle,
    state: State,
    height: int,
    width: int,
    device: torch.device,
    particles: int,
    max_steps: int,
    temperature: float = 1.0,
    top_k: int = 4,
    resample_every: int = 4,
    keep_fraction: float = 0.5,
    dynamics_backend: str = "torch",
    policy_interval: int = 1,
    model_batch_size: int = 128,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    distance_weight: float = 0.15,
    repeat_penalty: float = 1.0,
    profile: RolloutProfile | None = None,
) -> ParticleSearchResult:
    if particles <= 0 or max_steps <= 0:
        return ParticleSearchResult(False, (), 0, max(0, particles), 0)
    if puzzle.is_goal_state(state):
        return ParticleSearchResult(True, (), 0, particles, 0)
    if device.type != "cuda":
        device = torch.device("cpu")

    if dynamics_backend not in ("torch", "triton", "approx"):
        raise ValueError("dynamics_backend must be one of: torch, triton, approx")
    dynamics = TensorPuzzleDynamics(
        puzzle,
        height,
        width,
        device,
        use_triton=dynamics_backend == "triton",
        use_approx=dynamics_backend == "approx",
    )
    states = dynamics.states_to_tensor([state]).repeat(particles, 1, 1)
    paths = torch.full((particles, max_steps), -1, device=device, dtype=torch.long)
    scores = torch.zeros((particles,), device=device, dtype=torch.float32)
    path_lengths = torch.zeros((particles,), device=device, dtype=torch.long)
    active = torch.ones((particles,), device=device, dtype=torch.bool)
    seen: set[State] = {state}
    transitions = 0
    temperature = max(1e-4, float(temperature))
    keep_count = max(1, min(particles, int(round(particles * keep_fraction))))
    action_count = max(1, min(ACTION_COUNT, top_k))
    policy_interval = max(1, int(policy_interval))
    model_batch_size = max(1, int(model_batch_size))
    action_log_probs = None
    expected_distance = torch.zeros((particles,), device=device, dtype=torch.float32)

    for step_idx in range(max_steps):
        if profile is not None:
            profile.particle_steps += 1
        if action_log_probs is None or step_idx % policy_interval == 0:
            encode_start = time.perf_counter()
            batch = dynamics.encode(states)
            if profile is not None:
                profile.particle_encode_time_s += time.perf_counter() - encode_start

            model_start = time.perf_counter()
            action_chunks = []
            distance_chunks = []
            for batch_start in range(0, batch.shape[0], model_batch_size):
                batch_end = min(batch.shape[0], batch_start + model_batch_size)
                action_logits, distance_logits = model(batch[batch_start:batch_end])
                action_chunks.append(torch.log_softmax(action_logits, dim=-1))
                distance_probs = torch.softmax(distance_logits, dim=-1)
                distances = distance_bin_values(
                    distance_logits.shape[-1],
                    distance_target,
                    distance_max_steps,
                    device,
                )
                distance_chunks.append(torch.sum(distance_probs * distances.unsqueeze(0), dim=-1))
            action_log_probs = torch.cat(action_chunks, dim=0)
            expected_distance = torch.cat(distance_chunks, dim=0)
            if profile is not None:
                profile.model_forward_time_s += time.perf_counter() - model_start
                profile.model_forward_batches += math.ceil(int(states.shape[0]) / model_batch_size)
                profile.model_forward_states += int(states.shape[0])

        sample_start = time.perf_counter()
        top_values, top_indices = torch.topk(action_log_probs, k=action_count, dim=-1)
        sample_probs = torch.softmax(top_values / temperature, dim=-1)
        sampled_offsets = torch.multinomial(sample_probs, num_samples=1).squeeze(1)
        actions = top_indices[torch.arange(states.shape[0], device=device), sampled_offsets]
        chosen_log_probs = action_log_probs[torch.arange(states.shape[0], device=device), actions]
        if profile is not None:
            profile.particle_sample_time_s += time.perf_counter() - sample_start

        step_start = time.perf_counter()
        next_states = dynamics.step(states, actions)
        changed = torch.any(next_states != states, dim=(1, 2))
        transitions += int(states.shape[0])
        if profile is not None:
            profile.particle_step_time_s += time.perf_counter() - step_start
            profile.particle_transitions += int(states.shape[0])

        repeat_mask = torch.zeros((states.shape[0],), device=device, dtype=torch.bool)
        if repeat_penalty > 0.0:
            next_states_cpu = next_states.detach().cpu().tolist()
            repeat_values = [
                tuple((int(x), int(y)) for x, y in row[: dynamics.num_movables]) in seen
                for row in next_states_cpu
            ]
            repeat_mask = torch.as_tensor(repeat_values, device=device, dtype=torch.bool)
            for row, is_changed in zip(next_states_cpu, changed.detach().cpu().tolist(), strict=True):
                if is_changed:
                    seen.add(tuple((int(x), int(y)) for x, y in row[: dynamics.num_movables]))

        scores = scores - chosen_log_probs + distance_weight * expected_distance
        scores = scores + torch.where(changed, 0.0, 5.0)
        if repeat_penalty > 0.0:
            scores = scores + repeat_mask.float() * repeat_penalty

        paths[:, step_idx] = actions
        path_lengths += active.long()
        states = next_states

        solved = dynamics.is_goal(states)
        if bool(torch.any(solved)):
            best_idx = int(torch.nonzero(solved, as_tuple=False)[0].item())
            path_len = int(path_lengths[best_idx].item())
            path = tuple(int(action) for action in paths[best_idx, :path_len].detach().cpu().tolist())
            return ParticleSearchResult(True, path, step_idx + 1, particles, transitions)

        if resample_every > 0 and (step_idx + 1) % resample_every == 0 and keep_count < particles:
            rank_start = time.perf_counter()
            keep_indices = torch.topk(-scores, k=keep_count).indices
            refill = torch.randint(0, keep_count, (particles,), device=device)
            source_indices = keep_indices[refill]
            states = states[source_indices].clone()
            paths = paths[source_indices].clone()
            scores = scores[source_indices].clone()
            path_lengths = path_lengths[source_indices].clone()
            if profile is not None:
                profile.particle_rank_time_s += time.perf_counter() - rank_start
                profile.particle_resamples += 1

    best_idx = int(torch.argmin(scores).item())
    path_len = int(path_lengths[best_idx].item())
    path = tuple(int(action) for action in paths[best_idx, :path_len].detach().cpu().tolist())
    return ParticleSearchResult(False, path, max_steps, particles, transitions)


def rollout_rank_score(
    puzzle: PushWorldPuzzle,
    state: State,
    solved: bool,
    path_len: int,
    repeated_states: int,
    invalid_moves: int,
) -> float:
    achieved = puzzle.count_achieved_goals(state)
    goal_distance, _ = goal_rank_features(puzzle, state)
    return (
        (10_000.0 if solved else 0.0)
        + 100.0 * achieved
        - 2.0 * goal_distance
        - 0.25 * path_len
        - 1.0 * repeated_states
        - 0.5 * invalid_moves
    )


def cem_sampling_search(
    model: nn.Module,
    puzzle: PushWorldPuzzle,
    state: State,
    height: int,
    width: int,
    device: torch.device,
    puzzle_key: str,
    encode_cache: EncodeCache,
    max_cache_entries: int,
    rollouts: int,
    rounds: int,
    elite_fraction: float,
    max_steps: int,
    temperature: float = 1.0,
    top_k: int = 4,
    cem_prior_weight: float = 1.0,
    cem_smoothing: float = 0.2,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    prediction_cache: PredictionCache | None = None,
    profile: RolloutProfile | None = None,
) -> CEMSearchResult:
    if rollouts <= 0 or rounds <= 0 or max_steps <= 0:
        return CEMSearchResult(False, (), 0, max(0, rollouts), 0, float("-inf"))
    if puzzle.is_goal_state(state):
        return CEMSearchResult(True, (), 0, 0, 0, 10_000.0)

    temperature = max(1e-4, float(temperature))
    action_count = max(1, min(ACTION_COUNT, top_k))
    elite_count = max(1, min(rollouts, int(round(rollouts * elite_fraction))))
    prior_probs = torch.full((max_steps, ACTION_COUNT), 1.0 / ACTION_COUNT, dtype=torch.float32)
    best_path: tuple[int, ...] = ()
    best_score = float("-inf")
    best_solved = False
    transitions = 0

    for round_idx in range(rounds):
        if profile is not None:
            profile.cem_rounds += 1
            profile.cem_rollouts += rollouts
        states = [state for _ in range(rollouts)]
        paths: list[list[int]] = [[] for _ in range(rollouts)]
        solved_flags = [False for _ in range(rollouts)]
        repeated_counts = [0 for _ in range(rollouts)]
        invalid_counts = [0 for _ in range(rollouts)]
        seen_states = [{state} for _ in range(rollouts)]

        for step_idx in range(max_steps):
            active = [idx for idx, solved in enumerate(solved_flags) if not solved]
            if not active:
                break
            action_log_probs, _ = predict_batch(
                model,
                [(puzzle, f"{puzzle_key}:cem:{round_idx}", states[idx]) for idx in active],
                height,
                width,
                device,
                encode_cache,
                max_cache_entries,
                distance_target,
                distance_max_steps,
                prediction_cache,
                profile,
            )
            prior = torch.log(prior_probs[step_idx].clamp_min(1e-6))
            logits = action_log_probs + cem_prior_weight * prior.unsqueeze(0)
            if action_count < ACTION_COUNT:
                top_values, top_indices = torch.topk(logits, k=action_count, dim=-1)
                sample_probs = torch.softmax(top_values / temperature, dim=-1)
                sampled_offsets = torch.multinomial(sample_probs, num_samples=1).squeeze(1)
                sampled_actions = top_indices[torch.arange(len(active)), sampled_offsets]
            else:
                sample_probs = torch.softmax(logits / temperature, dim=-1)
                sampled_actions = torch.multinomial(sample_probs, num_samples=1).squeeze(1)

            for local_idx, rollout_idx in enumerate(active):
                action = int(sampled_actions[local_idx].item())
                next_state = puzzle.get_next_state(states[rollout_idx], action)
                transitions += 1
                if profile is not None:
                    profile.cem_transitions += 1
                paths[rollout_idx].append(action)
                if next_state == states[rollout_idx]:
                    invalid_counts[rollout_idx] += 1
                if next_state in seen_states[rollout_idx]:
                    repeated_counts[rollout_idx] += 1
                seen_states[rollout_idx].add(next_state)
                states[rollout_idx] = next_state
                if puzzle.is_goal_state(next_state):
                    solved_flags[rollout_idx] = True

        scored = []
        for rollout_idx in range(rollouts):
            score = rollout_rank_score(
                puzzle,
                states[rollout_idx],
                solved_flags[rollout_idx],
                len(paths[rollout_idx]),
                repeated_counts[rollout_idx],
                invalid_counts[rollout_idx],
            )
            path = tuple(paths[rollout_idx])
            scored.append((score, solved_flags[rollout_idx], path))
            if score > best_score or (solved_flags[rollout_idx] and not best_solved):
                best_score = score
                best_solved = solved_flags[rollout_idx]
                best_path = path

        elites = sorted(scored, key=lambda item: item[0], reverse=True)[:elite_count]
        counts = torch.full((max_steps, ACTION_COUNT), cem_smoothing, dtype=torch.float32)
        for _, _, path in elites:
            for step_idx, action in enumerate(path[:max_steps]):
                counts[step_idx, action] += 1.0
        prior_probs = counts / counts.sum(dim=-1, keepdim=True)
        if profile is not None:
            profile.cem_elite_updates += 1

        solved_elites = [path for _, solved, path in elites if solved]
        if solved_elites:
            shortest = min(solved_elites, key=len)
            return CEMSearchResult(True, shortest, round_idx + 1, rollouts * (round_idx + 1), transitions, best_score)

    return CEMSearchResult(best_solved, best_path, rounds, rollouts * rounds, transitions, best_score)


def choose_action(
    model: nn.Module,
    puzzle: PushWorldPuzzle,
    state: State,
    height: int,
    width: int,
    device: torch.device,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    puzzle_key: str,
    encode_cache: EncodeCache,
    max_cache_entries: int,
    seen_states: Iterable[State] | None = None,
    repeat_penalty: float = 0.0,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    beam_score: str = "policy_distance",
    distance_weight: float = 0.15,
    beam_length_normalization: float = 0.0,
    prediction_cache: PredictionCache | None = None,
    profile: RolloutProfile | None = None,
    closed_list_pruning: bool = False,
) -> int:
    seen = set(seen_states) if seen_states is not None and repeat_penalty > 0.0 else set()
    closed = set(seen_states) if seen_states is not None and closed_list_pruning else set()

    if beam_width <= 1 or beam_depth <= 1:
        action_log_probs, _ = predict_batch(
            model,
            [(puzzle, puzzle_key, state)],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
            prediction_cache,
            profile,
        )
        fallback_action: int | None = None
        for action in torch.argsort(action_log_probs[0], descending=True).tolist():
            step_start = time.perf_counter()
            next_state = puzzle.get_next_state(state, int(action))
            if profile is not None:
                profile.env_step_time_s += time.perf_counter() - step_start
            if next_state == state:
                continue
            if fallback_action is None:
                fallback_action = int(action)
            if next_state not in seen and next_state not in closed:
                return int(action)
        if fallback_action is not None:
            return fallback_action
        return int(torch.argmax(action_log_probs[0]).item())

    beams: list[tuple[State, tuple[int, ...], float]] = [(state, (), 0.0)]
    best_solved: tuple[int, ...] | None = None
    best_nonempty_path: tuple[int, ...] | None = None
    for _ in range(beam_depth):
        predictions = predict_batch(
            model,
            [(puzzle, puzzle_key, beam_state) for beam_state, _, _ in beams],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
            prediction_cache,
            profile,
        )
        action_log_probs, _ = predictions
        candidates_by_state: dict[State, tuple[State, tuple[int, ...], float]] = {}
        expand_start = time.perf_counter()
        for beam_idx, (beam_state, path, score) in enumerate(beams):
            action_count = min(top_k, ACTION_COUNT)
            top_actions = torch.topk(action_log_probs[beam_idx], k=action_count).indices.tolist()
            for action in top_actions:
                step_start = time.perf_counter()
                next_state = puzzle.get_next_state(beam_state, int(action))
                if profile is not None:
                    profile.env_step_time_s += time.perf_counter() - step_start
                if next_state == beam_state:
                    continue
                if next_state in closed:
                    if profile is not None:
                        profile.beam_closed_list_prunes += 1
                    continue
                next_path = path + (int(action),)
                best_nonempty_path = best_nonempty_path or next_path
                next_score = score - float(action_log_probs[beam_idx, action])
                if next_state in seen:
                    next_score += repeat_penalty
                if profile is not None:
                    profile.beam_candidate_count += 1
                if puzzle.is_goal_state(next_state):
                    best_solved = next_path
                    break
                previous = candidates_by_state.get(next_state)
                if previous is None or next_score < previous[2]:
                    candidates_by_state[next_state] = (next_state, next_path, next_score)
            if best_solved is not None:
                break
        if profile is not None:
            profile.beam_expand_time_s += time.perf_counter() - expand_start
        if best_solved is not None:
            return best_solved[0]
        candidates = list(candidates_by_state.values())
        if not candidates:
            break
        leaf_log_probs, leaf_distances = predict_batch(
            model,
            [(puzzle, puzzle_key, candidate[0]) for candidate in candidates],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
            prediction_cache,
            profile,
        )
        del leaf_log_probs
        rank_start = time.perf_counter()
        ranked = sorted(
            zip(candidates, leaf_distances.tolist(), strict=True),
            key=lambda item: beam_rank_score(
                policy_cost=item[0][2],
                expected_distance=float(item[1]),
                path_len=len(item[0][1]),
                beam_score=beam_score,
                distance_weight=distance_weight,
                beam_length_normalization=beam_length_normalization,
            ),
        )
        if profile is not None:
            profile.beam_rank_time_s += time.perf_counter() - rank_start
        beams = [candidate for candidate, _ in ranked[:beam_width]]

    if beams and beams[0][1]:
        return beams[0][1][0]
    if best_nonempty_path:
        return best_nonempty_path[0]
    return Actions.LEFT
