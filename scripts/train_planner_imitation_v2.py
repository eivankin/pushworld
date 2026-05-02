from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from pushworld_study.paths import PROJECT_ROOT, ensure_upstream_pushworld_on_path


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402


ACTION_CHARS = "LRUD"
ACTION_NAMES = ("left", "right", "up", "down")
SYMMETRY_TRANSFORMS = (
    "r0",
    "r90",
    "r180",
    "r270",
    "r0_flipped",
    "r90_flipped",
    "r180_flipped",
    "r270_flipped",
)
ACTION_CHAR_TO_INDEX = {char: idx for idx, char in enumerate(ACTION_CHARS)}
ACTION_INDEX_TO_CHAR = {idx: char for idx, char in enumerate(ACTION_CHARS)}
TRANSFORM_ACTION_MAP = {
    "r0": {"L": "L", "R": "R", "U": "U", "D": "D"},
    "r90": {"L": "U", "R": "D", "U": "R", "D": "L"},
    "r180": {"L": "R", "R": "L", "U": "D", "D": "U"},
    "r270": {"L": "D", "R": "U", "U": "L", "D": "R"},
    "r0_flipped": {"L": "L", "R": "R", "U": "D", "D": "U"},
    "r90_flipped": {"L": "U", "R": "D", "U": "L", "D": "R"},
    "r180_flipped": {"L": "R", "R": "L", "U": "U", "D": "D"},
    "r270_flipped": {"L": "D", "R": "U", "U": "R", "D": "L"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Trajectory:
    puzzle_path: Path
    plan: str
    solve_time_s: float


@dataclass(frozen=True)
class ExpertStep:
    puzzle: PushWorldPuzzle
    puzzle_height: int
    puzzle_width: int
    state: tuple[tuple[int, int], ...]
    action: int
    remaining: int
    transforms: tuple[str, ...]


class ExpertDataset(Dataset):
    def __init__(
        self,
        trajectories: list[Trajectory],
        height: int,
        width: int,
        transforms: tuple[str, ...] = ("r0",),
        transform_level0_only: bool = False,
        seed: int = 1,
    ) -> None:
        self.steps: list[ExpertStep] = []
        self.height = height
        self.width = width
        self.transforms = transforms
        self.transform_level0_only = transform_level0_only
        self.base_examples = 0
        self.augmented_base_examples = 0
        self.unaugmented_base_examples = 0
        self.rng = random.Random(seed)

        for trajectory in trajectories:
            puzzle = PushWorldPuzzle(str(trajectory.puzzle_path))
            puzzle_width, puzzle_height = puzzle.dimensions
            state = puzzle.initial_state
            plan_length = len(trajectory.plan)
            trajectory_transforms = transforms
            if transform_level0_only and not is_level0_path(trajectory.puzzle_path):
                trajectory_transforms = ("r0",)
            for step_idx, action_char in enumerate(trajectory.plan):
                action = Actions.FROM_CHAR[action_char]
                remaining = plan_length - step_idx
                self.base_examples += 1
                if len(trajectory_transforms) > 1:
                    self.augmented_base_examples += 1
                else:
                    self.unaugmented_base_examples += 1
                self.steps.append(
                    ExpertStep(
                        puzzle=puzzle,
                        puzzle_height=puzzle_height,
                        puzzle_width=puzzle_width,
                        state=state,
                        action=action,
                        remaining=remaining,
                        transforms=trajectory_transforms,
                    )
                )
                state = puzzle.get_next_state(state, action)
            if not puzzle.is_goal_state(state):
                raise ValueError(f"Planner trace does not solve {trajectory.puzzle_path}")

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step = self.steps[idx]
        transform = self.rng.choice(step.transforms) if len(step.transforms) > 1 else "r0"
        state = transform_encoded_state(
            encode_state(step.puzzle, step.state, self.height, self.width),
            puzzle_height=step.puzzle_height,
            puzzle_width=step.puzzle_width,
            height=self.height,
            width=self.width,
            transform=transform,
        )
        action = transform_action(step.action, transform)
        return (
            torch.from_numpy(state),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(step.remaining, dtype=torch.long),
        )


def is_level0_path(path: Path) -> bool:
    try:
        path.resolve().relative_to((PROJECT_ROOT / "data/level0").resolve())
        return True
    except ValueError:
        return False


class BoardTransformerPolicy(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 2,
        distance_bins: int = 101,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.token_proj = nn.Linear(channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.action_head = nn.Linear(d_model, len(ACTION_NAMES))
        self.distance_head = nn.Linear(d_model, distance_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = x.permute(0, 2, 3, 1).reshape(batch, self.height * self.width, -1)
        tokens = self.token_proj(tokens)
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        encoded = self.encoder(tokens)
        pooled = encoded[:, 0]
        return self.action_head(pooled), self.distance_head(pooled)


def sorted_puzzles(path: Path) -> list[Path]:
    def key(puzzle_path: Path) -> tuple[str, int]:
        match = re.search(r"_(\d+)$", puzzle_path.stem)
        return (puzzle_path.stem[: match.start()] if match else puzzle_path.stem, int(match.group(1)) if match else -1)

    return sorted(path.glob("*.pwp"), key=key)


def select_puzzles(paths: list[Path], limit: int, use_all: bool) -> list[Path]:
    puzzle_paths = []
    seen = set()
    for path in paths:
        for puzzle_path in sorted_puzzles(path):
            resolved = puzzle_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            puzzle_paths.append(puzzle_path)
    paths = puzzle_paths
    return paths if use_all else paths[:limit]


def max_dimensions(paths: list[Path]) -> tuple[int, int]:
    widths = []
    heights = []
    for path in paths:
        puzzle = PushWorldPuzzle(str(path))
        width, height = puzzle.dimensions
        widths.append(width)
        heights.append(height)
    return max(heights), max(widths)


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


def transform_action(action: int, transform: str) -> int:
    action_char = ACTION_INDEX_TO_CHAR[action]
    transformed_char = TRANSFORM_ACTION_MAP[transform][action_char]
    return ACTION_CHAR_TO_INDEX[transformed_char]


def transform_encoded_state(
    planes: np.ndarray,
    puzzle_height: int,
    puzzle_width: int,
    height: int,
    width: int,
    transform: str,
) -> np.ndarray:
    crop = planes[:, :puzzle_height, :puzzle_width]
    transformed = crop
    if transform.endswith("_flipped"):
        transformed = np.flip(transformed, axis=1)
    rotation = int(transform.split("_", maxsplit=1)[0][1:])
    for _ in range(rotation // 90):
        transformed = np.rot90(transformed, axes=(2, 1))

    transformed = transformed.copy()
    transformed_height, transformed_width = transformed.shape[1:]
    if transformed_height > height or transformed_width > width:
        raise ValueError(
            f"Transformed state {transform} has shape "
            f"{transformed_height}x{transformed_width}, exceeding {height}x{width}"
        )
    output = np.zeros_like(planes)
    output[:, :transformed_height, :transformed_width] = transformed
    return output


def encode_state(
    puzzle: PushWorldPuzzle,
    state: tuple[tuple[int, int], ...],
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


def solve_with_rgd(planner: Path, puzzle_path: Path, time_limit: float) -> Trajectory:
    start = time.perf_counter()
    result = subprocess.run(
        [str(planner), "N+RGD", str(puzzle_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=time_limit + 1.0,
    )
    elapsed = time.perf_counter() - start
    plan = result.stdout.strip()
    if result.returncode != 0 or not plan or not set(plan).issubset(set(ACTION_CHARS)):
        raise RuntimeError(
            f"Failed to solve {puzzle_path}: returncode={result.returncode}, "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    puzzle = PushWorldPuzzle(str(puzzle_path))
    actions = [Actions.FROM_CHAR[ch] for ch in plan]
    if not puzzle.is_valid_plan(actions):
        raise RuntimeError(f"Planner returned invalid plan for {puzzle_path}: {plan}")
    return Trajectory(puzzle_path=puzzle_path, plan=plan, solve_time_s=elapsed)


def solve_trajectories(
    planner: Path,
    puzzle_paths: list[Path],
    time_limit: float,
    workers: int,
) -> list[Trajectory]:
    if workers <= 1:
        return [
            solve_with_rgd(planner, path, time_limit)
            for path in tqdm(puzzle_paths, desc="solve expert traces", unit="puzzle")
        ]

    trajectories: list[Trajectory | None] = [None] * len(puzzle_paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(solve_with_rgd, planner, path, time_limit): idx
            for idx, path in enumerate(puzzle_paths)
        }
        progress = tqdm(as_completed(futures), total=len(futures), desc="solve expert traces", unit="puzzle")
        for future in progress:
            idx = futures[future]
            trajectories[idx] = future.result()
    return [trajectory for trajectory in trajectories if trajectory is not None]


def train(
    model: nn.Module,
    dataset: ExpertDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    distance_loss_weight: float,
    quick_eval_paths: list[Path],
    height: int,
    width: int,
    max_steps: int,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    max_cache_entries: int,
    quick_eval_every: int,
    log_every_batches: int,
    amp: bool,
    seed: int,
    writer: object | None = None,
) -> list[float]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    losses = []
    global_step = 0
    model.train()
    progress = tqdm(range(epochs), desc="train", unit="epoch")
    for _ in progress:
        total_loss = 0.0
        total_count = 0
        batch_progress = tqdm(loader, desc=f"epoch {len(losses) + 1}/{epochs}", unit="batch", leave=False)
        for states, actions, remaining in batch_progress:
            global_step += 1
            states = states.to(device)
            actions = actions.to(device)
            remaining = remaining.to(device).clamp_max(model.distance_head.out_features - 1)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                action_logits, distance_logits = model(states)
                action_loss = nn.functional.cross_entropy(action_logits, actions)
                distance_loss = nn.functional.cross_entropy(distance_logits, remaining)
                loss = action_loss + distance_loss_weight * distance_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * states.shape[0]
            total_count += states.shape[0]
            batch_loss = float(loss.detach().cpu())
            batch_progress.set_postfix(loss=f"{batch_loss:.4f}")
            if writer is not None and log_every_batches > 0 and global_step % log_every_batches == 0:
                writer.add_scalar("train/batch_loss", batch_loss, global_step)
                writer.add_scalar("train/action_loss", float(action_loss.detach().cpu()), global_step)
                writer.add_scalar("train/distance_loss", float(distance_loss.detach().cpu()), global_step)
        epoch_loss = total_loss / max(1, total_count)
        losses.append(epoch_loss)
        if writer is not None:
            writer.add_scalar("train/loss", epoch_loss, len(losses))
            writer.add_scalar("train/global_step", global_step, len(losses))
        progress.set_postfix(loss=f"{epoch_loss:.4f}")

        if quick_eval_paths and quick_eval_every > 0 and len(losses) % quick_eval_every == 0:
            quick_eval = evaluate(
                model,
                quick_eval_paths,
                height,
                width,
                device,
                max_steps,
                beam_width,
                beam_depth,
                top_k,
                f"quick epoch {len(losses)}",
                max_cache_entries,
                leave=False,
            )
            success_rate = quick_eval["solved"] / max(1, quick_eval["total"])
            if writer is not None:
                writer.add_scalar("eval_quick/level0_solved", quick_eval["solved"], len(losses))
                writer.add_scalar("eval_quick/level0_total", quick_eval["total"], len(losses))
                writer.add_scalar("eval_quick/level0_success_rate", success_rate, len(losses))
            model.train()
    return losses


def make_tensorboard_writer(log_dir: Path | None) -> object | None:
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requested, but tensorboard is not installed. "
            "Install it with: uv pip install tensorboard"
        ) from exc
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def evaluate(
    model: nn.Module,
    puzzle_paths: list[Path],
    height: int,
    width: int,
    device: torch.device,
    max_steps: int,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    label: str,
    max_cache_entries: int,
    leave: bool = True,
) -> dict[str, object]:
    model.eval()
    solved = 0
    results = []
    encode_cache: dict[tuple[str, tuple[tuple[int, int], ...]], torch.Tensor] = {}
    start = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(puzzle_paths, desc=f"eval {label}", unit="puzzle", leave=leave)
        for path in progress:
            puzzle = PushWorldPuzzle(str(path))
            state = puzzle.initial_state
            actions: list[str] = []
            repeated_states = 0
            seen = {state}
            for _ in range(max_steps):
                if puzzle.is_goal_state(state):
                    break
                action = choose_action(
                    model,
                    puzzle,
                    state,
                    height,
                    width,
                    device,
                    beam_width=beam_width,
                    beam_depth=beam_depth,
                    top_k=top_k,
                    puzzle_key=str(path),
                    encode_cache=encode_cache,
                    max_cache_entries=max_cache_entries,
                )
                actions.append(ACTION_CHARS[action])
                state = puzzle.get_next_state(state, action)
                if state in seen:
                    repeated_states += 1
                seen.add(state)
            did_solve = puzzle.is_goal_state(state)
            solved += int(did_solve)
            elapsed = max(time.perf_counter() - start, 1e-9)
            results.append(
                {
                    "puzzle": path.name,
                    "solved": did_solve,
                    "steps": len(actions),
                    "actions": "".join(actions),
                    "repeated_states": repeated_states,
                }
            )
            rate = len(results) / elapsed
            remaining = (len(puzzle_paths) - len(results)) / rate if rate > 0 else 0.0
            progress.set_postfix(
                solved=f"{solved}/{len(results)}",
                eta_s=f"{remaining:.0f}",
            )
    return {
        "solved": solved,
        "total": len(puzzle_paths),
        "time_s": time.perf_counter() - start,
        "cache_entries": len(encode_cache),
        "results": results,
    }


def encode_cached(
    puzzle: PushWorldPuzzle,
    puzzle_key: str,
    state: tuple[tuple[int, int], ...],
    height: int,
    width: int,
    cache: dict[tuple[str, tuple[tuple[int, int], ...]], torch.Tensor],
    max_cache_entries: int,
) -> torch.Tensor:
    key = (puzzle_key, state)
    cached = cache.get(key)
    if cached is not None:
        return cached
    encoded = torch.from_numpy(encode_state(puzzle, state, height, width))
    if max_cache_entries > 0 and len(cache) < max_cache_entries:
        cache[key] = encoded
    return encoded


def predict_batch(
    model: nn.Module,
    puzzle_states: list[tuple[PushWorldPuzzle, str, tuple[tuple[int, int], ...]]],
    height: int,
    width: int,
    device: torch.device,
    encode_cache: dict[tuple[str, tuple[tuple[int, int], ...]], torch.Tensor],
    max_cache_entries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [
        encode_cached(puzzle, puzzle_key, state, height, width, encode_cache, max_cache_entries)
        for puzzle, puzzle_key, state in puzzle_states
    ]
    batch = torch.stack(encoded).to(device)
    action_logits, distance_logits = model(batch)
    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    distance_probs = torch.softmax(distance_logits, dim=-1)
    distances = torch.arange(distance_logits.shape[-1], device=device, dtype=torch.float32)
    expected_distance = torch.sum(distance_probs * distances.unsqueeze(0), dim=-1)
    return action_log_probs.cpu(), expected_distance.cpu()


def choose_action(
    model: nn.Module,
    puzzle: PushWorldPuzzle,
    state: tuple[tuple[int, int], ...],
    height: int,
    width: int,
    device: torch.device,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    puzzle_key: str,
    encode_cache: dict[tuple[str, tuple[tuple[int, int], ...]], torch.Tensor],
    max_cache_entries: int,
) -> int:
    if beam_width <= 1 or beam_depth <= 1:
        action_log_probs, _ = predict_batch(
            model,
            [(puzzle, puzzle_key, state)],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
        )
        for action in torch.argsort(action_log_probs[0], descending=True).tolist():
            if puzzle.get_next_state(state, int(action)) != state:
                return int(action)
        return int(torch.argmax(action_log_probs[0]).item())

    beams: list[tuple[tuple[tuple[int, int], ...], tuple[int, ...], float]] = [(state, (), 0.0)]
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
        )
        action_log_probs, _ = predictions
        candidates_by_state: dict[
            tuple[tuple[int, int], ...],
            tuple[tuple[tuple[int, int], ...], tuple[int, ...], float],
        ] = {}
        for beam_idx, (beam_state, path, score) in enumerate(beams):
            action_count = min(top_k, len(ACTION_NAMES))
            top_actions = torch.topk(action_log_probs[beam_idx], k=action_count).indices.tolist()
            for action in top_actions:
                next_state = puzzle.get_next_state(beam_state, int(action))
                if next_state == beam_state:
                    continue
                next_path = path + (int(action),)
                best_nonempty_path = best_nonempty_path or next_path
                next_score = score - float(action_log_probs[beam_idx, action])
                if puzzle.is_goal_state(next_state):
                    best_solved = next_path
                    break
                previous = candidates_by_state.get(next_state)
                if previous is None or next_score < previous[2]:
                    candidates_by_state[next_state] = (next_state, next_path, next_score)
            if best_solved is not None:
                break
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
        )
        del leaf_log_probs
        ranked = sorted(
            zip(candidates, leaf_distances.tolist(), strict=True),
            key=lambda item: item[0][2] + 0.15 * float(item[1]),
        )
        beams = [candidate for candidate, _ in ranked[:beam_width]]

    if beams and beams[0][1]:
        return beams[0][1][0]
    if best_nonempty_path:
        return best_nonempty_path[0]
    return Actions.LEFT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=Path,
        action="append",
        default=None,
        help="Training puzzle directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        action="append",
        default=None,
        help="Held-out Level 0 puzzle directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--level1-dir",
        type=Path,
        default=PROJECT_ROOT / "external/pushworld/benchmark/puzzles/level1",
    )
    parser.add_argument(
        "--planner",
        type=Path,
        default=PROJECT_ROOT / "external/pushworld/cpp/build/bin/run_planner",
    )
    parser.add_argument("--train-puzzles", type=int, default=5, help="Number of train puzzles to use unless --all-train is set.")
    parser.add_argument("--test-puzzles", type=int, default=10, help="Number of held-out Level 0 puzzles to evaluate unless --all-test is set.")
    parser.add_argument("--level1-puzzles", type=int, default=5, help="Number of Level 1 puzzles to evaluate unless --all-level1 is set.")
    parser.add_argument("--all-train", action="store_true", help="Use all train puzzles.")
    parser.add_argument("--all-test", action="store_true", help="Evaluate all held-out Level 0 puzzles.")
    parser.add_argument("--all-level1", action="store_true", help="Evaluate all Level 1 puzzles.")
    parser.add_argument(
        "--level0-symmetry-augment",
        action="store_true",
        help="Augment train examples in memory with Level-0 rotation/flip symmetries.",
    )
    parser.add_argument(
        "--augment-transforms",
        default="all",
        help=(
            "Comma-separated transform names, or 'all'. Available: "
            + ",".join(SYMMETRY_TRANSFORMS)
        ),
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--distance-loss-weight", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision during training.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--beam-depth", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=0, help="Run quick held-out Level 0 eval every N epochs; 0 disables it.")
    parser.add_argument("--eval-puzzles", type=int, default=50, help="Number of held-out Level 0 puzzles for periodic quick eval.")
    parser.add_argument("--log-every-batches", type=int, default=10, help="Log batch losses to TensorBoard every N optimizer steps; 0 disables batch logging.")
    parser.add_argument("--max-cache-entries", type=int, default=250_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument("--print-expert-plans", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--planner-time-limit", type=float, default=10.0)
    parser.add_argument("--planner-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.planner_workers < 1:
        raise ValueError("--planner-workers must be >= 1")

    if args.augment_transforms == "all":
        train_transforms = SYMMETRY_TRANSFORMS if args.level0_symmetry_augment else ("r0",)
    else:
        train_transforms = tuple(name.strip() for name in args.augment_transforms.split(",") if name.strip())
        unknown_transforms = sorted(set(train_transforms) - set(SYMMETRY_TRANSFORMS))
        if unknown_transforms:
            raise ValueError(f"Unknown --augment-transforms values: {unknown_transforms}")
        if not args.level0_symmetry_augment:
            train_transforms = ("r0",)

    train_dirs = args.train_dir or [PROJECT_ROOT / "data/level0/base/train"]
    test_dirs = args.test_dir or [PROJECT_ROOT / "data/level0/base/test"]
    level1_dirs = [args.level1_dir]

    if len(train_dirs) > 1 and not args.all_train and args.train_puzzles == 5:
        raise ValueError(
            "Multiple --train-dir values were provided, but --all-train was not set "
            "and --train-puzzles is still the default 5. Pass --all-train for the "
            "full run, or set --train-puzzles explicitly for a small mixed smoke."
        )

    train_paths = select_puzzles(train_dirs, args.train_puzzles, args.all_train)
    test_paths = select_puzzles(test_dirs, args.test_puzzles, args.all_test)
    level1_paths = select_puzzles(level1_dirs, args.level1_puzzles, args.all_level1)
    all_paths = train_paths + test_paths + level1_paths
    height, width = max_dimensions(all_paths)
    if args.level0_symmetry_augment:
        max_side = max(height, width)
        height = max_side
        width = max_side

    print(f"device cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train_puzzles={len(train_paths)} test_puzzles={len(test_paths)} level1={len(level1_paths)}")
    print("train_dirs=" + json.dumps([str(path) for path in train_dirs], indent=2))
    print("test_dirs=" + json.dumps([str(path) for path in test_dirs], indent=2))
    print(f"level0_symmetry_augment={args.level0_symmetry_augment} transforms={list(train_transforms)}")
    print(f"board={height}x{width} planner={args.planner} planner_workers={args.planner_workers}")
    writer = make_tensorboard_writer(args.tensorboard_log)
    if writer is not None:
        writer.add_text("config/args", json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    solve_start = time.perf_counter()
    trajectories = solve_trajectories(args.planner, train_paths, args.planner_time_limit, args.planner_workers)
    solve_time = time.perf_counter() - solve_start
    expert_plan_summary = [
        {"puzzle": t.puzzle_path.name, "plan": t.plan, "solve_time_s": round(t.solve_time_s, 4)}
        for t in trajectories
    ]
    if args.print_expert_plans:
        print("expert_plans=" + json.dumps(expert_plan_summary, indent=2))
    else:
        plan_lengths = [len(t.plan) for t in trajectories]
        print(
            "expert_trace_summary="
            + json.dumps(
                {
                    "count": len(trajectories),
                    "total_actions": sum(plan_lengths),
                    "mean_plan_len": round(sum(plan_lengths) / max(1, len(plan_lengths)), 2),
                    "max_plan_len": max(plan_lengths) if plan_lengths else 0,
                    "solve_time_s": round(solve_time, 3),
                },
                indent=2,
            )
        )

    dataset = ExpertDataset(
        trajectories,
        height=height,
        width=width,
        transforms=train_transforms,
        transform_level0_only=args.level0_symmetry_augment,
        seed=args.seed,
    )
    print(
        "dataset_summary="
        + json.dumps(
            {
                "base_examples": dataset.base_examples,
                "augmented_examples": len(dataset),
                "augmentation_factor": round(len(dataset) / max(1, dataset.base_examples), 2),
                "level0_augmented_base_examples": dataset.augmented_base_examples,
                "unaugmented_base_examples": dataset.unaugmented_base_examples,
                "transforms": list(train_transforms),
            },
            indent=2,
        )
    )
    if writer is not None:
        writer.add_scalar("data/train_puzzles", len(train_paths), 0)
        writer.add_scalar("data/base_examples", dataset.base_examples, 0)
        writer.add_scalar("data/level0_augmented_base_examples", dataset.augmented_base_examples, 0)
        writer.add_scalar("data/unaugmented_base_examples", dataset.unaugmented_base_examples, 0)
        writer.add_scalar("data/examples", len(dataset), 0)
        writer.add_scalar("data/augmentation_factor", len(dataset) / max(1, dataset.base_examples), 0)
        writer.add_scalar("time/expert_solve_s", solve_time, 0)
    model = BoardTransformerPolicy(
        channels=7,
        height=height,
        width=width,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        distance_bins=args.max_steps + 1,
    ).to(device)
    quick_eval_paths = test_paths[: args.eval_puzzles] if args.eval_every > 0 else []

    train_start = time.perf_counter()
    losses = train(
        model,
        dataset,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.distance_loss_weight,
        quick_eval_paths,
        height,
        width,
        args.max_steps,
        args.beam_width,
        args.beam_depth,
        args.top_k,
        args.max_cache_entries,
        args.eval_every,
        args.log_every_batches,
        args.amp,
        args.seed,
        writer,
    )
    train_time = time.perf_counter() - train_start

    if args.skip_train_eval:
        train_eval = {"skipped": True, "solved": None, "total": len(train_paths)}
    else:
        train_eval = evaluate(
            model,
            train_paths,
            height,
            width,
            device,
            args.max_steps,
            args.beam_width,
            args.beam_depth,
            args.top_k,
            "train",
            args.max_cache_entries,
        )
    test_eval = evaluate(
        model,
        test_paths,
        height,
        width,
        device,
        args.max_steps,
        args.beam_width,
        args.beam_depth,
        args.top_k,
        "level0 test",
        args.max_cache_entries,
    )
    level1_eval = evaluate(
        model,
        level1_paths,
        height,
        width,
        device,
        args.max_steps,
        args.beam_width,
        args.beam_depth,
        args.top_k,
        "level1",
        args.max_cache_entries,
    )

    def compact_eval(payload: dict[str, object]) -> dict[str, object]:
        if payload.get("skipped"):
            return payload
        if args.verbose:
            return payload
        return {
            "solved": payload["solved"],
            "total": payload["total"],
            "solved_puzzles": [
                result["puzzle"]
                for result in payload["results"]
                if result["solved"]
            ],
        }

    summary = {
        "examples": len(dataset),
        "base_examples": dataset.base_examples,
        "augmentation_factor": round(len(dataset) / max(1, dataset.base_examples), 2),
        "level0_augmented_base_examples": dataset.augmented_base_examples,
        "unaugmented_base_examples": dataset.unaugmented_base_examples,
        "transforms": list(train_transforms),
        "solve_time_s": round(solve_time, 3),
        "train_time_s": round(train_time, 3),
        "final_loss": round(losses[-1], 6) if losses else None,
        "beam_width": args.beam_width,
        "beam_depth": args.beam_depth,
        "top_k": args.top_k,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "layers": args.layers,
        "amp": args.amp,
        "seed": args.seed,
        "planner_workers": args.planner_workers,
        "max_cache_entries": args.max_cache_entries,
        "train_eval": compact_eval(train_eval),
        "level0_test_eval": compact_eval(test_eval),
        "level1_eval": compact_eval(level1_eval),
    }
    if writer is not None:
        writer.add_scalar("time/train_s", train_time, 0)
        writer.add_scalar("eval/train_solved", train_eval["solved"] or 0, 0)
        writer.add_scalar("eval/train_total", train_eval["total"], 0)
        writer.add_scalar("eval/level0_test_solved", test_eval["solved"], 0)
        writer.add_scalar("eval/level0_test_total", test_eval["total"], 0)
        writer.add_scalar("eval/level1_solved", level1_eval["solved"], 0)
        writer.add_scalar("eval/level1_total", level1_eval["total"], 0)
        writer.add_scalar(
            "eval/level0_test_success_rate",
            test_eval["solved"] / max(1, test_eval["total"]),
            0,
        )
        writer.add_scalar(
            "eval/level1_success_rate",
            level1_eval["solved"] / max(1, level1_eval["total"]),
            0,
        )
        writer.flush()
        writer.close()
    print("summary=" + json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "height": height,
                "width": width,
                "channels": 7,
                "args": vars(args),
            },
            args.model_output,
        )
        print(f"wrote {args.model_output}")


if __name__ == "__main__":
    main()
