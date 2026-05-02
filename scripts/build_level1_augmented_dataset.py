from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

from pushworld_study.paths import PROJECT_ROOT, ensure_upstream_pushworld_on_path


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402
from pushworld.transform import get_puzzle_transforms  # noqa: E402


ACTION_CHARS = set(Actions.FROM_CHAR)
ADVANCED_METHODS = (
    "wall_toggle",
    "remove_movable",
    "add_movable",
    "add_goal",
    "move_goal_shift",
    "move_goal_random",
)


@dataclass(frozen=True)
class Candidate:
    split: str
    source_group: str
    source_puzzle: str
    kind: str
    transform: str
    variant_index: int
    text: str


def sorted_puzzles(path: Path) -> list[Path]:
    return sorted(path.glob("*.pwp"), key=lambda item: item.name.casefold())


def normalize_puzzle_text(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.split())


def puzzle_hash(text: str) -> str:
    return hashlib.sha1(normalize_puzzle_text(text).encode("utf-8")).hexdigest()[:12]


def canonical_symmetry_hash(text: str) -> str:
    normalized = [
        normalize_puzzle_text(transformed)
        for transformed in get_puzzle_transforms(text).values()
    ]
    return hashlib.sha1(min(normalized).encode("utf-8")).hexdigest()[:12]


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "puzzle"


def parse_grid(text: str) -> list[list[str]]:
    rows = [line.split() for line in text.splitlines() if line.split()]
    if not rows:
        raise ValueError("empty puzzle")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ragged puzzle")
    return rows


def format_grid(rows: list[list[str]]) -> str:
    return "\n".join("  ".join(row) for row in rows)


def split_cell(cell: str) -> list[str]:
    return [] if cell == "." else cell.split("+")


def join_cell(tokens: list[str]) -> str:
    return "+".join(tokens) if tokens else "."


def token_prefix(token: str) -> str:
    return re.match(r"[A-Za-z]+", token).group(0).upper()


def token_suffix(token: str) -> str:
    match = re.match(r"[A-Za-z]+(.+)", token)
    return match.group(1) if match else ""


def collect_token_ids(rows: list[list[str]], prefix: str) -> set[str]:
    ids = set()
    for row in rows:
        for cell in row:
            for token in split_cell(cell):
                if token_prefix(token) == prefix:
                    ids.add(token_suffix(token))
    return ids


def collect_token_cells(rows: list[list[str]], token_name: str) -> list[tuple[int, int]]:
    cells = []
    target = token_name.upper()
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            if any(token.upper() == target for token in split_cell(cell)):
                cells.append((row_idx, col_idx))
    return cells


def remove_token(rows: list[list[str]], token_name: str) -> None:
    target = token_name.upper()
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            tokens = [token for token in split_cell(cell) if token.upper() != target]
            rows[row_idx][col_idx] = join_cell(tokens)


def add_token(rows: list[list[str]], row_idx: int, col_idx: int, token_name: str) -> None:
    tokens = split_cell(rows[row_idx][col_idx])
    tokens.append(token_name)
    rows[row_idx][col_idx] = join_cell(tokens)


def is_empty_cell(rows: list[list[str]], row_idx: int, col_idx: int) -> bool:
    return rows[row_idx][col_idx] == "."


def shape_offsets(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    return [(row - min_row, col - min_col) for row, col in cells]


def valid_shape_placement(
    rows: list[list[str]],
    offsets: list[tuple[int, int]],
    origin: tuple[int, int],
) -> bool:
    height = len(rows)
    width = len(rows[0])
    origin_row, origin_col = origin
    for row_offset, col_offset in offsets:
        row_idx = origin_row + row_offset
        col_idx = origin_col + col_offset
        if not (0 <= row_idx < height and 0 <= col_idx < width):
            return False
        if not is_empty_cell(rows, row_idx, col_idx):
            return False
    return True


def random_shape_placement(
    rows: list[list[str]],
    offsets: list[tuple[int, int]],
    rng: random.Random,
    candidate_origins: list[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    if candidate_origins is None:
        candidate_origins = [
            (row_idx, col_idx)
            for row_idx in range(len(rows))
            for col_idx in range(len(rows[0]))
        ]
    rng.shuffle(candidate_origins)
    for origin in candidate_origins:
        if valid_shape_placement(rows, offsets, origin):
            return origin
    return None


def perturb_wall_toggle(text: str, rng: random.Random) -> str | None:
    rows = parse_grid(text)
    mutable_cells = [
        (row_idx, col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, token in enumerate(row)
        if token in {".", "W"}
    ]
    if not mutable_cells:
        return None
    row_idx, col_idx = rng.choice(mutable_cells)
    rows[row_idx][col_idx] = "W" if rows[row_idx][col_idx] == "." else "."
    return format_grid(rows)


def perturb_remove_movable(text: str, rng: random.Random) -> str | None:
    rows = parse_grid(text)
    movable_ids = collect_token_ids(rows, "M")
    goal_ids = collect_token_ids(rows, "G")
    obstacle_ids = sorted(movable_ids - goal_ids)
    if not obstacle_ids:
        return None
    remove_token(rows, "M" + rng.choice(obstacle_ids))
    return format_grid(rows)


def perturb_add_movable(text: str, rng: random.Random, max_cells: int) -> str | None:
    rows = parse_grid(text)
    movable_ids = collect_token_ids(rows, "M")
    numeric_ids = [int(item) for item in movable_ids if item.isdigit()]
    next_id = max(numeric_ids, default=-1) + 1
    token_name = f"M{next_id}"
    empty_cells = [
        (row_idx, col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, cell in enumerate(row)
        if cell == "."
    ]
    if not empty_cells:
        return None
    size = rng.randint(1, max(1, max_cells))
    if size == 1:
        chosen = [rng.choice(empty_cells)]
    else:
        pairs = []
        empty_set = set(empty_cells)
        for row_idx, col_idx in empty_cells:
            for drow, dcol in ((1, 0), (0, 1)):
                neighbor = (row_idx + drow, col_idx + dcol)
                if neighbor in empty_set:
                    pairs.append(((row_idx, col_idx), neighbor))
        if not pairs:
            chosen = [rng.choice(empty_cells)]
        else:
            chosen = list(rng.choice(pairs))
    for row_idx, col_idx in chosen:
        add_token(rows, row_idx, col_idx, token_name)
    return format_grid(rows)


def perturb_add_goal(text: str, rng: random.Random) -> str | None:
    rows = parse_grid(text)
    movable_ids = collect_token_ids(rows, "M")
    goal_ids = collect_token_ids(rows, "G")
    obstacle_ids = sorted(movable_ids - goal_ids)
    rng.shuffle(obstacle_ids)
    for movable_id in obstacle_ids:
        movable_cells = collect_token_cells(rows, "M" + movable_id)
        if not movable_cells:
            continue
        offsets = shape_offsets(movable_cells)
        origin = random_shape_placement(rows, offsets, rng)
        if origin is None:
            continue
        for row_offset, col_offset in offsets:
            add_token(rows, origin[0] + row_offset, origin[1] + col_offset, "G" + movable_id)
        return format_grid(rows)
    return None


def perturb_move_goal(
    text: str,
    rng: random.Random,
    *,
    local_shift: bool,
    max_shift: int,
) -> str | None:
    rows = parse_grid(text)
    goal_ids = sorted(collect_token_ids(rows, "G"))
    rng.shuffle(goal_ids)
    for goal_id in goal_ids:
        goal_cells = collect_token_cells(rows, "G" + goal_id)
        if not goal_cells:
            continue
        offsets = shape_offsets(goal_cells)
        old_origin = (min(row for row, _ in goal_cells), min(col for _, col in goal_cells))
        remove_token(rows, "G" + goal_id)
        if local_shift:
            candidate_origins = [
                (old_origin[0] + drow, old_origin[1] + dcol)
                for drow in range(-max_shift, max_shift + 1)
                for dcol in range(-max_shift, max_shift + 1)
                if drow != 0 or dcol != 0
            ]
        else:
            candidate_origins = None
        origin = random_shape_placement(rows, offsets, rng, candidate_origins)
        if origin is not None:
            for row_offset, col_offset in offsets:
                add_token(rows, origin[0] + row_offset, origin[1] + col_offset, "G" + goal_id)
            return format_grid(rows)
        for row_idx, col_idx in goal_cells:
            add_token(rows, row_idx, col_idx, "G" + goal_id)
    return None


def perturb_advanced(
    text: str,
    rng: random.Random,
    methods: tuple[str, ...],
    max_new_movable_cells: int,
    goal_shift_radius: int,
) -> tuple[str, str] | None:
    method = rng.choice(methods)
    if method == "wall_toggle":
        perturbed = perturb_wall_toggle(text, rng)
    if method == "remove_movable":
        perturbed = perturb_remove_movable(text, rng)
    if method == "add_movable":
        perturbed = perturb_add_movable(text, rng, max_new_movable_cells)
    if method == "add_goal":
        perturbed = perturb_add_goal(text, rng)
    if method == "move_goal_shift":
        perturbed = perturb_move_goal(text, rng, local_shift=True, max_shift=goal_shift_radius)
    if method == "move_goal_random":
        perturbed = perturb_move_goal(text, rng, local_shift=False, max_shift=goal_shift_radius)
    if method not in ADVANCED_METHODS:
        raise ValueError(f"Unknown advanced method: {method}")
    if perturbed is None:
        return None
    return method, perturbed


def perturb_advanced_stack(
    text: str,
    rng: random.Random,
    methods: tuple[str, ...],
    max_new_movable_cells: int,
    goal_shift_radius: int,
    stack_depth: int,
) -> tuple[str, str] | None:
    current = text
    applied_methods = []
    for _ in range(stack_depth):
        proposal = perturb_advanced(
            current,
            rng,
            methods,
            max_new_movable_cells,
            goal_shift_radius,
        )
        if proposal is None:
            return None
        method, current = proposal
        applied_methods.append(method)
    return "+".join(applied_methods), current


def solve_with_rgd(planner: Path, text: str, time_limit: float) -> tuple[bool, str, float, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".pwp", encoding="utf-8") as puzzle_file:
        puzzle_file.write(normalize_puzzle_text(text) + "\n")
        puzzle_file.flush()
        try:
            puzzle = PushWorldPuzzle(puzzle_file.name)
        except Exception as exc:  # noqa: BLE001
            return False, "", 0.0, f"parse_error:{exc}"

        start = time.perf_counter()
        try:
            result = subprocess.run(
                [str(planner), "N+RGD", puzzle_file.name],
                check=False,
                capture_output=True,
                text=True,
                timeout=time_limit + 1.0,
            )
        except subprocess.TimeoutExpired:
            return False, "", time.perf_counter() - start, "timeout"

        elapsed = time.perf_counter() - start
        plan = result.stdout.strip()
        if result.returncode != 0:
            return False, plan, elapsed, f"planner_returncode:{result.returncode}"
        if not plan or not set(plan).issubset(ACTION_CHARS):
            return False, plan, elapsed, "invalid_plan_text"
        actions = [Actions.FROM_CHAR[ch] for ch in plan]
        if not puzzle.is_valid_plan(actions):
            return False, plan, elapsed, "invalid_plan"
        return True, plan, elapsed, "ok"


def split_source_groups(
    groups: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    train_count = round(len(group_ids) * train_ratio)
    val_count = round(len(group_ids) * val_ratio)
    if train_count + val_count > len(group_ids):
        raise ValueError("train_ratio + val_ratio leaves no room for test split")

    split_by_group = {}
    for idx, group_id in enumerate(group_ids):
        if idx < train_count:
            split_by_group[group_id] = "train"
        elif idx < train_count + val_count:
            split_by_group[group_id] = "val"
        else:
            split_by_group[group_id] = "test"
    return split_by_group


def write_candidate(output_dir: Path, candidate: Candidate) -> Path:
    filename = (
        f"{slugify(Path(candidate.source_puzzle).stem)}"
        f"__{candidate.kind}"
        f"__{candidate.transform}"
        f"__{candidate.variant_index:03d}"
        f"__{puzzle_hash(candidate.text)}.pwp"
    )
    path = output_dir / candidate.split / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalize_puzzle_text(candidate.text) + "\n", encoding="utf-8")
    return path


def verify_base_layout(
    *,
    planner: Path,
    text: str,
    planner_time_limit: float,
    max_plan_len: int,
    skip_verification: bool,
) -> tuple[bool, str, float, str]:
    if skip_verification:
        return True, "", 0.0, "skipped"
    ok, plan, solve_time_s, status = solve_with_rgd(planner, text, planner_time_limit)
    if ok and len(plan) > max_plan_len:
        return False, plan, solve_time_s, "plan_too_long"
    return ok, plan, solve_time_s, status


def emit_symmetry_variants(
    *,
    output_dir: Path,
    source_manifest_base: dict[str, str],
    base_kind: str,
    base_variant_index: int,
    base_text: str,
    base_plan: str,
    base_solve_time_s: float,
    seen_hashes: set[str],
    manifest: list[dict[str, object]],
) -> int:
    accepted = 0
    base_hash = puzzle_hash(base_text)
    for transform_name, transformed_text in get_puzzle_transforms(base_text).items():
        normalized = normalize_puzzle_text(transformed_text)
        key = puzzle_hash(normalized)
        if key in seen_hashes:
            continue
        seen_hashes.add(key)

        candidate = Candidate(
            split=source_manifest_base["split"],
            source_group=source_manifest_base["source_group"],
            source_puzzle=source_manifest_base["source_puzzle"],
            kind=base_kind,
            transform=transform_name,
            variant_index=base_variant_index,
            text=normalized,
        )
        out_path = write_candidate(output_dir, candidate)
        manifest.append(
            {
                **source_manifest_base,
                "kind": candidate.kind,
                "transform": candidate.transform,
                "variant_index": candidate.variant_index,
                "path": str(out_path.relative_to(output_dir)),
                "base_hash": base_hash,
                "hash": key,
                "base_plan": base_plan,
                "base_plan_len": len(base_plan),
                "base_solve_time_s": round(base_solve_time_s, 4),
            }
        )
        accepted += 1
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/augmented/level1_verified",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-originals", type=int, default=None)
    parser.add_argument("--planner-time-limit", type=float, default=10.0)
    parser.add_argument("--max-plan-len", type=int, default=200)
    parser.add_argument("--skip-symmetry-verification", action="store_true")
    parser.add_argument("--advanced-per-original", type=int, default=0)
    parser.add_argument("--advanced-max-attempts-per-original", type=int, default=50)
    parser.add_argument(
        "--advanced-methods",
        default=",".join(ADVANCED_METHODS),
        help=(
            "Comma-separated advanced proposal methods. Available: "
            + ",".join(ADVANCED_METHODS)
        ),
    )
    parser.add_argument("--max-new-movable-cells", type=int, default=2)
    parser.add_argument("--goal-shift-radius", type=int, default=2)
    parser.add_argument(
        "--advanced-stack-depth",
        type=int,
        default=1,
        help="Number of advanced generators to apply sequentially per candidate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must be non-negative and leave a test split")
    advanced_methods = tuple(name.strip() for name in args.advanced_methods.split(",") if name.strip())
    unknown_methods = sorted(set(advanced_methods) - set(ADVANCED_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown --advanced-methods values: {unknown_methods}")
    if args.advanced_per_original > 0 and not advanced_methods:
        raise ValueError("--advanced-per-original requires at least one --advanced-methods value")
    if args.advanced_stack_depth < 1:
        raise ValueError("--advanced-stack-depth must be >= 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite to append/overwrite files")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    source_paths = sorted_puzzles(args.level1_dir)
    if args.limit_originals is not None:
        source_paths = source_paths[: args.limit_originals]
    if not source_paths:
        raise ValueError(f"No .pwp files found in {args.level1_dir}")

    groups: dict[str, list[Path]] = {}
    source_texts = {}
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        group_id = canonical_symmetry_hash(text)
        groups.setdefault(group_id, []).append(path)
        source_texts[path.name] = text

    split_by_group = split_source_groups(groups, args.train_ratio, args.val_ratio, args.seed)
    rng = random.Random(args.seed)
    seen_hashes: set[str] = set()
    manifest = []
    rejected = []

    source_progress = tqdm(sorted(groups), desc="augment source groups", unit="group")
    for group_id in source_progress:
        split = split_by_group[group_id]
        for source_path in groups[group_id]:
            source_text = source_texts[source_path.name]
            source_manifest_base = {
                "source_group": group_id,
                "source_puzzle": source_path.name,
                "split": split,
            }

            ok, plan, solve_time_s, status = verify_base_layout(
                planner=args.planner,
                text=source_text,
                planner_time_limit=args.planner_time_limit,
                max_plan_len=args.max_plan_len,
                skip_verification=args.skip_symmetry_verification,
            )
            if ok:
                emit_symmetry_variants(
                    output_dir=args.output_dir,
                    source_manifest_base=source_manifest_base,
                    base_kind="original",
                    base_variant_index=0,
                    base_text=source_text,
                    base_plan=plan,
                    base_solve_time_s=solve_time_s,
                    seen_hashes=seen_hashes,
                    manifest=manifest,
                )
            else:
                rejected.append(
                    {
                        **source_manifest_base,
                        "kind": "original",
                        "transform": "base",
                        "variant_index": 0,
                        "base_hash": puzzle_hash(source_text),
                        "hash": puzzle_hash(source_text),
                        "status": status,
                        "plan_len": len(plan),
                        "solve_time_s": round(solve_time_s, 4),
                    }
                )

            accepted_advanced = 0
            attempts = 0
            advanced_progress = tqdm(
                total=args.advanced_per_original,
                desc=f"advanced {source_path.stem[:24]}",
                unit="accepted",
                leave=False,
            )
            while (
                accepted_advanced < args.advanced_per_original
                and attempts < args.advanced_max_attempts_per_original
            ):
                attempts += 1
                advanced_progress.set_postfix(attempts=attempts, rejected=len(rejected))
                proposal = perturb_advanced_stack(
                    source_text,
                    rng,
                    advanced_methods,
                    args.max_new_movable_cells,
                    args.goal_shift_radius,
                    args.advanced_stack_depth,
                )
                if proposal is None:
                    continue
                method, perturbed = proposal
                base_hash = puzzle_hash(perturbed)
                if base_hash in seen_hashes:
                    continue
                ok, plan, solve_time_s, status = verify_base_layout(
                    planner=args.planner,
                    text=perturbed,
                    planner_time_limit=args.planner_time_limit,
                    max_plan_len=args.max_plan_len,
                    skip_verification=False,
                )
                if not ok:
                    rejected.append(
                        {
                            **source_manifest_base,
                            "kind": method,
                            "transform": "base",
                            "variant_index": accepted_advanced + 1,
                            "base_hash": base_hash,
                            "hash": base_hash,
                            "status": status,
                            "plan_len": len(plan),
                            "solve_time_s": round(solve_time_s, 4),
                        }
                    )
                    seen_hashes.add(base_hash)
                    continue
                accepted_count = emit_symmetry_variants(
                    output_dir=args.output_dir,
                    source_manifest_base=source_manifest_base,
                    base_kind=method,
                    base_variant_index=accepted_advanced + 1,
                    base_text=perturbed,
                    base_plan=plan,
                    base_solve_time_s=solve_time_s,
                    seen_hashes=seen_hashes,
                    manifest=manifest,
                )
                if accepted_count > 0:
                    accepted_advanced += 1
                    advanced_progress.update(1)
            advanced_progress.close()

    manifest_path = args.output_dir / "manifest.json"
    rejected_path = args.output_dir / "rejected.json"
    summary_path = args.output_dir / "summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected, indent=2) + "\n", encoding="utf-8")

    split_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    rejected_kind_counts: dict[str, int] = {}
    source_split_counts: dict[str, int] = {}
    for row in manifest:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        kind_counts[row["kind"]] = kind_counts.get(row["kind"], 0) + 1
    for row in rejected:
        rejected_kind_counts[row["kind"]] = rejected_kind_counts.get(row["kind"], 0) + 1
    for group_id, paths in groups.items():
        split = split_by_group[group_id]
        source_split_counts[split] = source_split_counts.get(split, 0) + len(paths)

    summary = {
        "source_puzzles": len(source_paths),
        "source_groups": len(groups),
        "source_split_counts": source_split_counts,
        "generated": len(manifest),
        "rejected": len(rejected),
        "split_counts": split_counts,
        "kind_counts": kind_counts,
        "rejected_kind_counts": rejected_kind_counts,
        "advanced_methods": list(advanced_methods),
        "advanced_per_original": args.advanced_per_original,
        "advanced_max_attempts_per_original": args.advanced_max_attempts_per_original,
        "advanced_stack_depth": args.advanced_stack_depth,
        "output_dir": str(args.output_dir),
        "augmentation_order": "verify original or advanced base layout once, then emit 8 symmetry transforms without extra RGD checks",
        "leakage_guard": "split source symmetry groups before augmentation; generated variants inherit source split",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
