from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from pushworld_study.paths import PROJECT_ROOT, ensure_upstream_pushworld_on_path


ensure_upstream_pushworld_on_path()

from pushworld.transform import get_puzzle_transforms  # noqa: E402


def sorted_puzzles(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.pwp") if p.is_file())


def build_dataset(input_dirs: list[Path], output_dir: Path, overwrite: bool) -> dict[str, object]:
    source_paths: list[Path] = []
    seen: set[Path] = set()
    for input_dir in input_dirs:
        for puzzle_path in sorted_puzzles(input_dir):
            resolved = puzzle_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            source_paths.append(puzzle_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    transforms_seen: set[str] = set()

    for puzzle_path in tqdm(source_paths, desc="augment level0", unit="puzzle"):
        text = puzzle_path.read_text(encoding="utf-8")
        try:
            rel_parent = puzzle_path.parent.relative_to(PROJECT_ROOT / "data/level0")
        except ValueError:
            rel_parent = Path(puzzle_path.parent.name)

        for transform_name, transformed_text in get_puzzle_transforms(text).items():
            transforms_seen.add(transform_name)
            target = output_dir / rel_parent / f"{puzzle_path.stem}__{transform_name}.pwp"
            if target.exists() and not overwrite:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(transformed_text.rstrip() + "\n", encoding="utf-8")
            written += 1

    summary = {
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "source_puzzles": len(source_paths),
        "transforms": sorted(transforms_seen),
        "expected_augmented_puzzles": len(source_paths) * len(transforms_seen),
        "written": written,
        "skipped_existing": skipped,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = build_dataset(args.input_dir, args.output_dir, args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
