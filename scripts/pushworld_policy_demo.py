from __future__ import annotations

import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from pushworld_study.paths import PROJECT_ROOT, ensure_upstream_pushworld_on_path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_planner_imitation_smoke import (
    ACTION_CHARS,
    BoardTransformerPolicy,
    choose_action,
    select_puzzles,
)


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "models/planner_imitation_level0_base_small.pt"
DEFAULT_PLANNER = PROJECT_ROOT / "external/pushworld/cpp/build/bin/run_planner"
LEVEL0_BASE_TEST = PROJECT_ROOT / "data/level0/base/test"
LEVEL1 = PROJECT_ROOT / "external/pushworld/benchmark/puzzles/level1"
LEGEND = {
    ".": "empty cell",
    "W": "wall",
    "A": "agent cell",
    "AW": "agent-only wall / gate",
    "M0, M1, ...": "movable object cells; same id = same rigid object",
    "G0, G1, ...": "goal cells; G0 is target for M0, G1 for M1, etc.",
    "G0+M1": "multiple annotations in one cell; here goal G0 overlaps movable M1",
}


def numeric_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    return (stem[: -len(suffix)], int(suffix)) if suffix.isdigit() else (stem, -1)


def list_puzzles(path: Path) -> list[Path]:
    return sorted(path.glob("*.pwp"), key=numeric_key)


def format_pwp_text(text: str) -> str:
    token_rows = []
    for raw_line in text.strip().splitlines():
        cells = raw_line.split()
        if not cells:
            continue
        token_rows.append(cells)
    if not token_rows:
        return ""
    cell_width = max(2, *(len(cell) for row in token_rows for cell in row))
    rows = ["  ".join(cell.ljust(cell_width) for cell in cells).rstrip() for cells in token_rows]
    return "\n".join(rows) + ("\n" if rows else "")


def set_editor_text(path: Path) -> None:
    st.session_state.selected_puzzle = path
    st.session_state.puzzle_text = format_pwp_text(path.read_text(encoding="utf-8"))
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.session_state.pop("last_rollout", None)
    st.session_state.pop("last_rgd", None)


def set_editor_text_value(text: str) -> None:
    st.session_state.puzzle_text = format_pwp_text(text)
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.session_state.pop("last_rollout", None)
    st.session_state.pop("last_rgd", None)


@st.cache_resource(show_spinner="Loading policy checkpoint...")
def load_policy(checkpoint_path: str, device_name: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    args = checkpoint.get("args", {})
    height = int(checkpoint["height"])
    width = int(checkpoint["width"])
    d_model = int(args.get("d_model", state_dict["token_proj.weight"].shape[0]))
    nhead = int(args.get("nhead", 4))
    layers = int(args.get("layers", 1))
    distance_bins = int(state_dict["distance_head.weight"].shape[0])

    device = torch.device(device_name)
    model = BoardTransformerPolicy(
        channels=7,
        height=height,
        width=width,
        d_model=d_model,
        nhead=nhead,
        layers=layers,
        distance_bins=distance_bins,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, height, width, device


def write_temp_puzzle(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".pwp",
        prefix="pushworld_demo_",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        handle.write(text.strip() + "\n")
    return Path(handle.name)


def load_puzzle_from_text(text: str) -> tuple[PushWorldPuzzle | None, Path | None, str | None]:
    try:
        path = write_temp_puzzle(text)
        return PushWorldPuzzle(str(path)), path, None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def rollout_policy(
    model: BoardTransformerPolicy,
    puzzle: PushWorldPuzzle,
    puzzle_key: str,
    height: int,
    width: int,
    device: torch.device,
    max_steps: int,
    beam_width: int,
    beam_depth: int,
    top_k: int,
) -> dict[str, object]:
    state = puzzle.initial_state
    frames = [puzzle.render(state)]
    rows = []
    seen = {state}
    repeated_states = 0
    encode_cache: dict[tuple[str, tuple[tuple[int, int], ...]], torch.Tensor] = {}

    start = time.perf_counter()
    with torch.inference_mode():
        for step_idx in range(max_steps):
            if puzzle.is_goal_state(state):
                break
            achieved_before = puzzle.count_achieved_goals(state)
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
                puzzle_key=puzzle_key,
                encode_cache=encode_cache,
                max_cache_entries=50_000,
            )
            next_state = puzzle.get_next_state(state, action)
            no_op = next_state == state
            repeated = next_state in seen
            repeated_states += int(repeated)
            rows.append(
                {
                    "step": step_idx + 1,
                    "action": ACTION_CHARS[action],
                    "action_name": ["left", "right", "up", "down"][action],
                    "no_op": no_op,
                    "repeated_state": repeated,
                    "goals_before": achieved_before,
                    "goals_after": puzzle.count_achieved_goals(next_state),
                }
            )
            state = next_state
            seen.add(state)
            frames.append(puzzle.render(state))
    elapsed = time.perf_counter() - start
    return {
        "solved": puzzle.is_goal_state(state),
        "steps": len(rows),
        "actions": "".join(row["action"] for row in rows),
        "rows": rows,
        "frames": frames,
        "repeated_states": repeated_states,
        "time_s": elapsed,
    }


def run_rgd(planner: Path, puzzle_path: Path, time_limit: float) -> dict[str, object]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [str(planner), "N+RGD", str(puzzle_path)],
            capture_output=True,
            text=True,
            timeout=time_limit + 1.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"solved": False, "plan": "", "time_s": time.perf_counter() - start, "status": "timeout"}

    elapsed = time.perf_counter() - start
    plan = result.stdout.strip()
    solved = result.returncode == 0 and bool(plan) and set(plan).issubset(set(ACTION_CHARS))
    return {
        "solved": solved,
        "plan": plan if solved else "",
        "time_s": elapsed,
        "status": "solved" if solved else result.stdout.strip() or result.stderr.strip() or "failed",
    }


def render_rollout_view(result: dict[str, object], fps: float) -> None:
    frames = result["frames"]
    if not frames:
        return
    frame_placeholder = st.empty()
    max_idx = len(frames) - 1
    selected = st.slider("Step", min_value=0, max_value=max_idx, value=0)
    frame_placeholder.image(frames[selected], caption=f"Step {selected}", use_container_width=False)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Play rollout"):
            delay = 1.0 / max(fps, 0.1)
            for idx, frame in enumerate(frames):
                frame_placeholder.image(frame, caption=f"Step {idx}", use_container_width=False)
                time.sleep(delay)
    with col_b:
        st.metric("Playback FPS", f"{fps:.1f}")


def main() -> None:
    st.set_page_config(page_title="PushWorld Policy Demo", layout="wide")
    st.markdown(
        """
        <style>
        textarea {
            font-family: "JetBrains Mono", "Fira Code", "Cascadia Mono",
                         "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
            font-size: 14px !important;
            line-height: 1.35 !important;
            white-space: pre !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("PushWorld Planner-Imitation Policy Demo")

    if "selected_puzzle" not in st.session_state:
        st.session_state.selected_puzzle = None
    if "puzzle_text" not in st.session_state:
        default = list_puzzles(LEVEL0_BASE_TEST)[0]
        set_editor_text(default)

    with st.sidebar:
        st.header("Model")
        checkpoint = Path(st.text_input("Checkpoint", str(DEFAULT_CHECKPOINT)))
        device_default = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = st.selectbox("Device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"], index=0 if device_default == "cuda" else 0)

        st.header("Puzzle")
        split = st.selectbox("Split", ["Level 0 base test", "Level 1"])
        puzzle_dir = LEVEL0_BASE_TEST if split == "Level 0 base test" else LEVEL1
        puzzles = list_puzzles(puzzle_dir)
        names = [path.name for path in puzzles]

        if st.button("Random puzzle"):
            path = random.choice(puzzles)
            set_editor_text(path)
            st.rerun()

        selected_name = st.selectbox("Puzzle", names, index=names.index(st.session_state.selected_puzzle.name) if st.session_state.selected_puzzle in puzzles else 0)
        selected_path = puzzle_dir / selected_name
        if selected_path != st.session_state.selected_puzzle:
            set_editor_text(selected_path)
            st.rerun()

        st.header("Rollout")
        max_steps = st.number_input("Max steps", min_value=1, max_value=500, value=100, step=10)
        beam_width = st.number_input("Beam width", min_value=1, max_value=64, value=8, step=1)
        beam_depth = st.number_input("Beam depth", min_value=1, max_value=32, value=8, step=1)
        top_k = st.number_input("Top-k actions", min_value=1, max_value=4, value=3, step=1)
        fps = st.slider("Playback FPS", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

        st.header("Planner")
        compare_rgd = st.checkbox("Compare with RGD", value=True)
        planner_timeout = st.number_input("RGD timeout, sec", min_value=0.1, max_value=30.0, value=3.0, step=0.5)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Editable `.pwp`")
        with st.expander("Cell legend", expanded=True):
            st.table(pd.DataFrame([{"token": key, "meaning": value} for key, value in LEGEND.items()]))
            st.caption("Cells are whitespace-separated. Keep object and goal ids consistent when editing.")
        puzzle_text = st.text_area(
            "Puzzle text",
            value=st.session_state.puzzle_text,
            height=260,
            key=f"puzzle_text_area_{st.session_state.get('editor_version', 0)}",
        )
        col_fmt, col_load = st.columns(2)
        with col_fmt:
            if st.button("Align cells"):
                set_editor_text_value(puzzle_text)
                st.rerun()
        with col_load:
            if st.button("Load edited puzzle"):
                set_editor_text_value(puzzle_text)
                st.rerun()

    puzzle, temp_path, error = load_puzzle_from_text(st.session_state.puzzle_text)
    if error is not None or puzzle is None or temp_path is None:
        st.error(f"Could not parse puzzle: {error}")
        return

    with right:
        st.subheader("Initial State")
        st.image(puzzle.render(puzzle.initial_state), use_container_width=False)
        st.caption(f"Dimensions: {puzzle.dimensions[0]}x{puzzle.dimensions[1]}, goals: {len(puzzle.goal_state)}")

    run_clicked = st.button("Run model rollout", type="primary")
    if run_clicked:
        model, height, width, device = load_policy(str(checkpoint), device_name)
        if puzzle.dimensions[1] > height or puzzle.dimensions[0] > width:
            st.error(
                f"Puzzle is larger than checkpoint padding: puzzle={puzzle.dimensions}, "
                f"checkpoint_width_height=({width}, {height})"
            )
            return

        with st.spinner("Running policy rollout..."):
            result = rollout_policy(
                model,
                puzzle,
                str(temp_path),
                height,
                width,
                device,
                int(max_steps),
                int(beam_width),
                int(beam_depth),
                int(top_k),
            )
        st.session_state.last_rollout = result
        st.session_state.last_temp_path = temp_path

        if compare_rgd:
            with st.spinner("Running RGD comparison..."):
                st.session_state.last_rgd = run_rgd(DEFAULT_PLANNER, temp_path, float(planner_timeout))
        else:
            st.session_state.last_rgd = None

    if st.session_state.get("last_rollout") is not None:
        result = st.session_state.last_rollout
        st.subheader("Model Rollout")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Solved", "yes" if result["solved"] else "no")
        col2.metric("Steps", result["steps"])
        col3.metric("Repeated states", result["repeated_states"])
        col4.metric("Inference time", f"{result['time_s']:.2f}s")

        st.code(result["actions"] or "(no actions)", language="text")
        render_rollout_view(result, float(fps))
        st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)

    if st.session_state.get("last_rgd") is not None:
        rgd = st.session_state.last_rgd
        st.subheader("RGD Comparison")
        col1, col2, col3 = st.columns(3)
        col1.metric("RGD status", rgd["status"])
        col2.metric("Plan length", len(rgd["plan"]))
        col3.metric("Solve time", f"{rgd['time_s']:.3f}s")
        st.code(rgd["plan"] or "(no plan)", language="text")


if __name__ == "__main__":
    main()
