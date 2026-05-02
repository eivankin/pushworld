from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


SYSTEM_PROMPT = """You're a helpful assistant. You always respond by first wrapping your thoughts in <think>...</think>, then giving your answer in <answer>...</answer>. Max response length: 200 words (tokens)."""

USER_HEADER = """You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer should be a sequence of actions, like <answer>Right || Right || Up</answer>

The meaning of each symbol in the state is:
#: wall, _: empty, O: target, ✓: box on target, X: box, P: player, S: player on target

Your available actions are:
Up, Down, Left, Right
You can make up to 10 actions, separated by the action separator " || "
"""


@dataclass(frozen=True)
class Case:
    name: str
    grid: tuple[str, ...]


CASES = [
    Case(
        name="adjacent_push_right",
        grid=(
            "#####",
            "#PXO#",
            "#####",
        ),
    ),
    Case(
        name="corridor_walk_then_push",
        grid=(
            "#######",
            "#P_XO_#",
            "#######",
        ),
    ),
    Case(
        name="approach_from_above",
        grid=(
            "#####",
            "#P__#",
            "#_X_#",
            "#_O_#",
            "#####",
        ),
    ),
]


class SokobanState:
    def __init__(self, grid: tuple[str, ...]) -> None:
        self.height = len(grid)
        self.width = max(len(row) for row in grid)
        self.walls: set[tuple[int, int]] = set()
        self.targets: set[tuple[int, int]] = set()
        self.boxes: set[tuple[int, int]] = set()
        self.player = (-1, -1)

        for y, row in enumerate(grid):
            for x, char in enumerate(row):
                if char == "#":
                    self.walls.add((x, y))
                elif char == "O":
                    self.targets.add((x, y))
                elif char == "X":
                    self.boxes.add((x, y))
                elif char == "P":
                    self.player = (x, y)
                elif char == "S":
                    self.player = (x, y)
                    self.targets.add((x, y))
                elif char == "✓":
                    self.boxes.add((x, y))
                    self.targets.add((x, y))
        if self.player == (-1, -1):
            raise ValueError("Grid has no player.")

    def render(self) -> str:
        rows = []
        for y in range(self.height):
            cells = []
            for x in range(self.width):
                pos = (x, y)
                if pos in self.walls:
                    cells.append("#")
                elif pos == self.player and pos in self.targets:
                    cells.append("S")
                elif pos == self.player:
                    cells.append("P")
                elif pos in self.boxes and pos in self.targets:
                    cells.append("✓")
                elif pos in self.boxes:
                    cells.append("X")
                elif pos in self.targets:
                    cells.append("O")
                else:
                    cells.append("_")
            rows.append("".join(cells))
        return "\n".join(rows)

    def solved(self) -> bool:
        return bool(self.boxes) and self.boxes <= self.targets

    def step(self, action: str) -> bool:
        dx, dy = ACTIONS[action.lower()]
        px, py = self.player
        next_pos = (px + dx, py + dy)
        if next_pos in self.walls:
            return False

        if next_pos in self.boxes:
            box_next = (next_pos[0] + dx, next_pos[1] + dy)
            if box_next in self.walls or box_next in self.boxes:
                return False
            self.boxes.remove(next_pos)
            self.boxes.add(box_next)

        self.player = next_pos
        return True


def parse_actions(text: str) -> list[str]:
    match = re.search(r"<(?:answer|ans)>\s*(.*?)\s*</(?:answer|ans)>", text, re.I | re.S)
    candidate = match.group(1) if match else text
    found = re.findall(r"\b(up|down|left|right)\b", candidate, re.I)
    return [action.capitalize() for action in found[:10]]


def make_prompt(state: SokobanState) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{USER_HEADER}\nCurrent state:\n{state.render()}"},
    ]


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    max_memory = {0: "3.4GiB", "cpu": "48GiB"} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    return tokenizer, model


def generate_actions(tokenizer, model, state: SokobanState, max_new_tokens: int) -> tuple[str, list[str], float]:
    prompt = tokenizer.apply_chat_template(
        make_prompt(state),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return text, parse_actions(text), elapsed


def run_case(tokenizer, model, case: Case, max_turns: int, max_new_tokens: int) -> dict[str, object]:
    state = SokobanState(case.grid)
    turns = []
    total_generation_time = 0.0
    invalid_actions = 0

    for turn_idx in range(max_turns):
        if state.solved():
            break
        before = state.render()
        text, actions, elapsed = generate_actions(tokenizer, model, state, max_new_tokens)
        total_generation_time += elapsed
        for action in actions:
            if action.lower() not in ACTIONS:
                invalid_actions += 1
                continue
            moved = state.step(action)
            invalid_actions += int(not moved)
            if state.solved():
                break
        turns.append(
            {
                "turn": turn_idx + 1,
                "before": before,
                "raw": text,
                "actions": actions,
                "after": state.render(),
            }
        )

    return {
        "case": case.name,
        "solved": state.solved(),
        "turns": len(turns),
        "invalid_actions": invalid_actions,
        "generation_time_s": total_generation_time,
        "trace": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BlankZ/ragen-checkpoint-step-1200-bf16")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"model={args.model}")

    tokenizer, model = load_model(args.model)
    print(f"device_map={getattr(model, 'hf_device_map', None)}")
    print(f"first_parameter_device={next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(
            "cuda_memory="
            f"{torch.cuda.memory_allocated() / 1024**3:.2f}GiB allocated, "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f}GiB reserved"
        )

    results = [
        run_case(tokenizer, model, case, args.max_turns, args.max_new_tokens)
        for case in CASES
    ]
    solved = sum(int(result["solved"]) for result in results)
    print(f"\nsummary: solved={solved}/{len(results)}")
    for result in results:
        print(
            f"\ncase={result['case']} solved={result['solved']} "
            f"turns={result['turns']} invalid={result['invalid_actions']} "
            f"gen_time={result['generation_time_s']:.2f}s"
        )
        for turn in result["trace"]:
            print(f"turn={turn['turn']} actions={turn['actions']}")
            print("before:")
            print(turn["before"])
            print("raw:")
            print(turn["raw"])
            print("after:")
            print(turn["after"])


if __name__ == "__main__":
    main()
