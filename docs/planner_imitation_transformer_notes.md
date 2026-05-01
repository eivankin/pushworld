# Planner-Imitation Transformer Notes

This branch is a stronger-baseline direction for PushWorld. The goal is not
environment speed this week. The goal is to train a policy from heuristic
planner trajectories and compare its closed-loop success against PPO/DQN.

## Current Prototype

Script:

- `scripts/train_planner_imitation_smoke.py`

Pipeline:

1. select PushWorld puzzles;
2. solve train puzzles with the official C++ `N+RGD` planner;
3. replay each plan into `(state, next_action, remaining_steps)` examples;
4. encode each state as compact symbolic planes;
5. train a small board-transformer policy;
6. evaluate with receding-horizon beam search.

Model:

- input: 7 symbolic planes;
- architecture: per-cell projection + transformer encoder + CLS token;
- heads:
  - next primitive action: `left/right/up/down`;
  - remaining expert-plan steps, used as a beam-search value proxy.

## Smoke Results

Initial greedy one-step imitation was too brittle. With 100 Level 0 train
puzzles, greedy rollout solved only a few puzzles and often collapsed into
repeated actions.

After adding the remaining-steps head and beam search:

| Train set | Examples | Train time | Train eval | Held-out Level 0 | Level 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 Level 0 puzzles | 26 | 5.47s | 5/5 | 0/10 | 0/5 |
| 100 Level 0 puzzles | 941 | 75.3s | 96/100 | 29/50 | 0/5 |

Interpretation:

- The policy can learn planner demonstrations quickly.
- Held-out Level 0 success appears with only 100 training puzzles.
- Level 1 zero-shot is still not solved by this small Level 0-only prototype.
- Beam search matters; plain greedy action classification is not enough.

## Full Base Run Results

Artifact:

- report: `reports/planner_imitation_full_level0_base_small.json`;
- checkpoint: `models/planner_imitation_level0_base_small.pt`;
- TensorBoard: `runs/planner_imitation_level0_base_small`.

Configuration:

- train split: all `data/level0/base/train` puzzles (`2000` puzzles);
- training examples from RGD expert traces: `19902` states;
- model: `d_model=64`, `nhead=4`, `layers=1`, AMP enabled;
- rollout: `8x8` beam, `top_k=3`;
- train-set evaluation skipped.

Results:

| Split | Solved | Total | Success |
| --- | ---: | ---: | ---: |
| Level 0 base test | 163 | 200 | 81.5% |
| Level 1 zero-shot | 3 | 68 | 4.4% |

Solved Level 1 puzzles:

- `Get Out Of My Spot.pwp`;
- `Small Wins.pwp`;
- `Victory Road.pwp`.

Timing:

- RGD trace generation: `23.5s`;
- training: `6680.0s` (`111.3min`, about `1h51m`);
- observed on the laptop run: about `1.7s/batch`, roughly `8min/epoch`.

Important caveat: this training run used the training script's default
`--max-steps 100` for final rollout evaluation. The expert traces for the
larger multi-variant run include plans up to `167` actions, so final reported
numbers should be regenerated with the eval-only script and
`--max-steps 200`.

## Current Optimizations

Already implemented:

- natural numeric puzzle ordering, so small smoke runs use puzzles `0..N`;
- `tqdm` progress bars for expert solving, training, and evaluation;
- optional quick held-out evaluation during training via `--eval-every` and
  `--eval-puzzles`;
- nested batch progress bars inside each epoch;
- TensorBoard batch-loss logging via `--log-every-batches`;
- explicit `--all-train`, `--all-test`, and `--all-level1` flags for full
  split runs;
- repeated `--train-dir` and `--test-dir` arguments for multi-variant Level 0
  training;
- optional `--skip-train-eval` for full runs, because evaluating 2000 train
  puzzles is not needed for the headline held-out result;
- cached state encodings during beam evaluation;
- duplicate beam-state pruning;
- no-op action pruning inside beam expansion;
- batched neural scoring of beam states and leaf states;
- compact JSON summary output via `--output`;
- model checkpoint output via `--model-output`.
- eval-only checkpoint loading via `scripts/eval_planner_imitation.py`, so
  rollout limits and beam settings can be compared without retraining.

The current beam evaluator is still Python-heavy. The cache/pruning changes
should reduce repeated encoding and wasted beam branches, but the exact speedup
needs to be measured against the previous script version or against a
`--beam-width 1 --beam-depth 1` greedy baseline.

## Speedup Log

Fill this table as we run comparable commands:

| Date | Command / setting | Train puzzles | Eval split | Beam | Eval time | Success |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 2026-05-01 | pre-cache prototype | 100 | Level 0 test 50 | 8x8 top-3 | not isolated | 29/50 |
| 2026-05-01 | cached/pruned beam quick check | 20 | Level 0 test 10 | 4x4 top-2 | 1.62s | 0/10 |

For a clean speedup number, run the same checkpoint through
`scripts/eval_planner_imitation.py` with only the beam settings changed.

## Planned Optimizations

Near-term:

- cache expert trajectories and tensors on disk, so repeated training runs do
  not call RGD or re-encode planner states;
- split train/validation and early-stop on held-out Level 0 success;
- save the best checkpoint by periodic held-out evaluation, not only the final
  epoch checkpoint;
- batch larger beam frontier calls more aggressively;
- compare greedy, `4x4`, `8x8`, and `16x8` beam settings.

Model-side:

- add recent-state history, not only the current state;
- try a CNN-only baseline vs transformer encoder;
- add goal/object identity channels if Level 1 transfer remains weak;
- train on all Level 0 variants, not only `base`;
- optionally include modern heuristic-solver Level 1 traces as a second-stage
  curriculum.

Wheeler-style systems optimizations:

- Wheeler reports about `32x` faster teacher-forced training after moving
  Sokoban policy training to GPU;
- Wheeler reports about `60x` faster rollout batches after moving rollout state
  advancement to GPU;
- the analogous PushWorld optimization would move batched rollout/beam state
  advancement out of Python and into a vectorized/GPU representation.

For this project, that is a later optimization. The immediate bottleneck is
getting a strong held-out baseline; after that, optimize beam evaluation time.

## Full Run Command

Train on all Level 0 base train puzzles, skip train-set eval, evaluate all
held-out Level 0 base test puzzles and all official Level 1 puzzles:

```bash
uv run python -u scripts/train_planner_imitation_smoke.py \
  --all-train \
  --all-test \
  --all-level1 \
  --epochs 60 \
  --batch-size 128 \
  --lr 0.001 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --eval-every 10 \
  --eval-puzzles 50 \
  --log-every-batches 10 \
  --skip-train-eval \
  --tensorboard-log runs/planner_imitation_level0_base \
  --output reports/planner_imitation_full_level0_base.json \
  --model-output models/planner_imitation_level0_base.pt
```

Watch it with:

```bash
uv run tensorboard --logdir runs/planner_imitation_level0_base
```

Expected rough runtime on GTX 1650:

- expert trace generation: under a minute for base-only in the observed run;
- training: about `1.5-2h` for `10` epochs with batch size `64`, `d_model=64`,
  `layers=1`, and AMP;
- held-out Level 0 + Level 1 beam eval: depends heavily on `--max-steps` and
  beam settings, so use the eval-only script for final measurements.

## Multi-Variant Level 0 Command

Train on four Level 0 variants (`base`, `all`, `shapes`, `obstacles`) and
evaluate on the corresponding held-out Level 0 splits plus all Level 1 puzzles:

```bash
uv run python -u scripts/train_planner_imitation_smoke.py \
  --train-dir data/level0/base/train \
  --train-dir data/level0/all/train \
  --train-dir data/level0/shapes/train \
  --train-dir data/level0/obstacles/train \
  --test-dir data/level0/base/test \
  --test-dir data/level0/all/test \
  --test-dir data/level0/shapes/test \
  --test-dir data/level0/obstacles/test \
  --all-train \
  --all-test \
  --all-level1 \
  --epochs 6 \
  --batch-size 64 \
  --lr 0.001 \
  --d-model 64 \
  --nhead 4 \
  --layers 1 \
  --amp \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --eval-every 1 \
  --eval-puzzles 100 \
  --log-every-batches 10 \
  --skip-train-eval \
  --tensorboard-log runs/planner_imitation_level0_multi4 \
  --output reports/planner_imitation_level0_multi4.json \
  --model-output models/planner_imitation_level0_multi4.pt
```

This uses `8000` train puzzles and `800` held-out Level 0 puzzles. At the
observed full-base speed, expect roughly `4x` the per-epoch time of base-only.

## Table-2-Style Evaluation Commands

The original PushWorld paper reports train/test success rates by Level 0 task
type. Use the eval-only script to generate the same style of table for any
saved imitation checkpoint.

Set the checkpoint once:

```bash
CKPT=models/planner_imitation_level0_base_small.pt
```

For the currently running multi-variant experiment, replace it with:

```bash
CKPT=models/planner_imitation_level0_multi4.pt
```

Evaluate every Level 0 task type on both train and test splits with a longer
rollout limit:

```bash
for variant in base goals obstacles shapes size walls all; do
  uv run python -u scripts/eval_planner_imitation.py \
    --checkpoint "$CKPT" \
    --eval-dir "data/level0/${variant}/train" \
    --split-name "level0_${variant}_train_200" \
    --all-eval \
    --max-steps 200 \
    --beam-width 8 \
    --beam-depth 8 \
    --top-k 3 \
    --output "reports/table2_like_level0_${variant}_train_200.json" \
    --tensorboard-log "runs/table2_like_level0_${variant}_train_200"

  uv run python -u scripts/eval_planner_imitation.py \
    --checkpoint "$CKPT" \
    --eval-dir "data/level0/${variant}/test" \
    --split-name "level0_${variant}_test_200" \
    --all-eval \
    --max-steps 200 \
    --beam-width 8 \
    --beam-depth 8 \
    --top-k 3 \
    --output "reports/table2_like_level0_${variant}_test_200.json" \
    --tensorboard-log "runs/table2_like_level0_${variant}_test_200"
done
```

Evaluate Level 1 zero-shot with the same rollout limit:

```bash
uv run python -u scripts/eval_planner_imitation.py \
  --checkpoint "$CKPT" \
  --eval-dir external/pushworld/benchmark/puzzles/level1 \
  --split-name level1_200 \
  --all-eval \
  --max-steps 200 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --output reports/table2_like_level1_200.json \
  --tensorboard-log runs/table2_like_level1_200
```

Print a compact Markdown table from the generated JSON files:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

print("| Split | Solved | Total | Success |")
print("| --- | ---: | ---: | ---: |")
for path in sorted(Path("reports").glob("table2_like_*_200.json")):
    data = json.loads(path.read_text())
    total = int(data["total"])
    solved = int(data["solved"])
    skipped = int(data.get("skipped", 0))
    denom = max(1, total - skipped)
    print(f"| {data['split']} | {solved} | {total} | {100 * solved / denom:.1f}% |")
PY
```
