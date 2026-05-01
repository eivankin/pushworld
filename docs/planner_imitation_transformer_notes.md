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

For a clean speedup number, run the same trained-policy configuration twice
with only the beam settings changed, or add an eval-only mode after checkpoint
loading.

## Planned Optimizations

Near-term:

- cache expert trajectories and tensors on disk, so repeated training runs do
  not call RGD or re-encode planner states;
- split train/validation and early-stop on held-out Level 0 success;
- batch larger beam frontier calls more aggressively;
- add an eval-only checkpoint loader to measure beam settings without retraining;
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

- expert trace generation: under a few minutes;
- training: roughly 20-40 minutes for this small transformer;
- held-out Level 0 + Level 1 beam eval: likely 10-30 minutes with current Python
  beam code;
- total: roughly 30-75 minutes.

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
