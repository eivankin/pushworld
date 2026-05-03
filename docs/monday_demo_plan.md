# Monday Demo Plan: Planner-Imitation Transformer

## Headline

The strongest current result is the `multi4` planner-imitation transformer.
It was trained from RGD traces on four Level-0 variants: `base`, `all`,
`shapes`, and `obstacles`. A small rollout-side repeat penalty improves both
Level-0 and Level-1 success without retraining.

## Key Metrics

Evaluation settings used in the repeat-penalty reports:

- checkpoint: `models/planner_imitation_level0_multi4.pt`;
- rollout: `max_steps=100`, `beam_width=8`, `beam_depth=8`, `top_k=3`;
- repeat penalty is inference-only, not a training change.

| Checkpoint / setting | Level-0 Base Test, penalty 0 | Level-0 Base Test, penalty 1 | Delta |
| --- | ---: | ---: | ---: |
| `multi4` | 164/200 = 82.0% | 181/200 = 90.5% | +17 solved |

| Checkpoint / setting | Level-1, penalty 0 | Level-1, penalty 1 | Delta |
| --- | ---: | ---: | ---: |
| `multi4` | 8/68 = 11.8% | 10/68 = 14.7% | +2 solved |

Repeat penalty also changes individual outcomes:

| Split | Gained with penalty 1 | Lost with penalty 1 | Net |
| --- | ---: | ---: | ---: |
| Level-0 base test | 18 | 1 | +17 |
| Level-1 | 2 | 0 | +2 |

Plan-length comparison against RGD, counted only on puzzles the model solves:

| Split / setting | Model solved | Model shorter | Same length | Model longer | Avg. model - RGD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Level-0 base, penalty 0 | 164 | 13 | 146 | 5 | -0.18 steps |
| Level-0 base, penalty 1 | 181 | 16 | 152 | 13 | -0.01 steps |
| Level-1, penalty 1 | 10 | 5 | 5 | 0 | -4.50 steps |

Interpretation: globally, the model does not clearly outperform RGD on Level-0
plan length; most solved Level-0 plans are exactly the same length as RGD. On
the solved Level-1 subset, however, the policy rollout is never longer than RGD
and is shorter on half of the solved puzzles.

The average above is computed over all puzzles solved by the model in that
split, not only over the improved cases. On Level-1, the shorter-than-RGD cases
improve by `1..28` steps, with a mean improvement of `9.0` steps among improved
cases.

Level-1 comparison to human-labeled reference trajectories:

| Level-1 solved subset, penalty 1 | Model shorter | Same length | Model longer | Avg. model - human | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model vs human | 1 | 7 | 2 | +2.60 steps | -1..+20 |
| RGD vs human | 0 | 6 | 4 | +7.10 steps | 0..+48 |

Interpretation: the model is still usually not better than the human labels,
but it comes closer to human plan length than RGD on this solved Level-1 subset.
The biggest gap is `Seeing Red.pwp`: human `28`, model `48`, RGD `76`.

The newer augmentation-heavy checkpoint underperformed on original Level-1
maps. The latest available split-aware eval artifact for that direction shows
`6/68` at epoch 1 (`5/34` train-source originals, `1/34` held-out-source
originals). The later run was reported as about `5/68`, so it is not the demo
checkpoint.

## Demo Puzzles

Primary Level-1 demo:

| Puzzle | Split | Penalty 0 | Penalty 1 | Why show it |
| --- | --- | --- | --- | --- |
| `Seeing Red.pwp` | held-out Level-1 source split | fails | solves in 48 steps | clean example of a config-only improvement |

Backup Level-1 demo:

| Puzzle | Split | Penalty 0 | Penalty 1 | Why show it |
| --- | --- | --- | --- | --- |
| `Three Goals.pwp` | Level-1 train-source split | fails | solves in 39 steps | harder-looking multi-goal puzzle, useful if live demo needs a second case |

These two are also strong plan-length examples:

| Puzzle | Model steps, penalty 1 | RGD plan length | Difference |
| --- | ---: | ---: | ---: |
| `Seeing Red.pwp` | 48 | 76 | -28 |
| `Three Goals.pwp` | 39 | 49 | -10 |

Human-reference comparison for the same examples:

| Puzzle | Human plan | Model steps | RGD plan | Comment |
| --- | ---: | ---: | ---: | --- |
| `Seeing Red.pwp` | 28 | 48 | 76 | model is much closer to human than RGD, but still longer |
| `Three Goals.pwp` | 32 | 39 | 49 | model is closer to human than RGD |
| `Two Goals.pwp` | 18 | 17 | 18 | model is one step shorter than the human/RGD reference |

Reliable success examples:

| Puzzle | Penalty 0 | Penalty 1 | Why show it |
| --- | --- | --- | --- |
| `Get Out Of My Spot.pwp` | solves | solves | quick sanity check that the model is loaded and working |
| `Small Wins.pwp` | solves | solves | simple Level-1 success example |

Level-0 repeat-penalty examples:

| Puzzle | Penalty 0 behavior | Penalty 1 behavior |
| --- | --- | --- |
| `level_0_base_test_101.pwp` | fails at 100 steps, 93 repeated states | solves in 23 steps, 0 repeated states |
| `level_0_base_test_115.pwp` | fails at 100 steps, 95 repeated states | solves in 25 steps, 0 repeated states |
| `level_0_base_test_116.pwp` | fails at 100 steps, 98 repeated states | solves in 15 steps, 0 repeated states |

Avoid as a penalty demo:

- `level_0_base_test_99.pwp`: solves without penalty in 14 steps, but fails
  with penalty 1. This is a useful caveat if asked, but not a live showcase.

## Demo Flow

1. Open Streamlit demo with checkpoint `models/planner_imitation_level0_multi4.pt`.
2. Select `Level 1` -> `Seeing Red.pwp`.
3. Run with `Repeat penalty = 0.0`: show failure / looping behavior.
4. Change only `Repeat penalty = 1.0`.
5. Run again: show solved rollout and action table.
6. Optional: switch to `level_0_base_test_101.pwp` and repeat the same
   penalty-0 vs penalty-1 comparison to show the loop-breaking mechanism more
   explicitly.

## Commands Behind The Numbers

Level-0 base test, no repeat penalty:

```bash
uv run python -u scripts/eval_planner_imitation.py \
  --checkpoint models/planner_imitation_level0_multi4.pt \
  --eval-dir data/level0/base/test \
  --split-name level0_base_test_200 \
  --all-eval \
  --max-steps 100 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --repeat-penalty 0 \
  --output reports/table2_like_level0_base_test_200_penalty0_steps100.json
```

Level-0 base test, repeat penalty:

```bash
uv run python -u scripts/eval_planner_imitation.py \
  --checkpoint models/planner_imitation_level0_multi4.pt \
  --eval-dir data/level0/base/test \
  --split-name level0_base_test_200 \
  --all-eval \
  --max-steps 100 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --repeat-penalty 1 \
  --output reports/table2_like_level0_base_test_200_p1_s100.json
```

Level-1, repeat penalty:

```bash
uv run python -u scripts/eval_planner_imitation.py \
  --checkpoint models/planner_imitation_level0_multi4.pt \
  --eval-dir external/pushworld/benchmark/puzzles/level1 \
  --split-name level1 \
  --all-eval \
  --max-steps 100 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --repeat-penalty 1 \
  --output reports/table2_like_level1_penalty1_steps100.json
```

## Caveats To Say Out Loud

- Repeat penalty is not a learned improvement; it is an inference-time rollout
  optimization.
- It helps net success, but it is not universally monotonic: one Level-0 base
  test puzzle regressed at penalty 1.
- The augmentation-heavy checkpoint did not improve Level-1 transfer yet, so
  Monday's demo should focus on the `multi4` checkpoint plus repeat-aware
  rollout.
