# Optimization Benchmark Plan

Goal: build a measurement-first pipeline for quickly testing speedups and RL
hypotheses on PushWorld.

The current bottleneck picture:

- raw RGB env stepping is reasonably fast on Level 0 base but slows sharply with
  puzzle complexity;
- PPO trains around hundreds of fps on GPU but scales poorly algorithmically;
- DQN learns better on small pixel tasks but runs around `8 fps`;
- RGB observations and image replay buffers are likely wasting memory bandwidth
  and learner time.

## Benchmark Principles

- Measure each optimization in isolation before combining them.
- Use the same puzzle sets and seeds for every speed comparison.
- Record wall-clock time, env steps/sec, learner updates/sec, GPU memory, CPU
  utilization if practical, and success rate on a small eval set.
- Keep the benchmark tasks small enough to run repeatedly:
  - one Level 0 base puzzle;
  - five Level 0 base puzzles;
  - twenty Level 0 base puzzles;
  - Level 0 base train/test subset.

## Baseline Timings To Preserve

Raw environment stepping:

| Puzzle set | Steps/sec |
| --- | ---: |
| Level 0 base train | 1,773 |
| Level 0 all train | 1,126 |
| Level 1 | 340 |

Training:

| Run | Algorithm | Puzzle set | FPS |
| --- | --- | --- | ---: |
| `PPO_5` | PPO | Level 0 base train | ~200 |
| `PPO_8/PPO_9` | PPO | one puzzle | ~195 |
| `DQN_3` | DQN | one puzzle | 7-10 |
| `DQN_4` | DQN | five puzzles | 7-10 |

## Stage 1: Instrument Current Pipeline

Add timing around:

- env reset;
- env transition;
- observation rendering;
- observation conversion to channel-first/uint8;
- replay buffer insertion/sampling for DQN;
- forward pass;
- backward/update step;
- eval callback.

Expected output: a CSV or JSONL file with per-stage timings.

Commands to add:

```bash
uv run pushworld-study profile-pipeline --algorithm dqn --puzzle-path data/debug/base_train_5 --steps 5000
uv run pushworld-study profile-pipeline --algorithm ppo --puzzle-path data/debug/base_train_5 --steps 5000
```

## Stage 2: More Efficient Observations

Replace RGB pixels with compact structured observations.

Candidates:

- integer grid with object ids;
- multi-channel binary planes: walls, agent, movable objects, goals, agent-only
  walls;
- object-table representation: object positions plus shape ids;
- goal-conditioned planes for relabeling later.

Measure:

- env steps/sec;
- DQN fps;
- replay buffer memory;
- model size;
- single-puzzle and five-puzzle success.

Hypothesis: structured planes should improve DQN wall-clock speed more than raw
GPU env stepping because DQN currently pays heavily for image replay/training.

## Stage 3: Vectorized CPU Environment

Before GPU kernels, implement a batched CPU environment API.

Variants:

- simple `SyncVectorEnv` with multiple env instances;
- custom batch wrapper avoiding repeated Python object setup;
- preloaded puzzle/state arrays.

Measure:

- steps/sec vs number of envs;
- PPO fps;
- DQN data-collection fps;
- CPU utilization and memory.

Hypothesis: for Level 0 base, vectorized CPU stepping may already be enough for
PPO. For DQN, learner/replay may remain dominant.

## Stage 4: GPU/Compiled Transition Prototype

Only after compact observations and CPU vectorization are measured:

- implement batched transition for a restricted Level 0 base representation;
- start with Numba or Triton kernels for 1x1 object puzzles;
- validate transition equivalence against official `PushWorldPuzzle` on sampled
  states;
- then extend to shapes/walls/obstacles.

Measure:

- transition-only steps/sec;
- end-to-end learner fps;
- validation mismatch rate;
- engineering complexity.

Hypothesis: GPU env stepping will help most on large batches and complex puzzle
sets, but may not dominate end-to-end training until observation/replay overhead
is reduced.

## Stage 5: Combined Speedups

Once individual speedups are measured, combine:

1. structured observations;
2. vectorized env stepping;
3. optimized replay/model;
4. compiled/GPU transitions.

Report:

- multiplicative speedup vs current RGB baseline;
- end-to-end DQN fps;
- end-to-end PPO fps;
- success rate on one/five/twenty-puzzle debug sets;
- memory footprint.

## Recommended Next Implementation

Start with instrumentation and structured observations, not GPU kernels.

Reasoning:

- DQN is the current algorithmic winner but is extremely slow.
- DQN's slowness is likely dominated by image replay/model updates, so GPU
  environment kernels alone may not solve the main wall-clock bottleneck.
- Structured observations are also required for goal relabeling, making them a
  useful bridge toward the later algorithmic milestone.

