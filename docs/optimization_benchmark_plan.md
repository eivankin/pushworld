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
| `PPO_10` | PPO | five puzzles, RGB | 131 final / 137.9 mean |
| `PPO_11` | PPO | five puzzles, RGB | 130 final / 134.5 mean |
| `ppo_planes_fair/PPO_1` | PPO | five puzzles, planes | 454 |
| `DQN_3` | DQN | one puzzle, RGB | 7-10 |
| `DQN_4` | DQN | five puzzles, RGB | 8.0 mean |
| `dqn_planes_fair/DQN_1` | DQN | five puzzles, planes | 161.7 mean |

## Stage 1: Instrument Current Pipeline

Use the lightweight `profile-pipeline` command before larger rewrites. The first
version works with the existing SB3 training loop and does not require custom
algorithms.

Measurements collected:

- environment reset time;
- environment transition/reward time, excluding observation extraction;
- observation extraction/rendering time;
- observation wrapper conversion time;
- model prediction time;
- short end-to-end `learn()` wall time.
- PPO rollout collection time and PPO gradient-update time.

Measurements still to add later:

- DQN replay insertion/sampling time;
- eval callback wall time;
- GPU memory, if readable outside the sandbox.

Output format:

- write one JSONL row per measurement window;
- include `algorithm`, `observation_mode`, `puzzle_path`, `seed`,
  `env_steps`, `train_model_timesteps`, `train_steps_per_second`, and `device`;
- keep raw timing columns separate so we can compute derived percentages later.

Commands to run:

```bash
uv run pushworld-study profile-pipeline ppo \
  --puzzle-path data/debug/base_train_5 \
  --observation-mode rgb \
  --steps 5000 \
  --output reports/profile_ppo_rgb_5.jsonl

uv run pushworld-study profile-pipeline ppo \
  --puzzle-path data/debug/base_train_5 \
  --observation-mode planes \
  --steps 5000 \
  --output reports/profile_ppo_planes_5.jsonl

uv run pushworld-study profile-pipeline dqn \
  --puzzle-path data/debug/base_train_5 \
  --observation-mode planes \
  --steps 5000 \
  --output reports/profile_dqn_planes_5.jsonl
```

Key fields:

- `env_transition_seconds`: PushWorld state transition and reward logic.
- `env_observe_seconds`: RGB rendering or plane extraction.
- `conversion_per_observation_seconds`: channel-first/uint8 conversion for RGB
  or float32 array handling for planes.
- `predict_per_call_seconds`: policy inference cost for one observation.
- `train_steps_per_second`: short end-to-end SB3 training throughput.
- `ppo_rollout_seconds` and `ppo_update_seconds`: PPO collection/update split
  during the short training phase.
- `*_profile_phase_seconds`: total wall time for each profiler phase, including
  env construction and model setup where applicable.

Acceptance criterion: the profiler should explain enough of observed wall time
to identify whether the next bottleneck is environment/observation work, model
inference, SB3 learner/replay overhead, or evaluation.

Initial Level 1/2 plane profiles:

| Profile | Observation shape | Env steps/sec | Train FPS | Env time / train time | Unaccounted setup in old profiler |
| --- | --- | ---: | ---: | ---: | ---: |
| Level 0 debug 5 | `6x7x7` | 45,590 | 586.4 | 1.29% | 0.91s |
| Level 1 | `6x51x42` | 24,589 | 478.8 | 1.95% | 14.11s |
| Level 2 | `6x39x32` | 13,776 | 487.7 | 3.54% | 94.25s |

Interpretation:

- Per-step plane environment work grows with puzzle complexity but is still not
  the dominant short-training cost through Level 2.
- Observation extraction remains the larger part of measured env time, but env
  time is still only a few percent of training wall time.
- The old profiler showed large setup/loading overhead on Level 1/2, especially
  Level 2. Phase-level timing was added so future profiles can separate setup
  cost from rollout/training cost.
- This does not yet justify GPU env kernels. The next optimization target is
  still learner/collection overhead and vectorized PPO, with setup caching worth
  revisiting if repeated experiment startup becomes painful.

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
- PPO fps;
- DQN fps;
- replay buffer memory;
- model size;
- single-puzzle and five-puzzle success;
- deterministic and stochastic eval success.

Hypothesis: structured planes should improve DQN wall-clock speed more than raw
GPU env stepping because DQN currently pays heavily for image replay/training.

Initial implementation:

- `--observation-mode planes` emits 6 channel-first float32 planes:
  - walls;
  - agent-only walls;
  - agent;
  - goal-associated movable objects;
  - other movable objects;
  - goal cells.
- The transition dynamics still use the official `PushWorldPuzzle` logic.
- RGB rendering is skipped for observations.

Initial smoke timing on `data/debug/base_train_5`:

| Observation mode | Episodes | Steps | Steps/sec |
| --- | ---: | ---: | ---: |
| RGB | 20 | 1,517 | 1,668 |
| Planes | 20 | 1,517 | 34,118 |

This is about a 20x isolated environment-observation speedup on the five-puzzle
debug set.

Initial training smoke checks:

- PPO with planes on CPU ran around 500-900 fps over a 1,024-step smoke run.
- DQN with planes on CPU ran around 96 fps over a 512-step smoke run.
- A full 100k five-puzzle PPO run with plane observations reached final
  TensorBoard FPS `454`. Compared with five-puzzle RGB PPO runs, the measured
  end-to-end TensorBoard `time/fps` speedup was about `3.3-3.5x`, depending on
  whether mean, post-warmup mean, or final FPS is used.
- A full 50k five-puzzle DQN run with plane observations reached mean
  TensorBoard FPS `161.7`, compared with `8.0` mean FPS for the RGB `DQN_4`
  run. This is about a `20.2x` end-to-end speedup while preserving repeated
  eval success: `80/200` deterministic and `77/200` stochastic.

These numbers are not final benchmark results, but they are strong enough to
justify rerunning the five-puzzle PPO/DQN comparisons with plane observations.

Fair PPO comparison command:

```bash
uv run pushworld-study train-baseline ppo \
  --puzzle-path data/debug/base_train_5 \
  --eval-puzzle-path data/debug/base_train_5 \
  --eval-freq 5000 \
  --n-eval-episodes 25 \
  --eval-stochastic \
  --total-timesteps 100000 \
  --learning-rate 0.0001 \
  --ent-coef 0.001 \
  --n-epochs 4 \
  --n-steps 256 \
  --batch-size 64 \
  --seed 0 \
  --device cuda \
  --observation-mode planes \
  --log-dir runs/ppo_planes_fair \
  --model-dir models/ppo_planes_fair
```

Compare against the earlier five-puzzle RGB PPO run on:

- training fps;
- rollout reward curve;
- eval mean reward curve;
- best stochastic success over 200 repeated evaluations;
- deterministic success on the same best checkpoint.

If planes are faster but not better, keep them anyway: they are still the
cleaner substrate for relabeling and object/state debugging.

Observed result for `ppo_planes_fair/PPO_1`:

- best eval checkpoint: step 85k, mean reward `3.293`;
- final eval at 100k: mean reward `2.439`, mean length `76.44`;
- repeated deterministic eval on best checkpoint: `0/200`;
- repeated stochastic eval on best checkpoint: `42/200`;
- conclusion: planes are worth keeping for speed and representation clarity, but
  PPO still needs algorithmic help before scaling beyond a few puzzles.

Observed result for `dqn_planes_fair/DQN_1`:

- final eval checkpoint: step 50k, mean reward `4.270`, mean length `53.44`;
- repeated deterministic eval on best/final checkpoint: `80/200`;
- repeated stochastic eval on best/final checkpoint: `77/200`;
- mean training FPS: `161.7`, about `20.2x` faster than RGB `DQN_4`;
- conclusion: plane observations should be the default for future DQN
  experiments.

## Stage 3: Vectorized CPU Environment

Before GPU kernels, implement a batched CPU environment API.

Variants:

- `DummyVecEnv`/single-process SB3 vectorization as a correctness baseline;
- `SubprocVecEnv` for parallel CPU collection;
- custom batch wrapper avoiding repeated Python object setup;
- preloaded puzzle/state arrays for resets.

Implemented controls:

- `train-baseline --n-envs N --vec-env dummy|subproc`
- `profile-pipeline --n-envs N --vec-env dummy|subproc`

Measure:

- steps/sec vs number of envs;
- PPO fps;
- DQN data-collection fps;
- CPU utilization and memory.

Hypothesis: for Level 0 base, vectorized CPU stepping may already be enough for
PPO. For DQN, learner/replay may remain dominant.

Benchmark matrix:

| Env count | Observation | Algorithm | Puzzle set |
| ---: | --- | --- | --- |
| 1 | planes | PPO | 5 puzzles |
| 4 | planes | PPO | 5 puzzles |
| 8 | planes | PPO | 5 puzzles |
| 16 | planes | PPO | 5 puzzles |
| 1 | planes | DQN | 5 puzzles |
| 4 | planes | DQN | 5 puzzles |

Implementation notes:

- `make_training_env()` now has factory/vectorized wrappers for SB3 envs;
- scale PPO `n_steps` carefully: total rollout size is `n_envs * n_steps`;
- keep the total rollout size comparable between runs when measuring learning
  quality, and vary it separately when measuring pure throughput;
- use separate log/model dirs for every env-count run.

Suggested PPO throughput probes:

```bash
uv run pushworld-study profile-pipeline ppo \
  --puzzle-path data/debug/base_train_5 \
  --observation-mode planes \
  --steps 8192 \
  --device cuda \
  --n-envs 1 \
  --vec-env dummy \
  --output reports/profile_ppo_planes_vec.jsonl

uv run pushworld-study profile-pipeline ppo \
  --puzzle-path data/debug/base_train_5 \
  --observation-mode planes \
  --steps 8192 \
  --device cuda \
  --n-envs 4 \
  --vec-env dummy \
  --output reports/profile_ppo_planes_vec.jsonl

uv run pushworld-study profile-pipeline ppo \
  --puzzle-path data/debug/base_train_5 \
  --observation-mode planes \
  --steps 8192 \
  --device cuda \
  --n-envs 4 \
  --vec-env subproc \
  --output reports/profile_ppo_planes_vec.jsonl
```

Initial vectorized PPO results on `data/debug/base_train_5` with plane
observations:

| Observation | Env count | Vec env | Train FPS | Speedup vs plane 1 env | Speedup vs RGB 1 env | Env time / train time |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| RGB | 1 | `dummy` | 195.1 | 0.33x | 1.00x | 11.15% |
| planes | 1 | `dummy` | 586.7 | 1.00x | 3.01x | 1.28% |
| planes | 4 | `dummy` | 1,327.7 | 2.26x | 6.80x | 2.88% |
| planes | 4 | `subproc` | 1,128.9 | 1.92x | 5.79x | 2.47% |
| planes | 8 | `dummy` | 1,609.5 | 2.74x | 8.25x | 3.58% |
| planes | 16 | `dummy` | 1,786.5 | 3.05x | 9.15x | 3.86% |

Interpretation:

- Same-process vectorization is already a large PPO throughput win.
- `SubprocVecEnv` is slower than `DummyVecEnv` at four envs on Level 0 plane
  observations, likely because each environment step is too cheap to amortize
  IPC overhead.
- `DummyVecEnv` keeps improving through 16 envs, but returns diminish after 4
  envs. The main jump is from single-env collection to batched collection.
- Combined plane observations plus `DummyVecEnv` reaches about `6.8x` speedup at
  4 envs and `9.2x` at 16 envs compared with the vanilla RGB single-env profile.
- These are throughput probes, not learning-quality runs. Increasing `n_envs`
  changes PPO's rollout batch shape (`n_envs * n_steps`), so learning
  comparisons should either keep total rollout size fixed or explicitly treat
  rollout size as an experimental variable.

Stop condition: if 8-16 CPU envs saturate PPO training and give acceptable fps,
GPU environment work should wait until Level 1 or larger batched experiments
need it.

## Stage 4: GPU/Compiled Transition Prototype

Only after compact observations and CPU vectorization are measured:

- define a dense tensor state representation:
  - static planes: walls, agent-only walls, goal cells;
  - dynamic planes: agent, movable object occupancy, object-goal ids if needed;
  - per-env metadata: step count, solved flag, active puzzle id;
- implement batched transition for a restricted Level 0 base representation;
- start with PyTorch tensor ops on GPU before writing custom kernels;
- if PyTorch ops are too slow, try Numba CUDA or Triton kernels;
- validate transition equivalence against official `PushWorldPuzzle` on sampled
  states;
- then extend to multi-cell shapes, walls, obstacles, and higher levels.

Measure:

- transition-only steps/sec;
- end-to-end learner fps;
- validation mismatch rate;
- engineering complexity.

Validation suite:

- generate random legal states from official env resets plus random rollouts;
- replay the same actions through official env and GPU env;
- compare reward, terminated/truncated flags, agent position, object positions,
  and rendered plane observations;
- run at least 10k random transitions per supported puzzle family before using
  the GPU env for training.

Prototype ladder:

1. CPU tensor reference implementation using the same dense representation.
2. GPU PyTorch implementation with batch size 1,024+.
3. Custom kernel only if PyTorch GPU ops leave transition time dominant.
4. SB3-compatible vector env wrapper.
5. Native learner integration if SB3 wrapper overhead becomes dominant.

Hypothesis: GPU env stepping will help most on large batches and complex puzzle
sets, but may not dominate end-to-end training until observation/replay overhead
is reduced.

Risks:

- exact PushWorld collision/object-shape semantics may be harder to reproduce
  than the Level 0 base case suggests;
- SB3 vector-env APIs may erase much of the GPU transition speedup through CPU
  synchronization;
- GPU envs are more valuable for PPO-style large-batch collection than for DQN
  if replay/model updates remain the bottleneck.

Decision gate: do not replace the official env until equivalence tests pass and
end-to-end PPO fps improves by at least 2x over vectorized CPU planes.

## Stage 4.5: Learner/Replay Optimizations

The DQN result suggests learner and replay overhead can dominate once RGB is
removed. These changes should be benchmarked separately from environment
kernels:

- store plane observations as `uint8`/bool in replay and cast on batch load;
- reduce replay copies by preallocating contiguous arrays;
- test smaller CNNs for 6-plane inputs;
- test MLP over object-table features for Level 0 base;
- use larger batches only after checking GPU utilization;
- compare SB3 DQN against a minimal project-local DQN loop if SB3 overhead
  remains high.

Metrics:

- replay memory per 100k transitions;
- sampled batches/sec;
- learner updates/sec;
- end-to-end fps;
- single/five-puzzle success at fixed wall-clock budgets.

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
