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

Observed DQN timing split after adding the first timing hooks on
`data/debug/base_train_5` with plane observations:

- `train_steps_per_second`: `173.3`
- `dqn_rollout_fraction`: `23.8%`
- `dqn_update_fraction`: `76.0%`
- separately measured environment work: `0.38%` of train time

Interpretation:

- after removing RGB rendering and replay-size overhead, DQN is also primarily
  learner/update bound rather than env bound;
- further environment-side optimization is unlikely to move wall-clock training
  much for the current DQN baseline;
- future DQN optimization should focus on replay/update efficiency, mixed
  precision, compiler availability, or smaller/cheaper model paths.

Observed `torch.compile` microbenchmark on CUDA for the PushWorld CNN policy
head with plane observations and batch `256`:

- eager inference: `1055.6` steps/s
- compiled inference: `1643.6` steps/s
- inference speedup: `1.56x`
- eager train step: `345.7` steps/s
- compiled train step: `359.1` steps/s
- train-step speedup: `1.04x`

Interpretation:

- compiler optimization helps the isolated forward path materially;
- it barely changes end-to-end training-step throughput;
- this matches the broader profiler story that current learner overhead is not
  just the raw model forward pass.

Follow-up at batch `512` showed the same qualitative result:

- eager inference: `614.5` steps/s
- compiled inference: `878.7` steps/s
- inference speedup: `1.43x`
- eager train step: `201.6` steps/s
- compiled train step: `208.3` steps/s
- train-step speedup: `1.03x`

This makes the result more robust: compilation is a real forward-pass win, but
it is still not the main lever for end-to-end learner throughput on this setup.

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
- Timed PPO profiling shows why the gain tapers: at 4 envs rollout collection
  and PPO update are roughly balanced (`50.4%` rollout, `49.6%` update), while
  at 16 envs rollout drops to `22.5%` and update rises to `77.4%`. Further env
  parallelism cannot help much unless update time is reduced.
- These are throughput probes, not learning-quality runs. Increasing `n_envs`
  changes PPO's rollout batch shape (`n_envs * n_steps`), so learning
  comparisons should either keep total rollout size fixed or explicitly treat
  rollout size as an experimental variable.

Stop condition: if 8-16 CPU envs saturate PPO training and give acceptable fps,
GPU environment work should wait until Level 1 or larger batched experiments
need it.

## Stage 4: Learner/Replay Optimizations

The DQN/PPO timing splits show that learner and replay overhead can dominate
once RGB is removed. These changes should be benchmarked before environment
kernels:

- add fine-grained update timing for replay sampling, tensor transfer,
  forward/backward, and optimizer step;
- store plane observations as `uint8`/bool in replay and cast on batch load;
- reduce replay copies by preallocating contiguous arrays;
- test smaller CNNs for 6-plane inputs;
- test AMP on policy updates;
- sweep rollout length, minibatch size, and update epochs for PPO;
- compare SB3 DQN against a minimal project-local DQN loop if SB3 overhead
  remains high.

Measure:

- replay memory per 100k transitions;
- sampled batches/sec;
- learner updates/sec;
- end-to-end fps;
- single/five-puzzle success at fixed wall-clock budgets.

## Stage 5: Goal-Conditioned Relabeling Pipeline

Add goal-conditioned observations to the existing plane pipeline before moving
to more expensive planner/LLM systems.

Implementation targets:

- current-state planes plus goal/subgoal planes;
- achieved-goal extraction from failed trajectories;
- relabeled replay/rollout records;
- evaluation under original goals and relabeled training goals.

Measure:

- success/time;
- update seconds per 100k env steps;
- original-task success rate;
- relabeled-task success rate;
- effect on deterministic vs stochastic policy collapse.

## Stage 6: Transformer Policy Pipeline

Pipeline:

- planner solutions;
- state-action dataset;
- plane encoder + transformer over recent states;
- action head and distance/solvability head;
- greedy rollout and beam search.

Optimization steps:

1. export planner traces and cache plane tensors;
2. train CNN-only no-history baseline;
3. add transformer over the last `k` states;
4. add bucketed distance/solvability head;
5. optimize dataloader, AMP, `torch.compile`, and batch size;
6. benchmark greedy rollout;
7. benchmark beam widths `8`, `16`, `32`;
8. batch all beam-candidate scoring in one forward pass;
9. cache repeated state/logit evaluations.

Metrics:

- action accuracy;
- greedy solve rate;
- beam solve rate;
- states/sec during search;
- GPU utilization;
- wall-clock per solved puzzle.

## Stage 7: Hybrid Solver Pipeline

Pipeline:

- current state;
- planner-produced subgoal or short solution prefix;
- goal-conditioned executor;
- progress monitor;
- accept/replan decision.

Optimization steps:

1. slice solution traces into reachable partial goals every `k` actions;
2. train executor with imitation and/or relabeling on Level 0 subgoals;
3. profile planner call time, executor inference, env stepping, and replanning;
4. cache planner subgoals;
5. replan only on timeout or distance-to-subgoal regression;
6. add learned ranker/value for planner candidates.

Metrics:

- subgoal success;
- replans per puzzle;
- planner time vs executor time;
- solved puzzles per minute;
- ablations: planner-only, executor-only, hybrid without ranker, hybrid with
  ranker.

## Stage 8: RAGEN LLM-Agent Pipeline

Pipeline:

- PushWorld Gym/text wrapper;
- deterministic text state + prompt;
- LLM reasoning/action generation;
- trajectory reward;
- StarPO/LoRA update.

Baseline and optimization steps:

1. fixed prompt, no fine-tune, at least `256` eval episodes;
2. log full trajectories and parse failures;
3. profile wall time, tokens/sec, tokens/episode, env-step time, generation
   time, update time, and GPU memory;
4. compare verbose text state vs compact ASCII planes;
5. cap reasoning tokens and require one-token action answers;
6. batch `P` initial states by `N` samples per state through vLLM;
7. tune Ray workers and env placement;
8. compare no-finetune, LoRA, and full update;
9. add high-signal rollout filtering.

The RAGEN paper provides runtime/resource context, but not a complete
throughput benchmark table. Treat the paper's Sokoban PPO total-time plots,
H100/A100 + vLLM/Ray/FSDP setup, and LoRA resource comparison as reference
points rather than final targets.

Metrics:

- prompt-only success;
- time-to-target success;
- tokens per solved puzzle;
- GPU memory;
- rollout generation fraction vs update fraction;
- LoRA/full-update quality-time trade-off.

## Deferred Stage: GPU/Compiled Transition Prototype

GPU env stepping is deferred until a later pipeline makes transition/search
state propagation the dominant bottleneck.

- define a dense tensor state representation;
- implement batched transition for a restricted Level 0 base representation;
- start with PyTorch tensor ops on GPU before writing custom kernels;
- validate transition equivalence against official `PushWorldPuzzle`;
- replace the official env only after equivalence tests pass and end-to-end
  speed improves materially over vectorized CPU planes.

## Combined Speedups

Once individual speedups are measured, combine:

1. structured observations;
2. vectorized env stepping;
3. optimized replay/model;
4. goal-conditioned relabeling;
5. planning/LLM pipeline-specific inference optimizations.

Report:

- multiplicative speedup vs current RGB baseline;
- end-to-end DQN fps;
- end-to-end PPO fps;
- success rate on one/five/twenty-puzzle debug sets;
- memory footprint.

## Recommended Next Implementation

Start with instrumentation, structured observations, and goal-conditioned
relabeling, not GPU kernels.

Reasoning:

- DQN is the current algorithmic winner but is extremely slow.
- DQN's slowness is likely dominated by image replay/model updates, so GPU
  environment kernels alone may not solve the main wall-clock bottleneck.
- Structured observations are also required for goal relabeling, making them a
  useful bridge toward the later algorithmic milestone.
