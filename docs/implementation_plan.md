# Implementation Plan

## Milestone 0: Reproducible Workspace

- Use uv with a project-local cache (`uv.toml`) and a normal `.venv`.
- Keep the official benchmark as `external/pushworld` submodule.
- Do not edit the upstream submodule unless we intentionally maintain a patch.
- Add smoke tests around the official Gym environment.

## Milestone 1: Baseline Environment and Profiling

- Status: initial implementation exists.
- Generate or unpack a small Level 0 training set from the official scripts.
- Build a wrapper that exposes observations in two forms:
  - RGB images, matching the official Gym wrapper;
  - compact plane/tensor observations for faster learning experiments.
- Measure:
  - single-env steps/sec;
  - vectorized env steps/sec;
  - rendering time;
  - reset/puzzle sampling time;
  - training loop samples/sec.

Implemented commands:

```bash
uv run pushworld-study profile-env --episodes 10 --max-steps 100
uv sync --group rl
uv run pushworld-study train-baseline ppo --total-timesteps 128
uv run pushworld-study train-baseline dqn --total-timesteps 16
uv run pushworld-study eval-baseline ppo models/ppo_smoke_seed0_100000.zip --puzzle-path data/level0/base/test --max-episodes 200
```

Current baseline limitations:

- PPO/DQN are wired through Stable-Baselines3, not Acme/JAX.
- The CNN matches the paper's kernels, strides, ReLU activations, and FC sizes,
  but uses assumed convolution channel counts `32, 64, 64`.
- DQN replay size is deliberately bounded for local smoke runs; the full
  experiment config still needs a memory budget and possibly compact state
  observations.
- Level 0 generation/splits are not automated yet.
- Held-out evaluation now exists, but training still needs better checkpoint
  naming and a compact inference-only save format.

## Milestone 2: PPO and DQN Smoke Baselines

- Start with Stable-Baselines3 for fast iteration, because it has both PPO and
  DQN and integrates easily with TensorBoard.
- Match the paper architecture where practical:
  - convolution kernels `3, 3, 5`;
  - strides `3, 1, 1`;
  - fully connected layers `256, 128`;
  - 100-step episodes;
  - reward settings from the official wrapper.
- Treat Acme/JAX as the exact-reproduction option if SB3 results diverge for
  reasons that matter scientifically.

## Milestone 3: Current RL Pipeline Optimization

- Keep plane observations as the default training representation.
- Add goal-conditioned plane observations:
  - current-state planes;
  - goal/subgoal planes;
  - optional achieved-goal indicators.
- Define achieved goals from object positions reached during failed episodes.
- Log both original-task and relabeled-task returns.
- Compare vanilla vs relabeled pipelines on success/time and update seconds per
  `100k` env steps.
- Add finer learner/update profiling:
  - dataloader/replay sampling;
  - policy forward/backward;
  - optimizer step;
  - host-device transfer if visible.
- Test AMP, rollout length, minibatch size, update epochs, and replay/buffer
  layout as separate optimization variables.

## Milestone 4: Transformer Policy Pipeline

- Export planner trajectories into an offline dataset:
  `(goal, state_t, action_t, state_{t+1}, remaining_distance)`.
- Cache plane tensors so supervised training does not wait on parsing or
  environment stepping.
- Implement model ladder:
  - CNN-only no-history baseline;
  - CNN encoder + transformer over the last `k` states;
  - action head plus bucketed distance/solvability head.
- Evaluate action accuracy, greedy solve rate, beam-search solve rate, and
  states/sec during search.
- Optimize dataloader throughput, AMP / `torch.compile`, larger batches,
  batched beam-candidate scoring, and repeated state/logit caching.

## Milestone 5: Hybrid Planner + Learned Executor

- Build planner-derived subgoal traces from existing solution trajectories.
- Train a goal-conditioned local executor on Level 0 subgoals with imitation
  and/or relabeling.
- Add a minimal runtime interface:
  current state, planner-produced subgoal, executor action, progress monitor,
  and replan trigger.
- Profile planner call time, executor inference time, env stepping, and
  replanning overhead separately.
- Compare planner-only, executor-only, hybrid without learned ranker, and hybrid
  with learned ranker/value model.

## Milestone 6: RAGEN LLM-Agent Pipeline

- Adapt PushWorld as a RAGEN-compatible text environment:
  - Gym wrapper;
  - deterministic ASCII/text state renderer;
  - strict action parser;
  - reward/success adapter.
- Run a fixed-prompt no-finetune baseline for at least `256` eval episodes and
  store full trajectory logs.
- If the baseline is non-empty, run a small LoRA/StarPO experiment on a Level 0
  split.
- Profile wall time, tokens/sec, tokens/episode, env-step time, generation time,
  update time, and GPU memory.
- Optimize prompt compression, reasoning-token caps, one-token action answers,
  vLLM rollout batching, Ray worker and env placement, no-finetune vs LoRA vs
  full update, and high-signal rollout filtering.

## Deferred: GPU Environment Prototype

GPU environment work is no longer the immediate next milestone for current
PPO/DQN baselines because plane observations and vectorized PPO have shifted the
short-run bottleneck to learner/update time. It remains relevant if later
search/rollout pipelines become synchronization-bound.

- Avoid RGB rendering in a GPU prototype.
- Represent each puzzle as fixed-shape integer tensors.
- Start with PyTorch tensor transitions before Numba/Triton kernels.
- Validate against the official Python environment before using it for learning.
- Require end-to-end speedup, not just transition-only speedup, before replacing
  the official environment.

## Experiment Metrics

- Throughput: env steps/sec, learner updates/sec, samples/sec, wall-clock to fixed
  number of samples.
- Learning: success rate, mean episode return, solved-puzzle fraction, time to
  first nonzero success, area under success curve.
- Reproducibility: random seeds, puzzle set hash/path, git commit, submodule
  commit, hardware, CUDA/driver versions when applicable.
