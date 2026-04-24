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
```

Current baseline limitations:

- PPO/DQN are wired through Stable-Baselines3, not Acme/JAX.
- The CNN matches the paper's kernels, strides, ReLU activations, and FC sizes,
  but uses assumed convolution channel counts `32, 64, 64`.
- DQN replay size is deliberately bounded for local smoke runs; the full
  experiment config still needs a memory budget and possibly compact state
  observations.
- Level 0 generation/splits are not automated yet.

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

## Milestone 3: GPU Environment Prototype

- Avoid RGB rendering in the first GPU prototype.
- Represent each puzzle as fixed-shape integer tensors:
  - grid occupancy;
  - object ids;
  - object shapes;
  - goal mask;
  - agent/object positions;
  - done flags and step counts.
- Prototype kernels in Numba or Triton:
  - collision checks;
  - push dynamics;
  - goal-count updates;
  - batched random action stepping for throughput tests.
- Validate against the official Python environment on selected puzzles before
  using it for learning.

## Milestone 4: Optional Goal Relabeling

- Define achieved goals from object positions reached during failed episodes.
- Log both original-task and relabeled-task returns.
- Compare vanilla vs relabeled pipelines with both official and accelerated envs.

## Experiment Metrics

- Throughput: env steps/sec, learner updates/sec, samples/sec, wall-clock to fixed
  number of samples.
- Learning: success rate, mean episode return, solved-puzzle fraction, time to
  first nonzero success, area under success curve.
- Reproducibility: random seeds, puzzle set hash/path, git commit, submodule
  commit, hardware, CUDA/driver versions when applicable.
