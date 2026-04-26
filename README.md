# PushWorld RL Study

This repository is a research workspace for improving RL performance on the
[DeepMind PushWorld benchmark](https://github.com/google-deepmind/pushworld).
The current focus is measurement-first optimization of PPO/DQN baselines:
compact observations, batching, profiling, compiler-level tuning, and only then
lower-level simulator work if it is still justified by the bottlenecks.

## Repository Layout

- `external/pushworld` - official DeepMind PushWorld benchmark, included as a
  git submodule.
- `src/pushworld_study` - local study code, wrappers, training scripts, and
  benchmark utilities.
- `docs/literature_review.md` - short literature and project survey.
- `docs/optimization_benchmark_plan.md` - staged plan for profiling and
  optimization experiments.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and keeps uv's cache inside
the project via [uv.toml](uv.toml):

```bash
uv cache dir
```

Expected output:

```text
.uv-cache
```

Clone with submodules, or initialize the submodule after cloning:

```bash
git submodule update --init --recursive
```

Create/update the environment:

```bash
uv sync
```

The upstream PushWorld Python code is not packaged as a normal wheel. The local
helpers add `external/pushworld/python3/src` to `sys.path` at runtime, and the
study code provides a native Gymnasium environment that reuses the official
puzzle parser, transition function, and renderer without importing the legacy
Gym wrapper.

## First Smoke Test

After `uv sync`, run:

```bash
uv run pushworld-study smoke-env
```

This checks that the native Gymnasium environment loads one official benchmark
puzzle, resets, and executes a few random actions.

Measure native environment throughput:

```bash
uv run pushworld-study profile-env --episodes 10 --max-steps 100
```

Install RL dependencies:

```bash
uv sync --group rl
```

Run short baseline smoke trainings:

```bash
uv run pushworld-study train-baseline ppo --total-timesteps 128
uv run pushworld-study train-baseline dqn --total-timesteps 16
```

These commands are integration checks, not meaningful learning experiments.
They save models under `models/` and TensorBoard logs under `runs/`.

Evaluate a saved model on held-out puzzles:

```bash
uv run pushworld-study eval-baseline ppo \
  models/ppo_smoke_seed0_100000.zip \
  --puzzle-path data/level0/base/test \
  --max-episodes 200
```

Run PPO with periodic held-out evaluation and best-checkpoint saving:

```bash
uv run pushworld-study train-baseline ppo \
  --puzzle-path data/level0/base/train \
  --eval-puzzle-path data/level0/base/test \
  --eval-freq 5000 \
  --n-eval-episodes 50 \
  --total-timesteps 100000 \
  --device cuda
```

Use compact structured observations instead of RGB pixels:

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
  --observation-mode planes
```

## Performance Figures

The figures below summarize the current findings.

![PushWorld benchmark overview](docs/assets/benchmark_overview.png)

![PushWorld observation modes](docs/assets/observation_modes.png)

![PushWorld bottleneck summary](docs/assets/bottleneck_summary.png)

![PushWorld performance summary](docs/assets/performance_summary.png)

Current findings:

- plane observations cut PPO's single-env throughput cost by about `3x`;
- batched planes with `4` to `16` envs push PPO to roughly `6.8x` to `9.2x`
  versus the vanilla RGB single-env baseline;
- DQN with planes is much faster than RGB, with roughly `20x` throughput gain
  on the five-puzzle benchmark;
- PPO and DQN are now mostly learner/update bound, not environment bound;
- `torch.compile` helps the isolated forward path, but only slightly improves
  full training throughput.

## Baseline Target

The original model-free setup from the paper is:

- algorithms: PPO and DQN;
- environment: OpenAI Gym wrapper with 100-step episodes;
- rewards: `+1` when an object moves onto its goal, `-1` when it moves off its
  goal, `+10` for completing all goals, `-0.01` per action;
- vision network: 3 convolutional layers with kernels `3x3`, `3x3`, `5x5`,
  strides `3`, `1`, `1`, followed by fully connected layers of sizes `256` and
  `128`, ReLU throughout;
- PPO hyperparameters: entropy cost `0.01`, learning rate `2e-4`, epochs `2`;
- DQN hyperparameters: learning rate `1e-4`, epsilon `0.05`,
  samples-per-insert `2`, batch size `256`, discount `1.0`, 1-step updates.

The benchmark still matters most as a controlled planning/RL testbed, but the
main study question in this repository is now where the training pipeline spends
time and which optimizations carry forward to later methods like relabeling,
sequence models, and hybrid planners.

Current implementation notes:

- local training uses a native Gymnasium PushWorld env;
- SB3 policies receive channel-first `uint8` observations so DQN replay buffers
  are practical;
- DQN uses a bounded replay buffer for smoke runs because SB3's default image
  replay buffer is too large for PushWorld observations;
- convolution channel counts are an explicit working assumption (`32, 64, 64`),
  because the paper specifies kernels/strides/FC sizes but not channels.
- the RL dependency group pins `torch==2.6.0`, which has official CUDA 12.4
  wheels and avoids accidentally resolving CUDA 13 wheels on systems with a
  CUDA 12.4 driver.
