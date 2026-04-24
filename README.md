# PushWorld RL Study

This repository is a research workspace for improving RL performance on the
[DeepMind PushWorld benchmark](https://github.com/google-deepmind/pushworld).
The initial focus is a faithful PPO/DQN baseline, then a batched/GPU environment
implementation to measure whether faster simulation improves wall-clock learning.

The original project notes were moved to [notes.md](notes.md).

## Repository Layout

- `external/pushworld` - official DeepMind PushWorld benchmark, included as a
  git submodule.
- `src/pushworld_study` - local study code, wrappers, training scripts, and
  benchmark utilities.
- `docs/literature_review.md` - short literature and project survey.
- `docs/implementation_plan.md` - staged plan for baseline, profiling, and GPU
  environment work.

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

## Baseline Target

The paper's model-free setup is:

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

The first reproduction should measure throughput and learning on Level 0 before
moving to Level 1, because the paper reports very low model-free success on
hand-designed Level 1 puzzles even after 350M steps.

Current implementation notes:

- local training uses a native Gymnasium PushWorld env;
- SB3 policies receive channel-first `uint8` observations so DQN replay buffers
  are practical;
- DQN uses a bounded replay buffer for smoke runs because SB3's default image
  replay buffer is too large for PushWorld observations;
- convolution channel counts are an explicit working assumption (`32, 64, 64`),
  because the paper specifies kernels/strides/FC sizes but not channels.
