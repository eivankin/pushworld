# Low-Level Optimization Notes

This note collects practical learner-side optimization ideas after the current
profiling and `torch.compile` experiments.

## Current Situation

The current measurements point to a consistent bottleneck picture:

- PPO with plane observations and vectorized rollout is mostly update-bound.
- DQN with plane observations is also mostly update-bound.
- `torch.compile` improves the isolated PushWorld CNN forward pass
  substantially, but barely changes end-to-end training-step throughput.

Observed numbers:

- PPO planes, `16` envs:
  - rollout: `22.5%`
  - update: `77.4%`
- DQN planes, `1` env:
  - rollout: `23.8%`
  - update: `76.0%`
- `torch.compile`, batch `256`:
  - inference: `1.56x`
  - full train step: `1.04x`
- `torch.compile`, batch `512`:
  - inference: `1.43x`
  - full train step: `1.03x`

Interpretation:

- raw model execution is not the only learner cost;
- optimizer/backward/training-loop/replay overhead are likely large enough that
  forward-pass acceleration alone is not sufficient.

## Recommended Next Low-Level Experiments

These are ordered by expected value, not by implementation novelty.

### 1. Automatic mixed precision

Hypothesis:

- if backward/optimizer math is a significant cost, AMP may improve full
  training more than `torch.compile` did.

Why it is promising:

- unlike compile, AMP affects forward, backward, and optimizer-related tensor
  traffic;
- it directly targets the update-heavy part of PPO/DQN.

Implementation options:

- easiest path is a custom training loop or a thin SB3 patch around the policy
  update;
- use `torch.autocast(device_type="cuda")` and `GradScaler` if needed.

Success criterion:

- full train-step or train-FPS speedup, not just forward-pass speedup.

### 2. Smaller or cheaper model variants

Hypothesis:

- if update time remains dominant, a narrower CNN may improve end-to-end
  training more than compiler optimizations.

Experiments:

- reduce `features_dim`;
- reduce channel counts or FC width;
- compare CNN encoder vs even simpler plane-specific encoder.

Why this matters:

- the benchmark is tiny spatially on Level 0 debug sets, so current model
  capacity may be oversized relative to the input.

### 3. PPO update-parameter sweep

Hypothesis:

- PPO may spend too much wall-clock time in repeated minibatch/epoch updates for
  limited learning benefit.

Parameters to probe:

- `n_epochs`
- `batch_size`
- `n_steps`
- `n_envs`

Important framing:

- this is still a systems experiment, not just an RL-tuning exercise, because
  these parameters directly change update/rollout cost balance.

### 4. DQN replay/update profiling refinement

Hypothesis:

- the current DQN update bucket is too coarse; replay sampling, host-device
  transfer, target-network updates, and backward pass may have very different
  costs.

Next instrumentation targets:

- replay sampling time;
- batch tensor transfer time;
- forward/backward split;
- optimizer step time.

Why it matters:

- once these are separated, we can choose between replay-layout changes,
  prefetching, AMP, or model simplification.

### 5. CUDA graph capture

Hypothesis:

- if training uses highly repetitive fixed-shape steps, CUDA graphs may reduce
  Python launch overhead more than compile alone.

Caveat:

- this is lower priority than AMP and better profiling;
- it is only worth trying after shapes and update loops are stable.

### 6. Triton custom kernels

Hypothesis:

- a custom Triton kernel may help specific hot tensor operations or replay/data
  movement paths.

Caveat:

- this is not the next rational step.
- We currently do not know which low-level tensor op is hot enough to justify a
  custom kernel.
- Writing Triton before finer learner profiling would be premature.

## What Not To Do Next

Based on current evidence, these are lower priority:

- GPU environment rewrite for current PPO/DQN baselines;
- subprocess vectorization for plane observations;
- more `torch.compile` variants without first improving training-loop
  instrumentation;
- random low-level Triton rewrites without a measured hotspot.

## Best Short-Term Experiment

If there is time for exactly one more low-level experiment soon, it should be:

1. add finer learner/update timing;
2. test mixed precision on the policy update.

Reason:

- AMP is the next most plausible optimization that could improve the whole
  update step rather than just isolated inference.

## Why This Matters For Future Algorithms

These low-level results are not specific to current PPO only.

- Relabeling still trains neural policies and will inherit the same learner
  bottlenecks.
- Goal-conditioned policies still depend on efficient updates.
- Hybrid planner+learner approaches still need fast local policy training and
  evaluation.
- Offline transformer policies may shift the exact bottleneck, but the
  measurement-first approach remains the same.
- RAGEN-style LLM-agent pipelines shift the bottleneck again toward token
  generation, rollout batching, Ray/vLLM scheduling, and LoRA/update cost. The
  same profiling discipline should be applied there, with tokens/sec and
  time-to-target-success added to the metrics.
