# Experiment Notes

## Environment Throughput

Raw random-action stepping with the native Gymnasium RGB environment:

| Puzzle set | Episodes | Steps | Steps/sec | Mean reward/episode |
| --- | ---: | ---: | ---: | ---: |
| `data/level0/base/train` | 200 | 19,015 | 1,773 | -0.200 |
| `data/level0/all/train` | 200 | 18,707 | 1,126 | -0.0195 |
| `external/pushworld/benchmark/puzzles/level1` | 200 | 19,931 | 340 | -0.9965 |

Initial interpretation:

- Environment cost grows sharply with puzzle complexity.
- Level 1 raw stepping is about 5.2x slower than Level 0 base.
- This supports optimizing transition/rendering later, but PPO training fps was
  much lower than raw env steps/sec, so learner overhead and observation format
  are also important bottlenecks.

## PPO Baseline

Implementation:

- Stable-Baselines3 PPO.
- Native Gymnasium PushWorld env.
- RGB image observations converted to channel-first `uint8`.
- Paper-inspired CNN: kernels `3, 3, 5`, strides `3, 1, 1`, FC layer `256`;
  convolution channel counts are an explicit assumption: `32, 64, 64`.
- Initial paper-like PPO settings: learning rate `2e-4`, entropy coefficient
  `0.01`, epochs `2`.
- Later single-puzzle and five-puzzle runs used lower learning rate, lower
  entropy, more epochs, larger rollouts.

### Full/Many-Puzzle Runs

On `data/level0/base/train`, a 100k PPO run reached better rollout reward than
random but failed held-out evaluation:

| Run | Train set | Timesteps | Eval set | Deterministic success |
| --- | --- | ---: | --- | ---: |
| `PPO_5` | Level 0 base train | 100k | Level 0 base test, first 20 | 0/20 |

The reward curve alone was misleading: rollout reward improved, but saved-model
evaluation showed no held-out solves.

On a 20-puzzle debug subset:

| Run | Train set | Observation | Result |
| --- | --- | --- | --- |
| `PPO_7` | 20 Level 0 base train puzzles | RGB pixels | Eval reward stayed at `-1`; rollout collapsed to `-1`; entropy collapsed. |

Conclusion: the current pixel PPO baseline does not scale even to 20 easy
generated puzzles.

### Single-Puzzle Sanity Check

The one-puzzle overfit experiment showed the implementation can learn at least
one easy puzzle:

| Run | Train/eval set | Result |
| --- | --- | --- |
| `PPO_8` | one Level 0 base puzzle | Stochastic final policy solved 57/100 repeated eval episodes; deterministic eval failed. |
| `PPO_9` | one Level 0 base puzzle | Best checkpoint reached 100% deterministic and 100% stochastic success; final checkpoint kept 100% stochastic success but deterministic success fell to 1%. |

Interpretation:

- PPO can learn the single puzzle, so the environment/reward/model pipeline is
  not fundamentally broken.
- Deterministic argmax behavior is fragile. Policies may solve by sampling while
  deterministic evaluation remains poor.
- Best-checkpoint selection is mandatory for this baseline.

### Five-Puzzle Debug Runs

On five Level 0 base puzzles:

| Run | Train/eval set | Best checkpoint deterministic | Best checkpoint stochastic |
| --- | --- | ---: |---------------------------:|
| `PPO_10` | 5 Level 0 base train puzzles | 0/200 |                     30/200 |
| follow-up lower-LR run | 5 Level 0 base train puzzles | 0/200 |       9/200 by ~150k steps |

Interpretation:

- Lowering learning rate did not solve the instability.
- Stochastic success on five puzzles is nonzero, but unstable and far from
  usable.
- Deterministic success remains zero.

## Current PPO Conclusion

Baseline PPO with RGB pixel observations is not scaling well beyond one or a few
easy Level 0 puzzles. It can overfit a single puzzle, but performance becomes
unstable on five puzzles and collapses on larger small subsets.

For now, PPO should be treated as a weak reference baseline. Before investing in
longer PPO runs, the likely improvements are:

- compact symbolic/plane observations instead of rendered RGB pixels;
- goal-conditioned observations and hindsight relabeling;
- curriculum learning from single-puzzle/few-puzzle subsets;
- better evaluation/checkpointing split for deterministic vs stochastic policy
  behavior;
- possibly recurrent policies or search-guided/hybrid methods.

## Next Baseline: DQN

DQN is worth testing next because the PushWorld paper reported DQN generalized
better than PPO on Level 0 despite lower train accuracy. In this repo, DQN uses
a bounded replay buffer to avoid huge image replay memory.

Initial DQN experiments should start with the same debug ladder:

1. one Level 0 base puzzle;
2. five Level 0 base puzzles;
3. twenty Level 0 base puzzles;
4. full Level 0 base train/test.

The key comparison is whether DQN's value function is more stable than PPO's
stochastic policy on small puzzle sets.

## DQN Baseline

Implementation:

- Stable-Baselines3 DQN.
- Same native Gymnasium RGB pixel environment and paper-inspired CNN as PPO.
- Fixed epsilon `0.05`, matching the paper's reported DQN epsilon.
- Bounded replay buffer to avoid hundreds of GiB of image replay memory.

### Single-Puzzle Sanity Check

| Run | Train/eval set | Timesteps | FPS | Best deterministic eval | Best stochastic eval |
| --- | --- | ---: | ---: | ---: | ---: |
| `DQN_3` | one Level 0 base puzzle | ~10k | 7-10 | 100/100 | 96/100 |

TensorBoard summary for `DQN_3`:

- rollout reward improved from `1.50` at step 400 to about `9.99` by step
  11.3k;
- rollout episode length dropped from `100` to about `2.2`;
- eval at step 5k was still unsolved (`-1`, length `100`);
- eval at step 10k solved the puzzle (`9.99`, length `2`);
- training was extremely slow compared to PPO and raw env stepping: about
  `7-10 fps`.

Interpretation:

- DQN is far more sample-efficient than PPO on the single-puzzle overfit task.
- The learned policy is stable under deterministic evaluation, unlike PPO's
  stochastic-policy failure mode.
- The main drawback is wall-clock speed. DQN's current replay/training loop is
  the bottleneck, not raw environment stepping.

### Five-Puzzle Debug Run

| Run | Train/eval set | Timesteps | FPS | Best checkpoint deterministic | Best checkpoint stochastic | Final checkpoint deterministic | Final checkpoint stochastic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DQN_4` | 5 Level 0 base train puzzles | 50k | 7-10 | 80/200 | 78/200 | 80/200 | 78/200 |

TensorBoard summary for `DQN_4`:

- rollout reward improved from `1.50` at step 400 to `2.62` at step 49.8k;
- rollout episode length dropped from `100` to `68.6`;
- eval mean reward improved from `-0.12` at step 5k to `3.39` at step 50k,
  with best logged eval reward `4.27` at step 45k;
- eval mean episode length improved from `92.2` to `61.0`, with best logged
  value `53.7` at step 45k;
- deterministic and stochastic repeated evaluation were similar, unlike PPO.

Interpretation:

- DQN is a stronger pixel baseline than PPO on the debug ladder.
- The value-based policy is much more stable under deterministic evaluation.
- The five-puzzle result is already enough for the current baseline phase:
  extending DQN to more timesteps or more puzzles is less important than making
  the pipeline fast enough to test representation and algorithmic hypotheses.
- The main bottleneck is wall-clock speed. DQN is running around `8 fps`, far
  below both raw environment throughput and PPO throughput.

Next DQN runs, if needed later, should use shorter timesteps and earlier/more
frequent evaluation:

- one-puzzle DQN does not need 100k steps; 10k-20k is enough;
- five-puzzle DQN should start around 50k-100k steps with eval every 5k;
- if five-puzzle DQN works, then try 20 puzzles before the full Level 0 base
  split.

## Current Baseline Conclusion

- PPO with RGB pixels is unstable and does not scale past very small puzzle
  sets without additional tricks.
- DQN with RGB pixels is slower but more sample-efficient and more stable on
  small puzzle sets.
- The paper's qualitative observation that DQN may generalize more stably than
  PPO is consistent with our early debug results, although our setup is not an
  exact Acme/JAX reproduction.
- The next milestone should focus on training-pipeline speed and observation
  representation, not longer runs of the current pixel baselines.

## External PPO/DQN Config Survey

The quick survey did not uncover a directly reusable PushWorld PPO configuration
outside the paper.

Findings:

- DeepMind PushWorld paper: Acme/JAX PPO with entropy cost `0.01`, learning
  rate `2e-4`, epochs `2`; DQN with learning rate `1e-4`, epsilon `0.05`, batch
  size `256`, discount `1.0`, one-step updates.
- Stable-Baselines3 defaults are not Sokoban-specific. SB3 docs note that PPO
  often benefits from vectorized environments and may run better on CPU with
  subprocess envs for non-CNN policies; for our CNN setting GPU is still useful,
  but single-env PPO is not a strong throughput baseline.
- DI-engine includes a Sokoban environment wrapper but the published wheel/source
  inspected here does not include a Sokoban PPO config. Its adjacent visual
  discrete-control configs are informative:
  - Procgen Maze PPO: 4 collector envs, batch size `64`, entropy `0.01`,
    learning rate `1e-4`, update-per-collect `5`, collect sample count `100`.
  - Atari Pong PPO: 8 collector envs, frame stack 4, batch size `320`, learning
    rate `3e-4`, entropy `0.001`, epochs/update-per-collect equivalent of a
    larger on-policy update, advantage normalization and value normalization.
- gym-pcgrl trains procedural-content-generation agents with Stable-Baselines
  PPO2 and uses 50 subprocess environments by default. For Sokoban generation it
  uses cropped small-map observations and custom convolutional policies. This is
  useful for future PushWorld level-generation ideas, but it is not a Sokoban
  solving config.
- The public PushWorld side projects listed in the literature notes did not
  expose reliable, directly reusable PPO hyperparameters during this pass. Some
  repositories were unavailable through shallow/partial fetches or did not have
  accessible experiment reports.

Practical implication:

- Our current PPO issue is less likely to be fixed by copying one public config.
- More useful next steps are vectorized collection, structured observations,
  deterministic-vs-stochastic checkpointing, and eventually goal relabeling.
