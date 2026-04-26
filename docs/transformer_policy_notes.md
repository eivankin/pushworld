# Transformer Policy Notes

This note summarizes the Sokoban transformer-policy direction and how it could
transfer to PushWorld.

## What Tim Wheeler's Sokoban Project Actually Does

Main references:

- Tim Wheeler, "A Transformer Sokoban Policy":
  https://timallanwheeler.com/blog/category/sokoban/
- Tim Wheeler, "Rollouts on the GPU":
  https://timallanwheeler.com/blog/category/sokoban/

Key design points from those posts:

- The setup is primarily supervised sequence modeling, not PPO or DQN.
- A board is encoded as an `8x8x5` tensor with channels for walls, floor,
  goals, player, and boxes.
- The model predicts:
  - the next action;
  - a discretized `nsteps` target, including an unsolvable bucket.
- The transformer consumes an entire sequence during teacher-forced training,
  with the goal state included in the context.
- The initial model scale was small: `3` transformer layers,
  `16`-dimensional embeddings, max sequence length `32`, and `8` attention
  heads.
- Validation is not just imitation accuracy. The policy is also used inside
  beam search to actually solve held-out puzzles.

Reported results from the blog:

- solvability accuracy: `97.7%`;
- top-1 policy accuracy: `86.7%` for the full transformer vs `83.1%` for a
  no-history baseline;
- solve rate with beam search: `92.1%` for the full transformer vs `87.9%` for
  the no-history baseline.

Important systems point from the follow-up GPU rollout post:

- training already benefited heavily from teacher forcing because entire
  sequences can be processed in parallel on GPU;
- rollout/search stayed slow because policy inference happened on GPU while
  state advancement stayed on CPU;
- moving rollout state propagation to GPU gave roughly `60x` rollout speedup in
  that setup;
- the optimization target was specifically CPU<->GPU synchronization during
  autoregressive inference.

## Why This Is Relevant To PushWorld

This direction matches PushWorld in several ways:

- PushWorld is deterministic and has a small discrete action space, so
  teacher-forced action prediction is natural.
- PushWorld is fundamentally a planning benchmark, so a policy that guides beam
  search is conceptually closer to the benchmark than plain PPO.
- We already have compact plane observations, which are a better starting point
  than RGB images for a transformer policy.
- The official repo includes planning infrastructure and benchmark solutions,
  and Level 0 generation already includes solvability filtering through the RGD
  planner.

This direction is not a drop-in replacement for our current baselines:

- it needs offline training data with state/action trajectories;
- it likely needs goal-conditioned inputs;
- evaluation should include search-guided solving, not just greedy rollout.

## What Would Need To Change For PushWorld

### 1. Training data pipeline

The first blocker is data.

For Sokoban, Wheeler trains from supervised sequences. For PushWorld we would
need a dataset of:

- initial puzzle state;
- goal description or goal state proxy;
- intermediate states along a solution trajectory;
- target actions;
- optional remaining-steps target or solvability target.

Potential data sources:

- official benchmark solutions under `external/pushworld/benchmark/solutions`;
- Level 0 generated puzzles filtered by the official RGD planner;
- future planner-generated trajectories from the C++ RGD solver or PDDL/SAS
  pipeline.

The benchmark solution files are useful for bootstrapping the format, but they
are too small to train a high-capacity policy by themselves. A realistic
supervised setup probably needs many more generated Level 0 trajectories.

### 2. Input representation

For PushWorld, the closest analogue to Wheeler's board tensor is a
goal-conditioned plane stack.

A reasonable first representation is:

- current-state planes:
  - walls;
  - agent-only walls;
  - agent;
  - goal-associated movable objects;
  - other movable objects;
  - goal cells.
- goal-context planes:
  - target object footprint planes;
  - optional goal-mask plane;
  - optional solved-object plane.

Two plausible input formats:

1. concatenate current-state and goal-state planes channel-wise, then patchify
   or flatten into transformer tokens;
2. encode each timestep with a small CNN first, then run the transformer over
   the sequence of latent state embeddings.

For PushWorld, option 2 is the safer first prototype. It fits the existing code
better and handles variable board sizes more easily.

### 3. Output heads

The Sokoban setup has two heads: action and `nsteps`.

For PushWorld the same pattern is attractive:

- action head over the 4 movement actions;
- solvability / distance-to-go head, likely bucketed rather than regressed.

The auxiliary head matters because PushWorld has many locally plausible but
globally doomed states. A value-like or distance-to-go signal could help search
ranking much more than plain imitation logits.

### 4. Evaluation protocol

A fair evaluation should not stop at action accuracy.

We should measure:

- top-1 and top-k action accuracy on held-out trajectories;
- solvability or distance-bucket accuracy;
- greedy rollout solve rate;
- beam-search solve rate using the policy as a proposal/scoring model.

The beam-search part is important. Wheeler's own results show that solve rate is
more informative than plain supervised metrics.

## How Hard Would It Be?

Short answer: feasible, but not cheap.

### Small proof of concept: medium difficulty

Scope:

- Level 0 only;
- offline imitation from planner trajectories;
- CNN encoder plus small transformer over state history;
- greedy rollout eval only.

Estimated difficulty: moderate.

Main work items:

- extract solution trajectories into a training dataset;
- build a sequence dataloader;
- implement the model and training loop;
- add held-out trajectory metrics.

This is a reasonable near-term milestone.

### Stronger version with search: hard, but realistic

Scope:

- add beam search guided by policy logits and/or `nsteps` head;
- compare greedy vs beam search on Level 0 and curated Level 1 subsets;
- optionally batch rollout/search on GPU.

Estimated difficulty: high.

This is where the architecture becomes genuinely interesting for PushWorld,
because it starts acting like a learned planner instead of just a behavioral
clone.

### Full Wheeler-style systems version: research project

Scope:

- GPU rollout state propagation;
- batched search on GPU;
- larger models and larger generated datasets;
- careful handling of variable board geometry and longer horizons.

Estimated difficulty: very high.

This is viable as a later milestone, but it is not a "quick baseline swap".

## What Makes PushWorld Harder Than Sokoban

There are several reasons this is harder to reproduce on PushWorld than on the
8x8 Sokoban setup from the blog:

- PushWorld levels have more heterogeneous object roles and shapes.
- Level geometry varies more.
- Some puzzles require tool use and obstacle choreography, not just direct
  box-to-goal transport.
- The natural action space is still primitive moves, so solution horizons can
  get long.
- Good search heuristics may need stronger object-identity handling than a
  plain occupancy stack.

This pushes us toward either:

- richer goal-conditioned planes with object-specific channels; or
- an object-centric representation later on.

## Why This Still Looks Worth Doing

Even with the extra engineering, this direction has two advantages over
continuing to tune PPO:

1. it matches the planning nature of the benchmark better;
2. it creates a clearer path to search-guided inference, which is likely more
   important than squeezing a few more percent out of PPO.

Given our current profiling results, this direction also aligns better with the
systems story:

- once PPO is vectorized, the learner becomes update-bound;
- a transformer policy trained offline with teacher forcing can exploit large
  batched GPU training much better than on-policy PPO rollouts;
- if we later need fast autoregressive search, that is exactly where a
  GPU-resident environment becomes valuable.

## Recommended Future Implementation Path

1. Build a trajectory export pipeline from benchmark solutions and generated
   Level 0 puzzles solved by the official planner.
2. Add a goal-conditioned offline dataset format:
   `(goal, state_0, action_0, state_1, action_1, ...)`.
3. Implement a conservative first model:
   CNN state encoder + small transformer over recent states + action head +
   bucketed distance-to-go head.
4. Benchmark against:
   - greedy action rollout;
   - beam search with width `8`, `16`, `32`.
5. Only after that, consider GPU rollout/search kernels.

## Bottom Line

Using a Wheeler-style transformer policy on PushWorld is possible, but it
should be framed as a new planning-oriented pipeline:

- offline supervised trajectories;
- goal-conditioned sequence model;
- search-guided inference.

It is not just "replace the PPO CNN with a transformer".
