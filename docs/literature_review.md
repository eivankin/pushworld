# Short Literature and Project Review

## PushWorld

- DeepMind introduced PushWorld as a benchmark for manipulation planning with
  tools and movable obstacles: https://arxiv.org/abs/2301.10289.
- Official benchmark/code: https://github.com/google-deepmind/pushworld.
- The benchmark is primarily a planning benchmark. The paper also includes
  model-free RL baselines with PPO and DQN, but reports poor Level 1 performance:
  DQN converged below 1% solved and PPO around 6% solved after 350M steps.
- The official repo provides Gym and dm_env wrappers, Level 1-4 hand-designed
  puzzles, Level 0 generation scripts, rendering, PDDL/SAS conversion, and the
  C++ Recursive Graph Distance planner.

## Sokoban and Box-Pushing RL

- The PushWorld reward setup follows the Sokoban-style shaped reward used in
  Guez et al., "An Investigation of Model-Free Planning" / Sokoban RL work:
  positive reward for placing boxes/goals, negative reward for removing them,
  terminal success reward, and a small step penalty.
- Sokoban remains the closest practical analog because it combines sparse
  rewards, irreversible-looking mistakes, long-horizon credit assignment, and
  strong classical planning baselines.

### Concrete Sokoban References

- Imagination-Augmented Agents for Deep Reinforcement Learning, Racaniere et al.,
  NeurIPS 2017: https://papers.nips.cc/paper/7152-imagination-augmented-agents-for-deep-reinforcement-learning.
  This is one of the main DeepMind Sokoban RL references. It combines a learned
  environment model with policy learning and explicitly evaluates on Sokoban.
  Relevance for PushWorld: useful as a model-based/model-free hybrid baseline
  idea after PPO/DQN reproduction.
- An Investigation of Model-Free Planning, Guez et al., ICML 2019:
  https://proceedings.mlr.press/v97/guez19a.html. This is the closest RL paper
  to PushWorld's baseline setup. It studies whether recurrent model-free agents
  can learn planning-like behavior on combinatorial domains including Sokoban.
  Relevance for PushWorld: motivates recurrent policies, test-time thinking
  probes, and Boxoban-style train/test generalization.
- Boxoban levels dataset: https://github.com/google-deepmind/boxoban-levels.
  DeepMind's standardized 10x10, four-box Sokoban-like dataset. It includes
  unfiltered, medium, and hard splits, and is used by model-free planning work.
  Relevance for PushWorld: good reference for procedural level split design and
  fair train/validation/test reporting.
- gym-sokoban: https://github.com/mpSchrader/gym-sokoban. A widely used legacy
  Gym environment for Sokoban, with random solvable level generation and Boxoban
  support. Relevance for PushWorld: useful for environment API comparisons and
  observation/reward conventions, but it is also legacy Gym-based.
- PCGRL Gym interface: https://github.com/amidos2006/gym-pcgrl. This project
  trains RL agents to generate playable levels, including Sokoban levels, using
  procedural-content-generation representations such as narrow, wide, and turtle.
  Relevance for PushWorld: useful design reference if we later want to generate
  PushWorld Level 0/Level 1-like puzzles instead of only consuming the official
  generator. Its Sokoban task is level generation, not Sokoban solving.
- MiniHack Boxoban port:
  https://minihack.readthedocs.io/en/latest/envs/ported/boxoban.html. MiniHack
  exposes Boxoban variants with a modern benchmark framing and shaped rewards.
  Relevance for PushWorld: useful example of presenting Sokoban-like tasks inside
  a broader environment suite.
- Fast Sokoban Environment for Deep Reinforcement Learning:
  https://github.com/AlexanderKoch-Koch/SokobanEnv. This repo emphasizes fast
  non-graphical simulation over rendering and cites roughly 10 microseconds per
  step in its README. Relevance for PushWorld: supports our profiling-first
  argument that RGB rendering should be separated from transition dynamics.
- An analysis of Single-Player Monte Carlo Tree Search performance in Sokoban,
  Expert Systems with Applications 2022:
  https://www.sciencedirect.com/science/article/pii/S0957417421015372.
  Relevance for PushWorld: search methods remain a serious comparison point;
  MCTS/IDA* results can shape hybrid RL+search baselines.
- Solving Sokoban with backward reinforcement learning, Shoham and Elidan, 2021:
  https://arxiv.org/abs/2105.01904. Relevance for PushWorld: backward or reverse
  curricula may be useful because PushWorld puzzle generators already need
  solvability checks and many puzzles are easier to reason about from solved
  states.
- Planning behavior in a recurrent neural network that plays Sokoban,
  Garriga-Alonso et al., 2024: https://arxiv.org/abs/2407.15421. Relevance for
  PushWorld: not a baseline algorithm, but a useful analysis template if we train
  recurrent policies and want to test whether they encode plans.
- Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks,
  Pastukhov, 2025: https://arxiv.org/abs/2504.04366. Relevance for PushWorld:
  modern example of learned subgoal decomposition applied to Boxoban/Sokoban,
  directly aligned with the optional goal-conditioned/relabeling direction.
- Learn to Follow: Decentralized Lifelong Multi-agent Pathfinding via Planning
  and Learning, Skrynnik et al., 2023/2024:
  https://arxiv.org/abs/2310.01207. Relevance for PushWorld: strong example of
  a hybrid pipeline where heuristic planning provides long-range structure and a
  learned policy handles local adaptation. The task is not the same as
  PushWorld, but the decomposition is highly relevant.
- Tim Wheeler, "A Transformer Sokoban Policy":
  https://timallanwheeler.com/blog/category/sokoban/. Relevance for PushWorld:
  practical example of training a goal-conditioned sequence policy with
  teacher forcing and evaluating it through beam search rather than plain RL
  reward curves.
- Tim Wheeler, "Rollouts on the GPU":
  https://timallanwheeler.com/blog/category/sokoban/. Relevance for PushWorld:
  concrete systems example showing that once a policy runs on GPU, rollout can
  still remain bottlenecked by CPU state advancement and CPU<->GPU transfers.
- RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn
  Reinforcement Learning, Wang et al., 2025:
  https://arxiv.org/abs/2504.20073; code:
  https://github.com/mll-lab-nu/RAGEN. RAGEN is a multi-turn RL framework for
  LLM agents: the model receives a text state, produces reasoning and an action,
  the environment returns feedback/reward, and the policy is updated from
  trajectory-level outcomes. The repository includes Sokoban-style tasks and
  Gym-compatible custom environments, which makes PushWorld adaptation
  realistic through a text renderer, action parser, and Gym wrapper. Relevance
  for PushWorld: RAGEN is not a fastest-solver target; it is a useful heavy
  LLM-agent pipeline to adapt and optimize. Concrete optimization targets are
  prompt/context length, reasoning-token budget, vLLM rollout batching,
  Ray/environment placement, LoRA vs full update cost, and rollout filtering.
  The paper reports runtime/resource context rather than a complete throughput
  table: Figure 5 gives Sokoban PPO total-time plots for filtering variants,
  Appendix C.2 describes the H100/A100 + vLLM/Ray/FSDP setup, and Appendix H
  compares LoRA resource usage.

### Relevant Directions to Compare Against

- Model-free baselines: PPO, DQN/Rainbow-style value learning, recurrent PPO/R2D2
  style policies, and test-time recurrent "thinking" probes.
- Model-based or planning-guided methods: I2A-style rollouts, MCTS, IDA*,
  learned heuristics, value-guided search.
- Curriculum and procedural generation: Level 0 PushWorld generation should be
  treated like Boxoban split design, with fixed train/test seeds and explicit
  difficulty parameters.
- Hindsight / goal relabeling: useful when failed trajectories still contain
  achieved object-goal configurations; HER is the canonical starting point:
  https://papers.nips.cc/paper/7090-hindsight-experience-replay.
- LLM-agent RL pipeline optimization: RAGEN provides a newer direction where
  PushWorld is exposed as a text-interaction environment for an LLM policy. This
  should be evaluated as a systems optimization target with explicit
  time-to-quality metrics, not as a hand-written solver replacement.

## Systems Motivation

- Tim Wheeler's Sokoban posts are a useful systems reference for fast batched
  simulation: https://timallanwheeler.com/blog/category/sokoban/.
- For PushWorld, the main systems question is whether environment stepping, image
  rendering, or policy learning dominates wall-clock time. Before building GPU
  kernels, profile:
  - raw Python environment steps/sec;
  - vectorized multi-process environment steps/sec;
  - observation rendering cost;
  - policy update cost on CPU vs GPU.
- The current profiling result weakens the case for making a GPU environment the
  immediate next step for PPO/DQN: after plane observations and vectorized
  rollout, learner/update cost dominates short training runs. GPU-resident state
  propagation becomes more relevant later for batched beam search,
  transformer-policy inference, or LLM rollout systems where environment/model
  synchronization becomes the bottleneck.

## Gymnasium Compatibility

- The official PushWorld repo exposes a legacy Gym environment, while modern RL
  libraries have mostly moved to Gymnasium.
- This study repo does not import the official `pushworld.gym_env` wrapper.
  Instead, `pushworld_study.envs.PushWorldGymnasiumEnv` is a native Gymnasium
  rewrite that uses the official parser, transition function, and renderer.
- This avoids Gym's deprecation warning, removes the old-Gym dependency, and
  keeps Stable-Baselines3/Puffer-style integration cleaner without patching the
  DeepMind submodule.

## Nearby Public Projects

- https://github.com/mazebench/pushworld_heuristic - recent heuristic-search
  project. Potential comparison point, but publication status should be checked
  before citing in a paper.
- https://github.com/julianh65/PufferPushworldExploration - possible PushWorld
  integration with PufferLib/PufferAI. Worth testing as an environment-throughput
  reference.
- https://github.com/Jgroner11/pushworld_bot - barebones bot implementation.
  Could be inspected for ideas, but current scientific value is unclear.

## Initial Position

The highest-value first result is not a new algorithm. It is a reproducible
baseline table with:

- official Python env throughput;
- PPO and DQN smoke-learning curves on Level 0;
- profile breakdown of stepping/rendering/training;
- a clear decision on whether a GPU environment is likely to move wall-clock
  learning speed.

## Transformer-Policy Follow-Up

See [transformer_policy_notes.md](transformer_policy_notes.md) for a more
concrete assessment of whether a Sokoban-style transformer policy plus search is
worth implementing for PushWorld. The short version is yes, but only as a new
offline planning pipeline rather than as a direct SB3 PPO replacement.

See [hybrid_solver_notes.md](hybrid_solver_notes.md) for a complementary note on
hybrid planning+learning approaches in the style of FOLLOWER. The short version
is that this looks promising for PushWorld as a medium-to-long-term direction,
especially after we have goal-conditioned observations and relabeling.

## RAGEN Follow-Up

RAGEN should be tracked as a separate LLM-agent pipeline:

1. adapt PushWorld as a Gym-compatible text environment;
2. run a no-finetune fixed-prompt baseline and log full trajectories;
3. profile wall time, tokens/sec, tokens/episode, generation/update/env-step
   split, and GPU memory;
4. test prompt compression, reasoning-token caps, action-only answers, rollout
   batching, vLLM/Ray settings, LoRA, and rollout filtering.

This direction is newer and heavier than the compact-policy pipelines, so it
belongs after the base RL, transformer-policy, and hybrid-solver directions in
the presentation narrative.
