# Hybrid Solver Notes

This note summarizes the relevance of hybrid planning+learning approaches for
PushWorld, with emphasis on:

- FOLLOWER: Learn to Follow: Decentralized Lifelong Multi-agent Pathfinding via
  Planning and Learning, Skrynnik et al., 2023/2024:
  https://arxiv.org/pdf/2310.01207.pdf
- the general idea of splitting long-horizon structure from local reactive
  adaptation.

## What FOLLOWER Actually Does

FOLLOWER is not an end-to-end RL solver. It is a two-module pipeline:

1. a heuristic path decider builds an individual long-range path;
2. a learnable follower uses PPO to reach the next waypoint while avoiding
   local conflicts and making short detours when needed.

From the paper:

- each agent plans its own path to the goal with heuristic search and replans
  as needed;
- the planner explicitly penalizes congested cells with:
  - a static cost derived from map topology;
  - a dynamic cost accumulated from observed traffic;
- the policy sees a local observation and learns short-term conflict
  resolution;
- the policy is trained with PPO and a recurrent network;
- the immediate learning target is reaching the next waypoint, which creates a
  dense reward signal without heavy manual reward shaping.

Important result from the paper:

- ablating either the RL follower or the congestion-aware planning costs hurts
  throughput;
- the hybrid works better than purely learnable baselines on unseen maps and
  scales better than a heavy search baseline in runtime.

## Why This Is Interesting For PushWorld

The core idea transfers well:

- PushWorld is long-horizon and combinatorial, so pure reactive RL struggles.
- A planner can provide global structure: which object to interact with, which
  region to clear first, which intermediate state to aim for.
- A learned controller can handle short-horizon execution: local movement,
  obstacle avoidance, and robust action selection once the subgoal is known.

This is attractive because our current results already suggest:

- PPO is weak as an end-to-end solver;
- DQN is more stable but still only a baseline;
- the benchmark itself is fundamentally planning-oriented.

## The Most Natural PushWorld Translation

PushWorld is not multi-agent MAPF, so the transfer is conceptual rather than
literal.

The right analogue is:

- planner for long-horizon subgoals or waypoints;
- learned policy for local execution toward those subgoals.

Concrete planner outputs for PushWorld could be:

- next target object to move;
- next target region or corridor to clear;
- next intermediate board state or partial-goal condition;
- short waypoint sequence for the agent position.

Concrete RL policy responsibilities could be:

- execute local motion efficiently;
- recover when the exact planner path is blocked or brittle;
- learn short tactical behaviors around object interactions;
- optionally decide when to give up on the current subgoal and request replanning.

## Candidate Hybrid Designs For PushWorld

### 1. Planner chooses subgoal, RL executes primitive actions

Planner output:

- a subgoal such as "put object A into region R" or "move the agent to entry
  point P".

RL input:

- current state planes;
- subgoal encoding;
- optional short action/state history.

RL output:

- primitive move action.

This is the most straightforward hybrid design and likely the best first one.

### 2. Planner proposes a path, RL follows with local deviations

Planner output:

- a path or waypoint sequence in state space or agent-position space.

RL input:

- local state;
- current target waypoint;
- optional next few waypoints.

RL output:

- primitive move action.

This is closest to FOLLOWER conceptually, but it is harder in PushWorld because
waypoints in pure agent-position space may be insufficient when the important
decisions are about object ordering.

### 3. Planner proposes search candidates, learned model ranks them

Planner output:

- several candidate partial plans or successor states.

Learned model output:

- a score/value/ranking over those candidates.

This is less about local control and more about learning a heuristic for search.
It may eventually fit PushWorld better than waypoint following.

## How Hard Would It Be?

### Short-term prototype: moderate

Feasible near-term version:

- planner provides simple subgoals based on official solutions or heuristics;
- RL policy receives goal-conditioned plane observations;
- evaluation on Level 0 debug sets.

This is realistic after relabeling / goal-conditioned observations.

### Stronger research version: hard

More ambitious version:

- planner generates subgoals online;
- RL decides when to replan;
- search and learning interact throughout inference;
- evaluation includes harder benchmark levels.

This is a real research milestone, not a quick engineering task.

## Why This May Be Better Than More PPO Tuning

The main reason is mismatch.

Pure PPO is being asked to do all of the following at once:

- infer global plan structure;
- discover useful intermediate goals;
- learn local movement control;
- cope with sparse long-horizon credit assignment.

The hybrid approach decomposes this:

- planning handles long-range structure;
- learning handles local adaptation.

That decomposition is exactly what FOLLOWER argues for in MAPF, and the same
logic is credible for PushWorld.

## What Needs To Exist First

This is not the immediate next implementation step. Before a hybrid solver is
worth building, we should have:

1. compact goal-conditioned observations;
2. relabeling or another way to make subgoal learning data-efficient;
3. a small interface for planner-derived subgoals;
4. stronger profiling around where inference/search time goes.

Without those pieces, a hybrid solver would be too unconstrained and difficult
to evaluate.

## Recommended Future Path

1. Short-term:
   implement goal-conditioned observations and relabeling.
2. Next:
   train a local subgoal-conditioned policy on Level 0.
3. Then:
   connect that policy to planner-produced subgoals or short solution prefixes.
4. Later:
   compare hybrid execution against:
   - pure DQN/PPO;
   - planner alone;
   - transformer policy + beam search.

## Optimization Plan

The presentation frames this as:

`current state -> planner subgoals -> goal-conditioned executor -> progress
monitor -> accept or replan`.

Concrete implementation and optimization steps:

1. Slice existing solution traces into reachable partial goals every `k`
   actions. Store the current state, subgoal state, action segment, and distance
   to the subgoal.
2. Train an executor baseline on Level 0 subgoals:
   - imitation from planner segments;
   - relabeling from partially successful attempts;
   - goal-conditioned plane observations.
3. Profile the runtime interface before making it more complex:
   - planner call time;
   - executor inference time;
   - environment stepping time;
   - progress-monitor time;
   - replanning overhead.
4. Reduce planner calls:
   - cache subgoal plans;
   - replan only on timeout;
   - replan when distance-to-subgoal regresses for several steps;
   - keep a short blacklist of recently failed subgoals.
5. Add a learned ranker/value only after the simple hybrid runs:
   - planner proposes several candidate subgoals or prefixes;
   - model ranks candidates;
   - executor follows the selected candidate.
6. Run ablations:
   - planner-only;
   - executor-only;
   - hybrid without ranker;
   - hybrid with ranker/value.

Metrics:

- subgoal success rate;
- replans per puzzle;
- planner time vs executor time;
- solved puzzles per minute;
- wall-clock per solved puzzle;
- failure modes by subgoal type.

## Bottom Line

The FOLLOWER idea is relevant to PushWorld not because the tasks are identical,
but because the decomposition is right:

- planning for long-horizon structure;
- learning for local adaptation.

For PushWorld, this looks more promising as a medium-to-long-term direction than
continued end-to-end PPO tuning.
