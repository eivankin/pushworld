# Planner-Imitation Demo Notes

Streamlit app:

- `scripts/pushworld_policy_demo.py`

Current features:

- load a planner-imitation checkpoint;
- select Level 0 base test puzzles, including random selection;
- select Level 1 puzzles by name;
- edit raw `.pwp` text before running the model;
- run model rollout with configurable beam width, beam depth, top-k, and max
  steps;
- play the rollout with an FPS slider or inspect a specific step;
- display the model action string and per-step action table;
- optionally run the official RGD planner for comparison and display its plan,
  plan length, and solve time.

Default checkpoint:

- `models/planner_imitation_level0_base_small.pt`

Run:

```bash
uv run streamlit run scripts/pushworld_policy_demo.py
```

The `.pwp` editor is intentionally raw text. This makes it easy to perturb a
known puzzle during a live demo and show that the model is not using hardcoded
solutions.
