# Friction Bloom

A small experimental algorithm prototype for probing systems with controlled friction and detecting useful adaptation.

## Core idea

Most algorithms optimize for ease, speed, or convergence. Friction Bloom intentionally disrupts a system just enough to see whether hidden structure emerges.

```txt
Baseline behavior
      ↓
Controlled friction
      ↓
Behavioral distortion
      ↓
Structure detection
      ↓
Bloom / noise / collapse
```

## What this repo does

This prototype uses a simple grid-world agent.

1. The agent finds a baseline path from start to goal.
2. Friction operators modify the environment.
3. The agent solves the modified environment.
4. The system scores whether the friction caused useful adaptation.

## Quick start

```bash
cd friction-bloom
python3 -m venv .venv
source .venv/bin/activate
python main.py
```

No third-party packages are required.

## Example output

```txt
=== FRICTION BLOOM RESULTS ===

1. block_baseline_midpoint
   outcome: bloom
   bloom_score: 2.42
   baseline_steps: 14
   altered_steps: 16
   notes: Agent recovered through a novel viable route.
```

## Repo structure

```txt
friction-bloom/
├── README.md
├── main.py
├── friction_bloom/
│   ├── __init__.py
│   ├── agent.py
│   ├── environment.py
│   ├── friction.py
│   ├── metrics.py
│   └── experiment.py
└── tests/
    └── test_smoke.py
```

## Algorithm sketch

```python
baseline = agent.solve(environment)

for friction in frictions:
    altered_env = friction(environment, baseline)
    altered = agent.solve(altered_env)
    score = bloom_score(baseline, altered)
```

## Interpretation

- **Bloom**: friction caused a novel, viable, coherent adaptation.
- **Noise**: friction changed behavior but not in a useful way.
- **Collapse**: friction made the system fail.

## Next upgrades

- Add a learned friction policy.
- Replace grid world with a prompt-chain simulator.
- Track repeated bloom points over many randomized maps.
- Export results as JSON or CSV.
