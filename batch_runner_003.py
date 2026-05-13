"""
Friction Bloom — Batch Runner
Dataset 003: Friction Combinations at Scale
Tests all pairwise and full combinations of friction types.
50 reps per combination per grid size.
Appends to existing data/runs.jsonl — Dataset 002 is preserved.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from friction_bloom.agent import GreedyAgent
from friction_bloom.environment import GridWorld
from friction_bloom.experiment import run_experiment
from friction_bloom.friction import (
    add_wall_gap_constraint,
    block_baseline_midpoint,
    block_first_third,
    add_scattered_obstacles,
)

# ── Configuration ─────────────────────────────────────────────────────────────

GRID_SIZES = list(range(5, 16))  # 5x5 through 15x15

# All pairwise combinations plus full four-friction combo
FRICTION_COMBOS = [
    [add_wall_gap_constraint, block_first_third],
    [add_wall_gap_constraint, add_scattered_obstacles],
    [add_wall_gap_constraint, block_baseline_midpoint],
    [block_first_third, block_baseline_midpoint],
    [block_first_third, add_scattered_obstacles],
    [add_scattered_obstacles, block_baseline_midpoint],
    [add_wall_gap_constraint, block_first_third, add_scattered_obstacles, block_baseline_midpoint],
]

COMBO_NAMES = [
    "wall_gap+first_third",
    "wall_gap+scattered",
    "wall_gap+midpoint",
    "first_third+midpoint",
    "first_third+scattered",
    "scattered+midpoint",
    "all_four",
]

REPS_PER_COMBO = 50

DATA = Path("data/runs.jsonl")

# ── Runner ────────────────────────────────────────────────────────────────────

def save_run(record: dict) -> None:
    DATA.parent.mkdir(exist_ok=True)
    with open(DATA, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_batch() -> None:
    total = len(GRID_SIZES) * len(FRICTION_COMBOS) * REPS_PER_COMBO
    completed = 0
    start_time = time.time()

    print(f"\nFriction Bloom Batch Runner — Dataset 003")
    print(f"Grid sizes: {GRID_SIZES[0]} to {GRID_SIZES[-1]}")
    print(f"Friction combinations: {len(FRICTION_COMBOS)}")
    print(f"Reps per combo: {REPS_PER_COMBO}")
    print(f"Total runs: {total}")
    print(f"Output: {DATA} (appending — Dataset 002 preserved)")
    print("-" * 60)

    for grid_size in GRID_SIZES:
        for combo_idx, friction_combo in enumerate(FRICTION_COMBOS):
            combo_name = COMBO_NAMES[combo_idx]
            for rep in range(REPS_PER_COMBO):
                env = GridWorld(
                    width=grid_size,
                    height=grid_size,
                    start=(0, 0),
                    goal=(grid_size - 1, grid_size - 1)
                )
                agent = GreedyAgent()

                results = run_experiment(agent, env, friction_combo)

                # Record combined result — use highest bloom score friction as label
                # but track all frictions in the combo field
                for r in results.rankings[:1]:  # take top result from combo
                    record = {
                        "grid_size": grid_size,
                        "friction": combo_name,
                        "friction_count": len(friction_combo),
                        "outcome": r.outcome,
                        "bloom_score": round(r.bloom_score, 10),
                        "steps": r.altered.steps,
                        "novelty": round(r.novelty, 10),
                        "coherence": round(r.coherence, 10),
                        "chaos": round(r.chaos, 10),
                        "baseline_path": [list(p) for p in results.baseline.path],
                        "altered_path": [list(p) for p in r.altered.path],
                        "notes": r.notes,
                        "rep": rep,
                        "dataset": "003",
                    }
                    save_run(record)

                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0

                if completed % 50 == 0 or completed == total:
                    print(
                        f"[{completed:>5}/{total}] "
                        f"grid={grid_size:>2} "
                        f"combo={combo_name:<28} "
                        f"outcome={r.outcome:<8} "
                        f"bloom={round(r.bloom_score, 4):<8} "
                        f"eta={int(remaining)}s"
                    )

    elapsed_total = time.time() - start_time
    print("-" * 60)
    print(f"Done. {total} runs completed in {round(elapsed_total, 1)}s")
    print(f"Data saved to {DATA}")


def summarize() -> None:
    if not DATA.exists():
        print("No data file found.")
        return

    import collections
    runs = []
    with open(DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))

    ds003 = [r for r in runs if r.get("dataset") == "003"]
    if not ds003:
        print("No Dataset 003 runs found.")
        return

    total = len(ds003)
    print(f"\nDataset 003 Summary — {total} runs")
    print("=" * 60)

    outcomes = collections.Counter(r["outcome"] for r in ds003)
    print("\nOutcome Distribution:")
    for outcome, count in sorted(outcomes.items()):
        pct = round(count / total * 100, 1)
        print(f"  {outcome:<10} {count:>5} runs  ({pct}%)")

    print("\nBy Friction Combination:")
    combo_groups = collections.defaultdict(list)
    for r in ds003:
        combo_groups[r["friction"]].append(r)

    for combo, group in sorted(combo_groups.items()):
        bloom_runs = [r for r in group if r["outcome"] == "bloom"]
        avg_bloom = round(sum(r["bloom_score"] for r in group) / len(group), 4)
        avg_novelty = round(sum(r["novelty"] for r in group) / len(group), 4)
        bloom_rate = round(len(bloom_runs) / len(group) * 100, 1)
        print(f"\n  {combo}")
        print(f"    Runs: {len(group)}  Bloom Rate: {bloom_rate}%")
        print(f"    Avg Bloom Score: {avg_bloom}  Avg Novelty: {avg_novelty}")

    print("\nBloom Rate by Grid Size (Dataset 003):")
    grid_groups = collections.defaultdict(list)
    for r in ds003:
        grid_groups[r["grid_size"]].append(r)

    for size in sorted(grid_groups.keys()):
        group = grid_groups[size]
        bloom_runs = [r for r in group if r["outcome"] == "bloom"]
        bloom_rate = round(len(bloom_runs) / len(group) * 100, 1)
        avg_score = round(sum(r["bloom_score"] for r in group) / len(group), 4)
        print(f"  {size:>2}x{size:<2}  Bloom Rate: {bloom_rate:>5}%  Avg Score: {avg_score}")

    print("\n" + "=" * 60)

    # Compare to Dataset 002
    ds002 = [r for r in runs if r.get("dataset") == "002"]
    if ds002:
        d002_bloom = round(len([r for r in ds002 if r["outcome"] == "bloom"]) / len(ds002) * 100, 1)
        d003_bloom = round(len([r for r in ds003 if r["outcome"] == "bloom"]) / len(ds003) * 100, 1)
        print(f"\nDataset 002 overall Bloom rate: {d002_bloom}%")
        print(f"Dataset 003 overall Bloom rate: {d003_bloom}%")
        print(f"Delta: {round(d003_bloom - d002_bloom, 1)}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        run_batch()
        print()
        summarize()
