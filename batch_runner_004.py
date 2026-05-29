"""
Friction Bloom — Batch Runner
Dataset 004: Prompt-Chain Simulator

Replaces the grid world with an LLM reasoning chain as the environment.
Friction is injected into the task — wrong checkpoints, method walls, distractors —
and we measure whether the model finds a novel but still-correct route (bloom).

Structure
---------
  11 difficulties  ×  11 friction configs  ×  66 reps  ≈  7,986 runs

Cost estimate (Claude Haiku 4.5, ~600 tokens/call):
  8,712 API calls  ≈  ~$15  at list pricing
  Wall time: ~2–3 hours sequential; ~20 min with 8-worker async pool

Friction configs
----------------
  Singles  : midpoint_block, first_third_block, wall_with_gap, scattered_context
  Pairs    : all 6 pairwise combos
  All-four : stacked simultaneously

Key upgrade over Dataset 003
-----------------------------
  Frictions are TRUE stacked composition — each function in a combo is applied
  sequentially to the same task before solving, rather than run independently
  and picking the best result.

Requires ANTHROPIC_API_KEY in environment.
Run: python batch_runner_004.py
     python batch_runner_004.py summary
     python batch_runner_004.py dry-run   (validate without API calls)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from prompt_chain.agent import ChainAgent
from prompt_chain.experiment import run_chain_experiment
from prompt_chain.friction import (
    inject_first_third_block,
    inject_midpoint_block,
    inject_scattered_context,
    inject_wall_with_gap,
)
from prompt_chain.metrics import score_chain_bloom
from prompt_chain.task import ReasoningTask, generate_task

# ── Configuration ─────────────────────────────────────────────────────────────

DIFFICULTIES = list(range(1, 12))  # 1–11 operations (analogous to grid sizes 5–15)

FRICTION_CONFIGS = [
    # Singles
    [inject_midpoint_block],
    [inject_first_third_block],
    [inject_wall_with_gap],
    [inject_scattered_context],
    # Pairs
    [inject_midpoint_block, inject_first_third_block],
    [inject_midpoint_block, inject_wall_with_gap],
    [inject_midpoint_block, inject_scattered_context],
    [inject_first_third_block, inject_wall_with_gap],
    [inject_first_third_block, inject_scattered_context],
    [inject_wall_with_gap, inject_scattered_context],
    # All four
    [inject_midpoint_block, inject_first_third_block, inject_wall_with_gap, inject_scattered_context],
]

CONFIG_NAMES = [
    "midpoint_block",
    "first_third_block",
    "wall_with_gap",
    "scattered_context",
    "midpoint_block+first_third_block",
    "midpoint_block+wall_with_gap",
    "midpoint_block+scattered_context",
    "first_third_block+wall_with_gap",
    "first_third_block+scattered_context",
    "wall_with_gap+scattered_context",
    "all_four",
]

REPS_PER_CONFIG = 66          # 11 × 11 × 66 = 7,986 ≈ 8,000 records
MODEL = "claude-haiku-4-5-20251001"
DATA = Path("data/runs.jsonl")

# ── I/O ───────────────────────────────────────────────────────────────────────

def save_run(record: dict) -> None:
    DATA.parent.mkdir(exist_ok=True)
    with open(DATA, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_batch(dry_run: bool = False) -> None:
    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    total_records = len(DIFFICULTIES) * len(FRICTION_CONFIGS) * REPS_PER_CONFIG
    # Baseline is reused across all configs per (difficulty, rep)
    total_api_calls = len(DIFFICULTIES) * REPS_PER_CONFIG * (1 + len(FRICTION_CONFIGS))

    print(f"\nFriction Bloom Batch Runner — Dataset 004: Prompt-Chain Simulator")
    print(f"Model          : {MODEL}")
    print(f"Difficulties   : {DIFFICULTIES[0]}–{DIFFICULTIES[-1]} operations")
    print(f"Friction configs: {len(FRICTION_CONFIGS)} (4 singles + 6 pairs + 1 all-four)")
    print(f"Reps per config : {REPS_PER_CONFIG}")
    print(f"Total records  : {total_records}")
    print(f"Est. API calls : {total_api_calls:,}  (baseline reused per rep)")
    print(f"Est. cost      : ~$15  (Haiku list pricing)")
    print(f"Output         : {DATA} (appending — prior datasets preserved)")
    if dry_run:
        print(f"\n[DRY RUN] Validating task generation and friction stack only.")
    print("-" * 64)

    agent = ChainAgent(model=MODEL) if not dry_run else None

    completed = 0
    start_time = time.time()

    for difficulty in DIFFICULTIES:
        for rep in range(REPS_PER_CONFIG):
            task = generate_task(difficulty, seed=rep)

            if dry_run:
                # Validate task generation and friction application without API
                for fns in FRICTION_CONFIGS:
                    modified = task
                    for fn in fns:
                        from prompt_chain.agent import ChainResult
                        dummy_baseline = ChainResult(steps=["step 1", "step 2"], answer=task.answer, correct=True)
                        modified = fn(modified, dummy_baseline)
                completed += len(FRICTION_CONFIGS)
                continue

            # Solve baseline once, reuse across all friction configs for this rep
            baseline = agent.solve(task.problem, task.answer)

            for config_idx, friction_fns in enumerate(FRICTION_CONFIGS):
                modified = task
                for fn in friction_fns:
                    modified = fn(modified, baseline)

                extra_system = "\n".join(modified.system_notes)
                altered = agent.solve(modified.problem, task.answer, extra_system)

                bloom_score, novelty, coherence, chaos, outcome, notes = score_chain_bloom(
                    baseline, altered
                )

                record = {
                    "difficulty": difficulty,
                    "friction": CONFIG_NAMES[config_idx],
                    "friction_count": len(friction_fns),
                    "outcome": outcome,
                    "bloom_score": round(bloom_score, 10),
                    "steps_baseline": len(baseline.steps),
                    "steps_altered": len(altered.steps),
                    "novelty": round(novelty, 10),
                    "coherence": round(coherence, 10),
                    "chaos": round(chaos, 10),
                    "baseline_steps": baseline.steps,
                    "altered_steps": altered.steps,
                    "baseline_answer": baseline.answer,
                    "altered_answer": altered.answer,
                    "correct": altered.correct,
                    "notes": notes,
                    "rep": rep,
                    "dataset": "004",
                    "task_type": task.task_type,
                    "tokens_used": baseline.tokens_used + altered.tokens_used,
                }
                save_run(record)
                completed += 1

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_records - completed) / rate if rate > 0 else 0

                if completed % 100 == 0 or completed == total_records:
                    print(
                        f"[{completed:>5}/{total_records}] "
                        f"diff={difficulty:>2}  "
                        f"config={CONFIG_NAMES[config_idx]:<38} "
                        f"outcome={outcome:<8} "
                        f"bloom={round(bloom_score, 3):<7} "
                        f"eta={int(remaining)}s"
                    )

    if dry_run:
        print(f"\n[DRY RUN] Task generation and friction stacking validated.")
        print(f"  Tasks generated: {len(DIFFICULTIES) * REPS_PER_CONFIG}")
        print(f"  Friction applications: {len(DIFFICULTIES) * REPS_PER_CONFIG * sum(len(c) for c in FRICTION_CONFIGS)}")
        return

    elapsed_total = time.time() - start_time
    print("-" * 64)
    print(f"Done. {completed} records in {round(elapsed_total, 1)}s")
    print(f"Data saved to {DATA}")


# ── Summary ───────────────────────────────────────────────────────────────────

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

    ds004 = [r for r in runs if r.get("dataset") == "004"]
    if not ds004:
        print("No Dataset 004 runs found.")
        return

    total = len(ds004)
    print(f"\nDataset 004 Summary — {total} runs")
    print("=" * 64)

    outcomes = collections.Counter(r["outcome"] for r in ds004)
    print("\nOutcome Distribution:")
    for outcome, count in sorted(outcomes.items()):
        pct = round(count / total * 100, 1)
        print(f"  {outcome:<10} {count:>5} runs  ({pct}%)")

    print("\nBy Friction Config:")
    config_groups = collections.defaultdict(list)
    for r in ds004:
        config_groups[r["friction"]].append(r)

    for config, group in sorted(config_groups.items()):
        bloom_runs = [r for r in group if r["outcome"] == "bloom"]
        avg_bloom = round(sum(r["bloom_score"] for r in group) / len(group), 4)
        avg_novelty = round(sum(r["novelty"] for r in group) / len(group), 4)
        correct_rate = round(sum(1 for r in group if r.get("correct")) / len(group) * 100, 1)
        bloom_rate = round(len(bloom_runs) / len(group) * 100, 1)
        print(f"\n  {config}")
        print(f"    Runs: {len(group)}  Correct: {correct_rate}%  Bloom Rate: {bloom_rate}%")
        print(f"    Avg Bloom Score: {avg_bloom}  Avg Novelty: {avg_novelty}")

    print("\nBloom Rate by Difficulty:")
    diff_groups = collections.defaultdict(list)
    for r in ds004:
        diff_groups[r["difficulty"]].append(r)

    for diff in sorted(diff_groups.keys()):
        group = diff_groups[diff]
        bloom_runs = [r for r in group if r["outcome"] == "bloom"]
        bloom_rate = round(len(bloom_runs) / len(group) * 100, 1)
        avg_score = round(sum(r["bloom_score"] for r in group) / len(group), 4)
        correct_rate = round(sum(1 for r in group if r.get("correct")) / len(group) * 100, 1)
        print(f"  diff={diff:>2}  Correct: {correct_rate:>5}%  Bloom: {bloom_rate:>5}%  Avg Score: {avg_score}")

    print("\n" + "=" * 64)

    # Compare to prior datasets
    for ds_id in ("002", "003"):
        prior = [r for r in runs if r.get("dataset") == ds_id]
        if prior:
            prior_bloom = round(len([r for r in prior if r["outcome"] == "bloom"]) / len(prior) * 100, 1)
            ds004_bloom = round(len([r for r in ds004 if r["outcome"] == "bloom"]) / total * 100, 1)
            print(f"\nDataset {ds_id} bloom rate: {prior_bloom}%")
            print(f"Dataset 004 bloom rate: {ds004_bloom}%")
            print(f"Delta: {round(ds004_bloom - prior_bloom, 1)}%")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    elif len(sys.argv) > 1 and sys.argv[1] == "dry-run":
        run_batch(dry_run=True)
    else:
        run_batch()
        print()
        summarize()
