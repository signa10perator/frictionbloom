from __future__ import annotations

import argparse

from friction_bloom.agent import GreedyAgent
from friction_bloom.environment import GridWorld
from friction_bloom.experiment import run_experiment
from friction_bloom.friction import (
    block_baseline_midpoint,
    block_first_third,
    add_wall_gap_constraint,
    add_scattered_obstacles,
)
from friction_bloom.visualize import save_all_visualizations


def build_environment() -> GridWorld:
    return GridWorld(
        width=10,
        height=8,
        start=(0, 0),
        goal=(9, 7),
        obstacles={
            (3, 0), (3, 1), (3, 2),
            (6, 5), (6, 6), (6, 7),
            (1, 5), (2, 5), (3, 5),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Friction Bloom grid-world experiment."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save baseline-vs-friction path visualizations as PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        default="visuals",
        help="Directory for generated PNG files. Default: visuals",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = build_environment()
    agent = GreedyAgent()

    frictions = [
        block_baseline_midpoint,
        block_first_third,
        add_wall_gap_constraint,
        add_scattered_obstacles,
    ]

    results = run_experiment(agent, env, frictions)

    print("\n=== BASELINE MAP ===")
    print(env.render(results.baseline.path))
    print(f"baseline_steps: {results.baseline.steps}")
    print(f"baseline_success: {results.baseline.success}")

    print("\n=== FRICTION BLOOM RESULTS ===\n")
    for idx, result in enumerate(results.rankings, start=1):
        print(f"{idx}. {result.friction_name}")
        print(f"   outcome: {result.outcome}")
        print(f"   bloom_score: {result.bloom_score:.2f}")
        print(f"   altered_steps: {result.altered.steps}")
        print(f"   novelty: {result.novelty:.2f}")
        print(f"   coherence: {result.coherence:.2f}")
        print(f"   chaos: {result.chaos:.2f}")
        print(f"   notes: {result.notes}")
        print()

    if args.visualize:
        paths = save_all_visualizations(
            env=env,
            baseline=results.baseline,
            rankings=results.rankings,
            output_dir=args.output_dir,
        )
        print("=== VISUALIZATIONS SAVED ===")
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
