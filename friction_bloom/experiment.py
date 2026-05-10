from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from friction_bloom.agent import GreedyAgent, SolveResult
from friction_bloom.environment import GridWorld
from friction_bloom.metrics import score_bloom

FrictionFn = Callable[[GridWorld, SolveResult], GridWorld]


@dataclass
class FrictionResult:
    friction_name: str
    altered: SolveResult
    bloom_score: float
    novelty: float
    coherence: float
    chaos: float
    outcome: str
    notes: str


@dataclass
class ExperimentResults:
    baseline: SolveResult
    rankings: list[FrictionResult]


def run_experiment(
    agent: GreedyAgent,
    env: GridWorld,
    frictions: list[FrictionFn],
) -> ExperimentResults:
    baseline = agent.solve(env)
    rankings: list[FrictionResult] = []

    for friction in frictions:
        modified_env = friction(env, baseline)
        altered = agent.solve(modified_env)
        bloom_score, novelty, coherence, chaos, outcome, notes = score_bloom(
            baseline, altered
        )

        rankings.append(
            FrictionResult(
                friction_name=friction.__name__,
                altered=altered,
                bloom_score=bloom_score,
                novelty=novelty,
                coherence=coherence,
                chaos=chaos,
                outcome=outcome,
                notes=notes,
            )
        )

    rankings.sort(key=lambda item: item.bloom_score, reverse=True)
    return ExperimentResults(baseline=baseline, rankings=rankings)
