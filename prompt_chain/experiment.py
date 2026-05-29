from __future__ import annotations

from dataclasses import dataclass

from prompt_chain.agent import ChainAgent, ChainResult
from prompt_chain.friction import FrictionFn
from prompt_chain.metrics import score_chain_bloom
from prompt_chain.task import ReasoningTask


@dataclass
class ChainFrictionResult:
    friction_name: str
    altered: ChainResult
    bloom_score: float
    novelty: float
    coherence: float
    chaos: float
    outcome: str
    notes: str


@dataclass
class ChainExperimentResults:
    baseline: ChainResult
    result: ChainFrictionResult


def run_chain_experiment(
    agent: ChainAgent,
    task: ReasoningTask,
    friction_combo: list[FrictionFn],
) -> ChainExperimentResults:
    """
    Run one friction experiment on a reasoning task.

    Frictions are STACKED — applied sequentially to the same task — so combined
    configs genuinely compose rather than competing for best score (unlike
    Dataset 003's pick-best approach).
    """
    baseline = agent.solve(task.problem, task.answer)

    modified = task
    for fn in friction_combo:
        modified = fn(modified, baseline)

    extra_system = "\n".join(modified.system_notes)
    altered = agent.solve(modified.problem, task.answer, extra_system)

    bloom_score, novelty, coherence, chaos, outcome, notes = score_chain_bloom(
        baseline, altered
    )
    friction_name = "+".join(fn.__name__ for fn in friction_combo)

    return ChainExperimentResults(
        baseline=baseline,
        result=ChainFrictionResult(
            friction_name=friction_name,
            altered=altered,
            bloom_score=bloom_score,
            novelty=novelty,
            coherence=coherence,
            chaos=chaos,
            outcome=outcome,
            notes=notes,
        ),
    )
