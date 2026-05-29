from __future__ import annotations

import math
import re
from collections import Counter

from prompt_chain.agent import ChainResult


def _word_set(steps: list[str]) -> set[str]:
    text = " ".join(steps).lower()
    return set(re.findall(r"\b\w+\b", text))


def chain_entropy(steps: list[str]) -> float:
    """Word-level Shannon entropy of a reasoning chain."""
    text = " ".join(steps).lower()
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def chain_jaccard_distance(a: list[str], b: list[str]) -> float:
    """Jaccard distance on word sets — mirrors jaccard_distance() in metrics.py."""
    set_a = _word_set(a)
    set_b = _word_set(b)
    if not set_a and not set_b:
        return 0.0
    return 1.0 - len(set_a & set_b) / len(set_a | set_b)


def score_chain_bloom(
    baseline: ChainResult, altered: ChainResult
) -> tuple[float, float, float, float, str, str]:
    if not altered.correct:
        return (
            -10.0, 0.0, 0.0, 10.0,
            "collapse",
            "Friction caused reasoning failure — wrong answer produced.",
        )

    novelty = chain_jaccard_distance(baseline.steps, altered.steps)

    extra_steps = len(altered.steps) - len(baseline.steps)
    coherence = 2.0 if extra_steps <= 0 else max(0.0, 2.0 - extra_steps * 0.25)

    entropy_delta = chain_entropy(altered.steps) - chain_entropy(baseline.steps)
    chaos = max(0.0, entropy_delta * 0.5)

    bloom_score = (novelty * 3.0) + coherence - chaos

    if novelty > 0.35 and coherence > 0.5:
        outcome = "bloom"
        notes = "LLM recovered through a novel reasoning route."
    elif novelty > 0.1:
        outcome = "noise"
        notes = "Reasoning changed, but adaptation was weak or inefficient."
    else:
        outcome = "stable"
        notes = "Friction did not meaningfully alter the reasoning chain."

    return bloom_score, novelty, coherence, chaos, outcome, notes
