from __future__ import annotations

import math
from collections import Counter

from friction_bloom.agent import SolveResult
from friction_bloom.environment import Point


def path_entropy(path: list[Point]) -> float:
    if not path:
        return 0.0

    counts = Counter(path)
    total = len(path)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def jaccard_distance(a: list[Point], b: list[Point]) -> float:
    set_a = set(a)
    set_b = set(b)

    if not set_a and not set_b:
        return 0.0

    return 1.0 - (len(set_a & set_b) / len(set_a | set_b))


def score_bloom(baseline: SolveResult, altered: SolveResult) -> tuple[float, float, float, float, str, str]:
    if not altered.success:
        return -10.0, 0.0, 0.0, 10.0, "collapse", "Friction prevented the agent from reaching the goal."

    novelty = jaccard_distance(baseline.path, altered.path)

    extra_steps = altered.steps - baseline.steps
    if extra_steps <= 0:
        coherence = 2.0
    else:
        coherence = max(0.0, 2.0 - (extra_steps * 0.25))

    entropy_delta = path_entropy(altered.path) - path_entropy(baseline.path)
    chaos = max(0.0, entropy_delta * 0.5)

    bloom_score = (novelty * 3.0) + coherence - chaos

    if novelty > 0.35 and coherence > 0.5:
        outcome = "bloom"
        notes = "Agent recovered through a novel viable route."
    elif novelty > 0.1:
        outcome = "noise"
        notes = "Behavior changed, but adaptation was weak or inefficient."
    else:
        outcome = "stable"
        notes = "Friction did not meaningfully alter behavior."

    return bloom_score, novelty, coherence, chaos, outcome, notes
