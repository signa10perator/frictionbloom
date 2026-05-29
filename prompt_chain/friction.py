"""
Prompt-chain friction functions — direct analogs to friction_bloom/friction.py.

  inject_midpoint_block    ↔  block_baseline_midpoint
  inject_first_third_block ↔  block_first_third
  inject_wall_with_gap     ↔  add_wall_gap_constraint
  inject_scattered_context ↔  add_scattered_obstacles

Each function takes the original task and the baseline chain result, and returns
a new task (immutable — uses dataclasses.replace) with friction injected into
the problem text or the system_notes list.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from prompt_chain.agent import ChainResult
from prompt_chain.task import ReasoningTask

FrictionFn = Callable[[ReasoningTask, ChainResult], ReasoningTask]


def inject_midpoint_block(task: ReasoningTask, baseline: ChainResult) -> ReasoningTask:
    """Inject a wrong running-total checkpoint at the sentence midpoint."""
    sentences = task.problem.rstrip("?").split(". ")
    mid = max(1, len(sentences) // 2)
    try:
        wrong_val = int(task.answer) - task.difficulty * 3
    except ValueError:
        wrong_val = 0
    insert = f"(Checkpoint: your running total at this point is {wrong_val})"
    sentences.insert(mid, insert)
    return replace(task, problem=". ".join(sentences))


def inject_first_third_block(task: ReasoningTask, baseline: ChainResult) -> ReasoningTask:
    """Inject a wrong running-total note after the first-third sentence."""
    sentences = task.problem.rstrip("?").split(". ")
    idx = max(1, len(sentences) // 3)
    try:
        wrong_val = int(task.answer) + task.difficulty * 2
    except ValueError:
        wrong_val = 0
    insert = f"(Note: after this step your total should be {wrong_val})"
    sentences.insert(idx, insert)
    return replace(task, problem=". ".join(sentences))


def inject_wall_with_gap(task: ReasoningTask, baseline: ChainResult) -> ReasoningTask:
    """Force reasoning through a single allowed path — no doubling permitted.

    The 'gap in the wall' is that repeated addition still reaches the correct
    answer; the LLM must find that route rather than its default shortcut.
    """
    note = (
        "CONSTRAINT: You may only use addition and subtraction — "
        "no multiplication or doubling. "
        "If a step says 'doubles', express it as adding the current value to itself."
    )
    return replace(task, system_notes=task.system_notes + [note])


def inject_scattered_context(task: ReasoningTask, baseline: ChainResult) -> ReasoningTask:
    """Scatter irrelevant numeric facts throughout the problem sentences."""
    distractors = [
        "(unrelated: 72 units were reported yesterday)",
        "(aside: the reference value is 365)",
        "(note: total capacity is 1024)",
    ]
    sentences = task.problem.split(". ")
    result: list[str] = []
    d_idx = 0
    for i, sentence in enumerate(sentences):
        result.append(sentence)
        if d_idx < len(distractors) and 0 < i < len(sentences) - 1:
            result.append(distractors[d_idx])
            d_idx += 1
    return replace(task, problem=". ".join(result))
