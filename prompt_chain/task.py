from __future__ import annotations

import random
from dataclasses import dataclass, field


_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"]
_ITEMS = ["apples", "books", "coins", "points", "tokens", "marbles", "stars", "cards"]


@dataclass
class ReasoningTask:
    problem: str
    answer: str
    difficulty: int  # 1–11, analogous to grid_size
    task_type: str = "arithmetic"
    system_notes: list[str] = field(default_factory=list)


def generate_task(difficulty: int, seed: int) -> ReasoningTask:
    """Generate a deterministic multi-step arithmetic word problem."""
    rng = random.Random(seed * 100 + difficulty)
    name = _NAMES[seed % len(_NAMES)]
    item = _ITEMS[(seed // len(_NAMES)) % len(_ITEMS)]

    value = rng.randint(20, 50)
    sentences = [f"{name} starts with {value} {item}"]

    for _ in range(difficulty):
        op = rng.choice(["add", "subtract", "double"])
        if op == "add":
            amount = rng.randint(3, 15)
            value += amount
            sentences.append(f"{name} receives {amount} more {item}")
        elif op == "subtract":
            max_sub = min(value - 1, 10)
            if max_sub < 1:
                amount = rng.randint(3, 15)
                value += amount
                sentences.append(f"{name} receives {amount} more {item}")
            else:
                amount = rng.randint(1, max_sub)
                value -= amount
                sentences.append(f"{name} gives away {amount} {item}")
        else:
            value *= 2
            sentences.append(f"{name}'s {item} count doubles")

    sentences.append(f"How many {item} does {name} have in total?")
    problem = ". ".join(sentences[:-1]) + ". " + sentences[-1]
    return ReasoningTask(problem=problem, answer=str(value), difficulty=difficulty)
