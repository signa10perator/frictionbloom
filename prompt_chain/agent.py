from __future__ import annotations

import re
from dataclasses import dataclass

import anthropic


@dataclass
class ChainResult:
    steps: list[str]   # each line of the reasoning chain
    answer: str        # extracted final answer
    correct: bool
    tokens_used: int = 0


_BASE_SYSTEM = (
    "You are a careful step-by-step reasoner. "
    "Work through the problem one numbered step at a time. "
    "State the running total after each step. "
    "End your response with exactly: Answer: <integer>"
)


def _extract_steps(text: str) -> list[str]:
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def _extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else ""


class ChainAgent:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def solve(self, problem: str, expected_answer: str, extra_system: str = "") -> ChainResult:
        system = (_BASE_SYSTEM if not extra_system
                  else extra_system.strip() + "\n\n" + _BASE_SYSTEM)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": problem}],
        )

        raw = response.content[0].text
        steps = _extract_steps(raw)
        answer = _extract_answer(raw)
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return ChainResult(
            steps=steps,
            answer=answer,
            correct=(answer.strip() == expected_answer.strip()),
            tokens_used=tokens,
        )
