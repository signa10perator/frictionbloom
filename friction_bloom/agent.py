from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from friction_bloom.environment import GridWorld, Point


@dataclass
class SolveResult:
    path: list[Point]
    success: bool
    steps: int


class GreedyAgent:
    """
    A simple pathfinding agent.

    It uses breadth-first search for reliability, while neighbor ordering favors
    movement toward the goal. This makes the path deterministic but still easy
    to disrupt with friction.
    """

    def solve(self, env: GridWorld) -> SolveResult:
        frontier: deque[Point] = deque([env.start])
        came_from: dict[Point, Point | None] = {env.start: None}

        while frontier:
            current = frontier.popleft()

            if current == env.goal:
                break

            for next_point in self._ordered_neighbors(env, current):
                if next_point not in came_from:
                    frontier.append(next_point)
                    came_from[next_point] = current

        if env.goal not in came_from:
            return SolveResult(path=[], success=False, steps=0)

        path = self._reconstruct_path(came_from, env.start, env.goal)
        return SolveResult(path=path, success=True, steps=max(0, len(path) - 1))

    def _ordered_neighbors(self, env: GridWorld, point: Point) -> list[Point]:
        gx, gy = env.goal
        return sorted(
            env.neighbors(point),
            key=lambda p: abs(p[0] - gx) + abs(p[1] - gy),
        )

    @staticmethod
    def _reconstruct_path(
        came_from: dict[Point, Point | None], start: Point, goal: Point
    ) -> list[Point]:
        current: Point | None = goal
        path: list[Point] = []

        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()
        return path
