from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set, Tuple

Point = Tuple[int, int]


@dataclass
class GridWorld:
    width: int
    height: int
    start: Point
    goal: Point
    obstacles: Set[Point] = field(default_factory=set)

    def copy(self) -> "GridWorld":
        return GridWorld(
            width=self.width,
            height=self.height,
            start=self.start,
            goal=self.goal,
            obstacles=set(self.obstacles),
        )

    def in_bounds(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, point: Point) -> bool:
        return point not in self.obstacles

    def neighbors(self, point: Point) -> list[Point]:
        x, y = point
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        return [p for p in candidates if self.in_bounds(p) and self.passable(p)]

    def block(self, points: Iterable[Point]) -> None:
        for point in points:
            if point not in {self.start, self.goal} and self.in_bounds(point):
                self.obstacles.add(point)

    def render(self, path: list[Point] | None = None) -> str:
        path_set = set(path or [])
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                p = (x, y)
                if p == self.start:
                    row.append("S")
                elif p == self.goal:
                    row.append("G")
                elif p in self.obstacles:
                    row.append("#")
                elif p in path_set:
                    row.append("·")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)
