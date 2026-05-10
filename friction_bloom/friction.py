from __future__ import annotations

from friction_bloom.agent import SolveResult
from friction_bloom.environment import GridWorld, Point


def block_baseline_midpoint(env: GridWorld, baseline: SolveResult) -> GridWorld:
    modified = env.copy()
    if baseline.path:
        midpoint = baseline.path[len(baseline.path) // 2]
        modified.block([midpoint])
    return modified


def block_first_third(env: GridWorld, baseline: SolveResult) -> GridWorld:
    modified = env.copy()
    if baseline.path:
        index = max(1, len(baseline.path) // 3)
        modified.block([baseline.path[index]])
    return modified


def add_wall_gap_constraint(env: GridWorld, baseline: SolveResult) -> GridWorld:
    modified = env.copy()
    wall_x = env.width // 2
    gap_y = env.height // 2
    wall: list[Point] = [(wall_x, y) for y in range(env.height) if y != gap_y]
    modified.block(wall)
    return modified


def add_scattered_obstacles(env: GridWorld, baseline: SolveResult) -> GridWorld:
    modified = env.copy()
    points = {
        (2, 1),
        (4, 2),
        (5, 3),
        (7, 4),
        (8, 5),
    }
    modified.block(points)
    return modified
