from __future__ import annotations

from pathlib import Path

from friction_bloom.agent import SolveResult
from friction_bloom.environment import GridWorld, Point
from friction_bloom.experiment import FrictionResult

CELL = 48
PAD = 28


def save_path_visualization(
    env: GridWorld,
    baseline: SolveResult,
    result: FrictionResult,
    output_dir: str | Path = "visuals",
) -> Path:
    """Save a dependency-free SVG comparing baseline and altered paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    path = output_path / f"{result.friction_name}.svg"
    width = env.width * CELL + PAD * 2
    height = env.height * CELL + PAD * 2 + 70

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101418"/>',
        f'<text x="{PAD}" y="24" fill="#e8edf2" font-family="monospace" font-size="16">{result.friction_name} | {result.outcome} | bloom_score={result.bloom_score:.2f}</text>',
        f'<text x="{PAD}" y="48" fill="#9fb3c8" font-family="monospace" font-size="12">baseline dashed / altered solid</text>',
    ]

    top = PAD + 50

    for y in range(env.height):
        for x in range(env.width):
            px = PAD + x * CELL
            py = top + y * CELL
            fill = "#18212a"
            if (x, y) in env.obstacles:
                fill = "#374250"
            svg.append(
                f'<rect x="{px}" y="{py}" width="{CELL}" height="{CELL}" fill="{fill}" stroke="#2a3642" stroke-width="1"/>'
            )

    svg.append(_polyline(baseline.path, top, stroke="#84a9ff", dashed=True))
    svg.append(_polyline(result.altered.path, top, stroke="#f2c14e", dashed=False))
    svg.append(_label(env.start, top, "S", "#8df0b4"))
    svg.append(_label(env.goal, top, "G", "#ff8f8f"))

    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def save_all_visualizations(
    env: GridWorld,
    baseline: SolveResult,
    rankings: list[FrictionResult],
    output_dir: str | Path = "visuals",
) -> list[Path]:
    return [save_path_visualization(env, baseline, result, output_dir) for result in rankings]


def _center(point: Point, top: int) -> tuple[float, float]:
    x, y = point
    return PAD + x * CELL + CELL / 2, top + y * CELL + CELL / 2


def _polyline(path: list[Point], top: int, stroke: str, dashed: bool) -> str:
    if not path:
        return ""
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_center(p, top) for p in path))
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" opacity="0.92"{dash}/>'


def _label(point: Point, top: int, text: str, fill: str) -> str:
    x, y = _center(point, top)
    return (
        f'<circle cx="{x}" cy="{y}" r="15" fill="{fill}" opacity="0.95"/>'
        f'<text x="{x}" y="{y + 5}" text-anchor="middle" fill="#101418" '
        f'font-family="monospace" font-size="16" font-weight="bold">{text}</text>'
    )
