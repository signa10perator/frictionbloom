from friction_bloom.agent import GreedyAgent
from friction_bloom.environment import GridWorld
from friction_bloom.experiment import run_experiment
from friction_bloom.friction import block_baseline_midpoint


def test_experiment_runs():
    env = GridWorld(width=5, height=5, start=(0, 0), goal=(4, 4))
    agent = GreedyAgent()
    results = run_experiment(agent, env, [block_baseline_midpoint])

    assert results.baseline.success is True
    assert len(results.rankings) == 1
