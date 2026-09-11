from flypareto.demo import run_demo


def test_demo_writes_nonempty_frontier(tmp_path):
    feasible, frontier = run_demo(tmp_path, seed=4, candidates=24)
    assert feasible > 0
    assert frontier > 0
    assert (tmp_path / "candidates.csv").exists()
    assert (tmp_path / "frontier.csv").exists()

