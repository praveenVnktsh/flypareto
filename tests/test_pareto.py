import numpy as np

from flypareto.pareto import Objective, feasible_mask, nondominated_mask


def test_nondominated_mixed_directions():
    values = np.array([
        [0.8, 4.0],
        [0.7, 3.0],
        [0.9, 6.0],
        [0.6, 7.0],
    ])
    result = nondominated_mask(values, [Objective("score"), Objective("cost", False)])
    assert result.tolist() == [True, True, True, False]


def test_survival_gate_rejects_nonfinite_and_low_values():
    values = np.array([[0.8, 0.9], [0.9, 0.2], [np.nan, 1.0]])
    result = feasible_mask(values, ["intelligence", "survival"], {"survival": 0.5})
    assert result.tolist() == [True, False, False]

