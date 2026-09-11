"""Survival-gated multiobjective analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Objective:
    name: str
    maximize: bool = True


def feasible_mask(
    values: np.ndarray,
    columns: Sequence[str],
    floors: dict[str, float],
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError("values and columns have incompatible shapes")
    result = np.all(np.isfinite(values), axis=1)
    lookup = {name: index for index, name in enumerate(columns)}
    for name, floor in floors.items():
        if name not in lookup:
            raise KeyError(f"unknown survival constraint: {name}")
        result &= values[:, lookup[name]] >= floor
    return result


def nondominated_mask(values: np.ndarray, objectives: Iterable[Objective]) -> np.ndarray:
    """Return points not Pareto-dominated; non-finite rows are excluded."""
    values = np.asarray(values, dtype=float)
    objectives = tuple(objectives)
    if values.ndim != 2 or values.shape[1] != len(objectives):
        raise ValueError("one objective is required for each value column")
    signed = values.copy()
    for index, objective in enumerate(objectives):
        if not objective.maximize:
            signed[:, index] *= -1
    finite = np.all(np.isfinite(signed), axis=1)
    result = finite.copy()
    valid_indices = np.flatnonzero(finite)
    for i in valid_indices:
        others = signed[valid_indices]
        dominates = np.all(others >= signed[i], axis=1) & np.any(others > signed[i], axis=1)
        if np.any(dominates):
            result[i] = False
    return result

