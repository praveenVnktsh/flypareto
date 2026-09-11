"""Deterministic end-to-end smoke experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .graph import Connectome
from .pareto import Objective, feasible_mask, nondominated_mask
from .sim import EventDrivenLIF, LIFConfig

COLUMNS = ("intelligence", "survival", "robustness", "energy", "fly_distance")


def synthetic_graph(seed: int = 7, neurons: int = 512, degree: int = 12) -> Connectome:
    rng = np.random.default_rng(seed)
    counts = np.full(neurons, degree, dtype=np.int64)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    targets = rng.integers(0, neurons, size=neurons * degree, dtype=np.int32)
    weights = rng.integers(1, 16, size=neurons * degree).astype(np.float32)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=neurons, p=[0.3, 0.7])
    return Connectome(np.arange(neurons, dtype=np.int64), indptr, targets, weights, signs)


def run_demo(output: Path, seed: int = 7, candidates: int = 48) -> tuple[int, int]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    graph = synthetic_graph(seed)
    rows = []
    for candidate in range(candidates):
        gain = float(rng.uniform(0.002, 0.03))
        stimulus = float(rng.uniform(0.7, 1.4))
        external = np.zeros((60, graph.neuron_count), dtype=np.float32)
        external[:4, :24] = stimulus
        result = EventDrivenLIF(graph, LIFConfig(synaptic_scale=gain)).run(60, external)
        activity = float(result.population_spikes.sum())
        propagation = float(np.count_nonzero(result.spike_counts)) / graph.neuron_count
        intelligence = 0.55 * propagation + 0.45 * (1.0 - abs(gain - 0.015) / 0.015)
        survival = float(np.exp(-((activity / 850.0) - 1.0) ** 2))
        robustness = float(max(0.0, 1.0 - abs(propagation - 0.45)))
        energy = activity
        fly_distance = float(abs(propagation - 0.35))
        rows.append((candidate, gain, stimulus, intelligence, survival, robustness, energy,
                     fly_distance))

    matrix = np.asarray([row[3:] for row in rows], dtype=float)
    feasible = feasible_mask(matrix, COLUMNS, {"survival": 0.35})
    objectives = (
        Objective("intelligence", True),
        Objective("survival", True),
        Objective("robustness", True),
        Objective("energy", False),
        Objective("fly_distance", False),
    )
    frontier = np.zeros(candidates, dtype=bool)
    frontier_indices = np.flatnonzero(feasible)
    frontier[frontier_indices] = nondominated_mask(matrix[feasible], objectives)

    header = ("candidate", "synaptic_scale", "stimulus", *COLUMNS, "feasible", "frontier")
    with (output / "candidates.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, row in enumerate(rows):
            writer.writerow((*row, bool(feasible[index]), bool(frontier[index])))
    with (output / "frontier.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in np.flatnonzero(frontier):
            writer.writerow((*rows[index], True, True))
    return int(np.count_nonzero(feasible)), int(np.count_nonzero(frontier))

