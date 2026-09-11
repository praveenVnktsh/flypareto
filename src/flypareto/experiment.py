"""Full-connectome operating-point experiments.

These experiments calibrate neural dynamics before embodiment. They are deliberately
not labelled as intelligence or survival measurements.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np
from pyarrow import feather

from .data import first_column
from .graph import BODY_ALIASES, Connectome
from .pareto import Objective, nondominated_mask
from .sim import EventDrivenLIF, LIFConfig

SENSORY_SUPERCLASSES = {
    "ol_sensory",
    "cb_sensory",
    "vnc_sensory",
    "sensory_ascending",
}
OUTPUT_SUPERCLASSES = {
    "descending_neuron",
    "cb_motor",
    "vnc_motor",
    "vnc_efferent",
}


@dataclass(frozen=True)
class OperatingPoint:
    synaptic_scale: float
    threshold: float


def annotated_indices(
    graph: Connectome,
    annotations_path: Path,
    superclasses: set[str],
) -> np.ndarray:
    """Resolve traced graph indices for a set of MaleCNS superclasses."""
    table = feather.read_table(annotations_path, memory_map=True)
    body_col = first_column(table.column_names, BODY_ALIASES)
    superclass_col = first_column(table.column_names, ("superclass",))
    if body_col is None or superclass_col is None:
        raise ValueError("annotations require body ID and superclass columns")
    body_ids = np.asarray(table[body_col].combine_chunks()).astype(np.int64, copy=False)
    labels = table[superclass_col].combine_chunks().to_pylist()
    selected_ids = body_ids[np.fromiter((label in superclasses for label in labels), bool)]
    positions = np.searchsorted(graph.node_ids, selected_ids)
    valid = positions < graph.neuron_count
    valid[valid] &= graph.node_ids[positions[valid]] == selected_ids[valid]
    return np.unique(positions[valid].astype(np.int32, copy=False))


def _single_probe(
    graph: Connectome,
    sensory_indices: np.ndarray,
    output_indices: np.ndarray,
    point: OperatingPoint,
    seed: int,
    steps: int,
    stimulus_count: int,
    stimulus_amplitude: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        sensory_indices,
        size=min(stimulus_count, sensory_indices.size),
        replace=False,
    )
    external = np.zeros((steps, graph.neuron_count), dtype=np.float32)
    external[:3, chosen] = stimulus_amplitude
    result = EventDrivenLIF(
        graph,
        LIFConfig(synaptic_scale=point.synaptic_scale, threshold=point.threshold),
    ).run(steps, external)
    total_spikes = int(result.population_spikes.sum())
    active = result.spike_counts > 0
    output_active = result.spike_counts[output_indices] > 0
    late_start = max(3, steps // 2)
    late_spikes = int(result.population_spikes[late_start:].sum())
    probabilities = result.spike_counts[active].astype(float)
    if probabilities.size:
        probabilities /= probabilities.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        effective_population = float(np.exp(entropy))
    else:
        effective_population = 0.0
    return {
        "total_spikes": float(total_spikes),
        "active_fraction": float(np.mean(active)),
        "output_recruitment": float(np.mean(output_active)),
        "peak_fraction": float(result.population_spikes.max() / graph.neuron_count),
        "late_fraction": float(late_spikes / max(total_spikes, 1)),
        "effective_population": effective_population,
    }


def run_operating_sweep(
    graph: Connectome,
    annotations_path: Path,
    output: Path,
    scales: list[float],
    thresholds: list[float],
    seeds: list[int],
    steps: int = 100,
    stimulus_count: int = 100,
    stimulus_amplitude: float = 1.2,
) -> tuple[int, int]:
    """Measure and Pareto-rank full-connectome response operating points."""
    sensory = annotated_indices(graph, annotations_path, SENSORY_SUPERCLASSES)
    outputs = annotated_indices(graph, annotations_path, OUTPUT_SUPERCLASSES)
    if not sensory.size or not outputs.size:
        raise ValueError("could not resolve sensory and output neurons")
    rows: list[dict[str, float]] = []
    metrics = (
        "total_spikes",
        "active_fraction",
        "output_recruitment",
        "peak_fraction",
        "late_fraction",
        "effective_population",
    )
    for scale, threshold in itertools.product(scales, thresholds):
        point = OperatingPoint(scale, threshold)
        trials = [
            _single_probe(
                graph,
                sensory,
                outputs,
                point,
                seed,
                steps,
                stimulus_count,
                stimulus_amplitude,
            )
            for seed in seeds
        ]
        row: dict[str, float] = {"synaptic_scale": scale, "threshold": threshold}
        for metric in metrics:
            values = [trial[metric] for trial in trials]
            row[metric] = fmean(values)
            row[f"{metric}_sd"] = pstdev(values)
        rows.append(row)

    objective_names = (
        "output_recruitment",
        "effective_population",
        "total_spikes",
        "peak_fraction",
    )
    objective_values = np.asarray(
        [[row[name] for name in objective_names] for row in rows], dtype=float
    )
    frontier = nondominated_mask(
        objective_values,
        (
            Objective("output_recruitment", True),
            Objective("effective_population", True),
            Objective("total_spikes", False),
            Objective("peak_fraction", False),
        ),
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) + ["frontier"]
    with (output / "operating-points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, is_frontier in zip(rows, frontier):
            writer.writerow({**row, "frontier": bool(is_frontier)})
    with (output / "frontier.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, is_frontier in zip(rows, frontier):
            if is_frontier:
                writer.writerow({**row, "frontier": True})
    return len(rows), int(np.count_nonzero(frontier))

