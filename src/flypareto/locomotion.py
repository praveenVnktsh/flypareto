"""Embodied locomotion-viability Pareto experiments.

This is the first survival-gated assay in FlyPareto. It measures locomotor
viability and resource tradeoffs; it does not yet measure general intelligence
or lifetime reproductive fitness.
"""

from __future__ import annotations

import csv
import itertools
from collections.abc import Callable
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np

from .flygym_bridge import ClosedLoopSmokeResult, run_closed_loop_smoke
from .graph import Connectome
from .pareto import Objective, nondominated_mask

EmbodiedRunner = Callable[..., ClosedLoopSmokeResult]


def _mean_and_sd(
    row: dict[str, float | bool],
    name: str,
    values: list[float],
) -> None:
    row[name] = fmean(values)
    row[f"{name}_sd"] = pstdev(values)


def run_locomotion_sweep(
    graph: Connectome,
    annotations_path: Path,
    output: Path,
    synaptic_scales: list[float],
    decoder_half_saturations: list[float],
    proprioceptive_speed_scales: list[float],
    seeds: list[int],
    neural_steps: int = 100,
    modality: str = "olfactory",
    stimulus_steps: int = 3,
    runner: EmbodiedRunner | None = None,
) -> tuple[int, int]:
    """Run, survival-gate, and Pareto-rank full-connectome locomotion trials.

    A parameter point is feasible only if every replicate remains finite, keeps
    the thorax at least 0.35 mm above the substrate, and stays within 60 degrees
    of its initial orientation for at least 90% of sampled neural steps.
    """
    grids = (
        synaptic_scales,
        decoder_half_saturations,
        proprioceptive_speed_scales,
        seeds,
    )
    if any(not values for values in grids):
        raise ValueError("parameter grids and seeds must be non-empty")
    if neural_steps <= 0:
        raise ValueError("neural_steps must be positive")
    if any(value <= 0 for values in grids[:3] for value in values):
        raise ValueError("all swept continuous parameters must be positive")
    execute = run_closed_loop_smoke if runner is None else runner

    rows: list[dict[str, float | bool]] = []
    for synaptic_scale, half_saturation, speed_scale in itertools.product(*grids[:3]):
        trials = [
            execute(
                graph,
                annotations_path,
                neural_steps=neural_steps,
                modality=modality,
                stimulus_steps=stimulus_steps,
                synaptic_scale=synaptic_scale,
                decoder_half_saturation=half_saturation,
                proprioceptive_speed_scale=speed_scale,
                seed=seed,
            )
            for seed in seeds
        ]
        row: dict[str, float | bool] = {
            "synaptic_scale": synaptic_scale,
            "decoder_half_saturation": half_saturation,
            "proprioceptive_speed_scale": speed_scale,
            "replicates": float(len(trials)),
        }
        metrics = {
            "forward_displacement_mm": [trial.displacement_mm[0] for trial in trials],
            "lateral_deviation_mm": [abs(trial.displacement_mm[1]) for trial in trials],
            "upright_fraction": [trial.upright_fraction for trial in trials],
            "min_thorax_height_mm": [trial.min_thorax_height_mm for trial in trials],
            "neural_energy_proxy": [trial.neural_energy_proxy for trial in trials],
            "mechanical_effort_proxy": [trial.mechanical_effort_proxy for trial in trials],
            "descending_spikes": [float(trial.descending_spikes) for trial in trials],
        }
        for name, values in metrics.items():
            _mean_and_sd(row, name, values)
        row["min_thorax_height_mm_worst"] = min(metrics["min_thorax_height_mm"])
        row["upright_fraction_worst"] = min(metrics["upright_fraction"])
        row["viable_fraction"] = fmean(float(trial.locomotion_viable) for trial in trials)
        row["feasible"] = bool(all(trial.locomotion_viable for trial in trials))
        rows.append(row)

    objective_names = (
        "forward_displacement_mm",
        "upright_fraction",
        "lateral_deviation_mm",
        "neural_energy_proxy",
        "mechanical_effort_proxy",
    )
    feasible_indices = [index for index, row in enumerate(rows) if row["feasible"]]
    frontier = np.zeros(len(rows), dtype=bool)
    if feasible_indices:
        values = np.asarray(
            [[float(rows[index][name]) for name in objective_names] for index in feasible_indices]
        )
        feasible_frontier = nondominated_mask(
            values,
            (
                Objective("forward_displacement_mm", True),
                Objective("upright_fraction", True),
                Objective("lateral_deviation_mm", False),
                Objective("neural_energy_proxy", False),
                Objective("mechanical_effort_proxy", False),
            ),
        )
        frontier[np.asarray(feasible_indices)[feasible_frontier]] = True

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = [*rows[0], "frontier"]
    with (output / "locomotion-points.csv").open("w", newline="") as handle:
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
