"""Circuit-level calibration sweeps between sensory channels and steering DNs."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from statistics import fmean

import numpy as np

from .graph import Connectome
from .interface import BilateralNeuralInterface, MaleCNSPopulations
from .pareto import Objective, nondominated_mask
from .sim import EventDrivenLIF, LIFConfig


def _write_rows(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError("cannot infer columns from an empty table")
        fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_intervention_calibration(
    graph: Connectome,
    annotations_path: Path,
    output: Path,
    synaptic_scales: list[float],
    thresholds: list[float],
    odor_channels: tuple[str, ...] = ("ORN_DM1", "ORN_VA2"),
    dn_types: tuple[str, ...] = ("DNa01", "DNa02", "DNg13"),
    steps: int = 100,
    stimulus_steps: int = 10,
    stimulus_amplitude: float = 1.2,
    maximum_peak_fraction: float = 0.25,
    maximum_late_fraction: float = 0.75,
) -> tuple[int, int]:
    """Measure and Pareto-rank receptor-to-steering transmission.

    Every candidate runs unilateral interventions for each named receptor channel.
    The matrix retains each DN type separately; aggregate scores are used only for
    candidate ranking. This calibration does not alter or reduce graph topology.
    """
    if not synaptic_scales or not thresholds or not odor_channels or not dn_types:
        raise ValueError("parameter grids, odor channels, and DN types must be non-empty")
    if steps <= 1 or not 0 < stimulus_steps < steps:
        raise ValueError("stimulus_steps must be positive and smaller than steps")
    if stimulus_amplitude <= 0:
        raise ValueError("stimulus_amplitude must be positive")

    populations = MaleCNSPopulations.from_annotations(graph, annotations_path)
    interface = BilateralNeuralInterface(graph, populations)
    missing_channels = [
        channel for channel in odor_channels if channel not in populations.sensory_types
    ]
    if missing_channels:
        raise ValueError(f"unresolved odor channels: {', '.join(missing_channels)}")
    active_dn_types = [
        name
        for name in dn_types
        if name in populations.descending_types
        and (
            populations.descending_types[name].left.size
            + populations.descending_types[name].right.size
        )
    ]
    if not active_dn_types:
        raise ValueError("none of the requested DN types resolve to graph neurons")

    matrix_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for scale, threshold in itertools.product(synaptic_scales, thresholds):
        trial_energy: list[float] = []
        trial_peaks: list[float] = []
        trial_late: list[float] = []
        selectivities: list[float] = []
        response_rates: list[float] = []
        for channel, stimulus_side in itertools.product(odor_channels, ("L", "R")):
            neural = EventDrivenLIF(
                graph,
                LIFConfig(synaptic_scale=scale, threshold=threshold),
            )
            neural.reset()
            external = interface.encode_sensory_types(
                (channel,),
                stimulus_amplitude if stimulus_side == "L" else 0.0,
                stimulus_amplitude if stimulus_side == "R" else 0.0,
                amplitude=1.0,
            )
            population_spikes = np.zeros(steps, dtype=np.int32)
            dn_counts = {
                name: np.zeros(2, dtype=np.int32) for name in active_dn_types
            }
            for step in range(steps):
                spikes = neural.step(external if step < stimulus_steps else None)
                population_spikes[step] = int(np.count_nonzero(spikes))
                for name in active_dn_types:
                    population = populations.descending_types[name]
                    dn_counts[name][0] += int(np.count_nonzero(spikes[population.left]))
                    dn_counts[name][1] += int(np.count_nonzero(spikes[population.right]))

            total_spikes = int(population_spikes.sum())
            peak_fraction = float(population_spikes.max() / graph.neuron_count)
            late_start = max(stimulus_steps + 1, steps // 2)
            late_fraction = float(population_spikes[late_start:].sum() / max(total_spikes, 1))
            trial_energy.append(float(total_spikes))
            trial_peaks.append(peak_fraction)
            trial_late.append(late_fraction)

            for name in active_dn_types:
                population = populations.descending_types[name]
                sizes = np.asarray([population.left.size, population.right.size])
                rates = dn_counts[name] / np.maximum(sizes * steps, 1)
                ipsilateral_index = 0 if stimulus_side == "L" else 1
                contralateral_index = 1 - ipsilateral_index
                ipsilateral = float(rates[ipsilateral_index])
                contralateral = float(rates[contralateral_index])
                response = ipsilateral + contralateral
                selectivity = (ipsilateral - contralateral) / max(response, 1e-12)
                selectivities.append(selectivity)
                response_rates.append(response)
                matrix_rows.append(
                    {
                        "synaptic_scale": scale,
                        "threshold": threshold,
                        "odor_channel": channel,
                        "stimulus_side": stimulus_side,
                        "dn_type": name,
                        "left_spikes": int(dn_counts[name][0]),
                        "right_spikes": int(dn_counts[name][1]),
                        "ipsilateral_rate": ipsilateral,
                        "contralateral_rate": contralateral,
                        "lateral_selectivity": selectivity,
                        "total_population_spikes": total_spikes,
                        "peak_population_fraction": peak_fraction,
                        "late_spike_fraction": late_fraction,
                    }
                )

        row: dict[str, object] = {
            "synaptic_scale": scale,
            "threshold": threshold,
            "mean_lateral_selectivity": fmean(selectivities),
            "mean_dn_response_rate": fmean(response_rates),
            "mean_total_spikes": fmean(trial_energy),
            "max_peak_fraction": max(trial_peaks),
            "mean_late_fraction": fmean(trial_late),
        }
        row["feasible"] = bool(
            row["mean_dn_response_rate"] > 0
            and row["max_peak_fraction"] <= maximum_peak_fraction
            and row["mean_late_fraction"] <= maximum_late_fraction
        )
        candidate_rows.append(row)

    feasible_indices = [
        index for index, row in enumerate(candidate_rows) if row["feasible"]
    ]
    frontier = np.zeros(len(candidate_rows), dtype=bool)
    if feasible_indices:
        values = np.asarray(
            [
                [
                    float(candidate_rows[index]["mean_lateral_selectivity"]),
                    float(candidate_rows[index]["mean_dn_response_rate"]),
                    float(candidate_rows[index]["mean_total_spikes"]),
                    float(candidate_rows[index]["max_peak_fraction"]),
                    float(candidate_rows[index]["mean_late_fraction"]),
                ]
                for index in feasible_indices
            ]
        )
        selected = nondominated_mask(
            values,
            (
                Objective("mean_lateral_selectivity", True),
                Objective("mean_dn_response_rate", True),
                Objective("mean_total_spikes", False),
                Objective("max_peak_fraction", False),
                Objective("mean_late_fraction", False),
            ),
        )
        frontier[np.asarray(feasible_indices)[selected]] = True

    for row, is_frontier in zip(candidate_rows, frontier):
        row["frontier"] = bool(is_frontier)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "intervention-matrix.csv", matrix_rows)
    _write_rows(output / "calibration-points.csv", candidate_rows)
    _write_rows(
        output / "frontier.csv",
        [row for row in candidate_rows if row["frontier"]],
        list(candidate_rows[0]),
    )
    return len(candidate_rows), int(np.count_nonzero(frontier))


def plot_calibration_frontier(points_path: Path, output_path: Path) -> None:
    """Render a compact scientific projection of the calibration frontier."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional analysis install
        raise ImportError("plotting requires the 'analysis' extra") from error

    rows: list[dict[str, str]]
    with Path(points_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("calibration points table is empty")
    selectivity = np.asarray([float(row["mean_lateral_selectivity"]) for row in rows])
    energy = np.asarray([float(row["mean_total_spikes"]) for row in rows])
    response = np.asarray([float(row["mean_dn_response_rate"]) for row in rows])
    frontier = np.asarray([row["frontier"].lower() == "true" for row in rows])
    sizes = 30 + 270 * response / max(float(response.max()), 1e-12)

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.scatter(energy[~frontier], selectivity[~frontier], s=sizes[~frontier],
                 color="#9ca3af", alpha=0.75, label="candidate")
    axis.scatter(energy[frontier], selectivity[frontier], s=sizes[frontier],
                 color="#d97706", edgecolor="#111827", linewidth=0.8,
                 label="nondominated")
    for row, x, y, is_frontier in zip(rows, energy, selectivity, frontier):
        if is_frontier:
            axis.annotate(
                f"s={float(row['synaptic_scale']):g}, θ={float(row['threshold']):g}",
                (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7,
            )
    axis.axhline(0, color="#374151", linewidth=0.8, linestyle="--")
    axis.set_title("Receptor-to-steering calibration frontier")
    axis.set_xlabel("Mean population spikes per intervention")
    axis.set_ylabel("Mean ipsilateral DN selectivity")
    axis.text(
        0.01,
        0.02,
        "Marker area ∝ mean DN response rate",
        transform=axis.transAxes,
        fontsize=8,
    )
    axis.legend(frameon=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
