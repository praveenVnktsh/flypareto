"""Closed-loop odor-source localization with the complete MaleCNS graph."""

from __future__ import annotations

import csv
import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np

from .flygym_bridge import _FlyGymPhysics
from .graph import Connectome
from .interface import (
    BilateralNeuralInterface,
    DescendingDriveDecoder,
    MaleCNSPopulations,
    SteeringDriveDecoder,
)
from .pareto import Objective, nondominated_mask
from .sim import EventDrivenLIF, LIFConfig


@dataclass(frozen=True)
class ChemotaxisResult:
    neural_steps: int
    physics_steps: int
    target_xy_mm: tuple[float, float]
    initial_distance_mm: float
    final_distance_mm: float
    best_distance_mm: float
    progress_mm: float
    source_reached: bool
    total_spikes: int
    descending_spikes: int
    neural_energy_proxy: float
    mechanical_effort_proxy: float
    min_thorax_height_mm: float
    upright_fraction: float
    mean_odor_asymmetry: float
    mean_descending_asymmetry: float
    mean_drive_asymmetry: float
    odor_drive_correlation: float
    locomotion_viable: bool
    finite: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _gaussian_odor(sensor_xy: np.ndarray, target_xy: np.ndarray, sigma_mm: float) -> float:
    distance_squared = float(np.sum(np.square(sensor_xy - target_xy)))
    return float(np.exp(-distance_squared / (2 * sigma_mm**2)))


def _finite_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    result = float(np.corrcoef(first, second)[0, 1])
    return result if np.isfinite(result) else 0.0


def run_chemotaxis_trial(
    graph: Connectome,
    annotations_path: Path,
    neural_steps: int = 500,
    target_offset_mm: tuple[float, float] = (2.0, 0.75),
    source_radius_mm: float = 0.5,
    odor_sigma_mm: float = 1.5,
    odor_gain: float = 2.0,
    odor_channels: tuple[str, ...] = ("ORN_DM1", "ORN_VA2"),
    synaptic_scale: float = 0.008,
    proprioceptive_speed_scale: float = 5.0,
    decoder_half_saturation: float = 0.01,
    decoder_smoothing: float = 0.9,
    decoder_mode: str = "steering",
    seed: int = 0,
) -> ChemotaxisResult:
    """Test whether bilateral odor feedback brings the embodied fly to a source.

    The odor field is an isotropic Gaussian sampled at the left and right
    funiculi. This isolates sensorimotor capacity while holding plume physics
    fixed; turbulent plume experiments belong in a later assay tier.
    """
    if neural_steps <= 0:
        raise ValueError("neural_steps must be positive")
    if source_radius_mm <= 0 or odor_sigma_mm <= 0 or odor_gain < 0:
        raise ValueError("source radius and odor sigma must be positive; odor gain cannot be negative")
    if proprioceptive_speed_scale <= 0:
        raise ValueError("proprioceptive_speed_scale must be positive")
    if not odor_channels:
        raise ValueError("at least one odor channel is required")

    physics = _FlyGymPhysics(seed=seed)
    populations = MaleCNSPopulations.from_annotations(graph, annotations_path)
    interface = BilateralNeuralInterface(graph, populations)
    if decoder_mode == "aggregate":
        decoder: DescendingDriveDecoder | SteeringDriveDecoder = DescendingDriveDecoder(
            half_saturation=decoder_half_saturation,
            smoothing=decoder_smoothing,
        )
    elif decoder_mode == "steering":
        decoder = SteeringDriveDecoder(
            forward_half_saturation=decoder_half_saturation,
            smoothing=decoder_smoothing,
        )
        steering_population_size = sum(
            populations.descending_types[name].left.size
            + populations.descending_types[name].right.size
            for name in ("DNa01", "DNa02")
        )
        if steering_population_size == 0:
            raise ValueError("annotations do not resolve DNa01/DNa02 steering neurons")
    else:
        raise ValueError("decoder_mode must be 'aggregate' or 'steering'")
    neural = EventDrivenLIF(graph, LIFConfig(synaptic_scale=synaptic_scale))
    neural.reset()
    physics_per_neural = max(1, round((neural.config.dt_ms / 1000) / physics.timestep))
    initial_position = physics.thorax_position
    target_xy = initial_position[:2] + np.asarray(target_offset_mm, dtype=float)
    initial_distance = float(np.linalg.norm(initial_position[:2] - target_xy))
    distances = [initial_distance]
    heights: list[float] = []
    upright: list[bool] = []
    total_spikes = 0
    descending_spikes = 0
    mechanical_effort = 0.0
    odor_asymmetries: list[float] = []
    descending_asymmetries: list[float] = []
    drive_asymmetries: list[float] = []

    for _ in range(neural_steps):
        antennae = physics.antenna_positions
        left_odor = odor_gain * _gaussian_odor(antennae[0, :2], target_xy, odor_sigma_mm)
        right_odor = odor_gain * _gaussian_odor(antennae[1, :2], target_xy, odor_sigma_mm)
        odor_asymmetries.append(left_odor - right_odor)
        joint_speed = physics.bilateral_joint_speed
        proprioception = np.tanh(joint_speed / proprioceptive_speed_scale)
        external = interface.encode_sensory_types(
            odor_channels,
            left_odor,
            right_odor,
        )
        external += interface.encode(
            "mechanosensory_proprioceptive",
            float(proprioception[0]),
            float(proprioception[1]),
        )
        spikes = neural.step(external)
        total_spikes += int(np.count_nonzero(spikes))
        descending_spikes += int(np.count_nonzero(spikes[populations.descending.left]))
        descending_spikes += int(np.count_nonzero(spikes[populations.descending.right]))
        bilateral_activity = interface.descending_activity(spikes)
        if decoder_mode == "steering":
            steering_activity = interface.descending_type_activity(spikes, "DNa01", "DNa02")
            descending_asymmetries.append(
                float(steering_activity[0] - steering_activity[1])
            )
            drive = decoder.update(bilateral_activity, steering_activity)
        else:
            descending_asymmetries.append(float(bilateral_activity[0] - bilateral_activity[1]))
            drive = decoder.update(bilateral_activity)
        drive_asymmetries.append(float(drive[0] - drive[1]))
        for _ in range(physics_per_neural):
            mechanical_effort += physics.step(drive)
        position = physics.thorax_position
        distances.append(float(np.linalg.norm(position[:2] - target_xy)))
        heights.append(float(position[2]))
        upright.append(physics.orientation_deviation_radians < np.deg2rad(60))

    final_distance = distances[-1]
    best_distance = min(distances)
    min_height = min(heights)
    upright_fraction = float(np.mean(upright))
    finite = bool(
        np.all(np.isfinite(distances))
        and np.isfinite(mechanical_effort)
        and np.isfinite(min_height)
    )
    locomotion_viable = finite and min_height >= 0.35 and upright_fraction >= 0.9
    odor_array = np.asarray(odor_asymmetries)
    drive_array = np.asarray(drive_asymmetries)
    return ChemotaxisResult(
        neural_steps=neural_steps,
        physics_steps=neural_steps * physics_per_neural,
        target_xy_mm=tuple(target_xy.tolist()),
        initial_distance_mm=initial_distance,
        final_distance_mm=final_distance,
        best_distance_mm=best_distance,
        progress_mm=initial_distance - best_distance,
        source_reached=locomotion_viable and best_distance <= source_radius_mm,
        total_spikes=total_spikes,
        descending_spikes=descending_spikes,
        neural_energy_proxy=float(total_spikes),
        mechanical_effort_proxy=mechanical_effort,
        min_thorax_height_mm=min_height,
        upright_fraction=upright_fraction,
        mean_odor_asymmetry=float(np.mean(odor_array)),
        mean_descending_asymmetry=float(np.mean(descending_asymmetries)),
        mean_drive_asymmetry=float(np.mean(drive_array)),
        odor_drive_correlation=_finite_correlation(odor_array, drive_array),
        locomotion_viable=locomotion_viable,
        finite=finite,
    )


ChemotaxisRunner = Callable[..., ChemotaxisResult]


def run_chemotaxis_sweep(
    graph: Connectome,
    annotations_path: Path,
    output: Path,
    synaptic_scales: list[float],
    decoder_half_saturations: list[float],
    odor_gains: list[float],
    seeds: list[int],
    target_laterals_mm: list[float],
    neural_steps: int = 500,
    target_forward_mm: float = 2.0,
    source_radius_mm: float = 0.5,
    odor_sigma_mm: float = 1.5,
    odor_channels: tuple[str, ...] = ("ORN_DM1", "ORN_VA2"),
    minimum_reach_fraction: float = 0.5,
    decoder_mode: str = "steering",
    runner: ChemotaxisRunner | None = None,
) -> tuple[int, int]:
    """Pareto-rank ecological candidates across mirrored sources and seeds."""
    grids = (
        synaptic_scales,
        decoder_half_saturations,
        odor_gains,
        seeds,
        target_laterals_mm,
    )
    if any(not values for values in grids):
        raise ValueError("parameter grids, seeds, and target laterals must be non-empty")
    if not 0 <= minimum_reach_fraction <= 1:
        raise ValueError("minimum_reach_fraction must be between zero and one")
    execute = run_chemotaxis_trial if runner is None else runner
    rows: list[dict[str, float | bool]] = []

    for synaptic_scale, half_saturation, odor_gain in itertools.product(*grids[:3]):
        trials = [
            execute(
                graph,
                annotations_path,
                neural_steps=neural_steps,
                target_offset_mm=(target_forward_mm, lateral),
                source_radius_mm=source_radius_mm,
                odor_sigma_mm=odor_sigma_mm,
                odor_gain=odor_gain,
                odor_channels=odor_channels,
                synaptic_scale=synaptic_scale,
                decoder_half_saturation=half_saturation,
                decoder_mode=decoder_mode,
                seed=seed,
            )
            for seed, lateral in itertools.product(seeds, target_laterals_mm)
        ]
        reach_fraction = fmean(float(trial.source_reached) for trial in trials)
        progress_fractions = [trial.progress_mm / trial.initial_distance_mm for trial in trials]
        row: dict[str, float | bool] = {
            "synaptic_scale": synaptic_scale,
            "decoder_half_saturation": half_saturation,
            "odor_gain": odor_gain,
            "replicates": float(len(trials)),
            "source_reach_fraction": reach_fraction,
            "locomotion_viable_fraction": fmean(
                float(trial.locomotion_viable) for trial in trials
            ),
            "progress_fraction": fmean(progress_fractions),
            "progress_fraction_sd": pstdev(progress_fractions),
            "neural_energy_proxy": fmean(trial.neural_energy_proxy for trial in trials),
            "neural_energy_proxy_sd": pstdev(trial.neural_energy_proxy for trial in trials),
            "mechanical_effort_proxy": fmean(
                trial.mechanical_effort_proxy for trial in trials
            ),
            "mechanical_effort_proxy_sd": pstdev(
                trial.mechanical_effort_proxy for trial in trials
            ),
        }
        row["feasible"] = bool(
            all(trial.locomotion_viable for trial in trials)
            and reach_fraction >= minimum_reach_fraction
        )
        rows.append(row)

    feasible_indices = [index for index, row in enumerate(rows) if row["feasible"]]
    frontier = np.zeros(len(rows), dtype=bool)
    if feasible_indices:
        values = np.asarray(
            [
                [
                    float(rows[index]["source_reach_fraction"]),
                    float(rows[index]["progress_fraction"]),
                    float(rows[index]["neural_energy_proxy"]),
                    float(rows[index]["mechanical_effort_proxy"]),
                ]
                for index in feasible_indices
            ]
        )
        selected = nondominated_mask(
            values,
            (
                Objective("source_reach_fraction", True),
                Objective("progress_fraction", True),
                Objective("neural_energy_proxy", False),
                Objective("mechanical_effort_proxy", False),
            ),
        )
        frontier[np.asarray(feasible_indices)[selected]] = True

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = [*rows[0], "frontier"]
    for filename, only_frontier in (
        ("chemotaxis-points.csv", False),
        ("frontier.csv", True),
    ):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, is_frontier in zip(rows, frontier):
                if not only_frontier or is_frontier:
                    writer.writerow({**row, "frontier": bool(is_frontier)})
    return len(rows), int(np.count_nonzero(frontier))
