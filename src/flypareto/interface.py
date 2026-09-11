"""Typed boundary between MaleCNS neural activity and an embodied environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyarrow import feather

from .data import first_column
from .graph import BODY_ALIASES, Connectome


@dataclass(frozen=True)
class BilateralPopulation:
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class MaleCNSPopulations:
    """Graph indices for biological sensory modalities and descending outputs."""

    sensory: dict[str, BilateralPopulation]
    sensory_types: dict[str, BilateralPopulation]
    descending: BilateralPopulation
    descending_types: dict[str, BilateralPopulation]

    @classmethod
    def from_annotations(cls, graph: Connectome, path: Path) -> MaleCNSPopulations:
        table = feather.read_table(path, memory_map=True)
        body_col = first_column(table.column_names, BODY_ALIASES)
        required = (body_col, "status", "superclass", "class", "rootSide", "somaSide")
        if any(column is None or column not in table.column_names for column in required):
            raise ValueError("MaleCNS annotations are missing interface columns")

        body_ids = np.asarray(table[body_col].combine_chunks()).astype(np.int64, copy=False)
        status = table["status"].combine_chunks().to_pylist()
        superclass = table["superclass"].combine_chunks().to_pylist()
        neuron_class = table["class"].combine_chunks().to_pylist()
        root_side = table["rootSide"].combine_chunks().to_pylist()
        soma_side = table["somaSide"].combine_chunks().to_pylist()
        neuron_type = (
            table["type"].combine_chunks().to_pylist()
            if "type" in table.column_names
            else [None] * table.num_rows
        )

        def resolve(mask: np.ndarray) -> np.ndarray:
            selected = body_ids[mask]
            positions = np.searchsorted(graph.node_ids, selected)
            valid = positions < graph.neuron_count
            valid[valid] &= graph.node_ids[positions[valid]] == selected[valid]
            return np.unique(positions[valid].astype(np.int32, copy=False))

        traced = np.fromiter((value == "Traced" for value in status), bool)
        modalities: dict[str, BilateralPopulation] = {}
        class_array = np.asarray(neuron_class, dtype=object)
        root_array = np.asarray(root_side, dtype=object)
        for modality in (
            "visual",
            "olfactory",
            "gustatory",
            "mechanosensory",
            "mechanosensory_tactile",
            "mechanosensory_proprioceptive",
            "thermosensory",
            "hygrosensory",
        ):
            base = traced & (class_array == modality)
            modalities[modality] = BilateralPopulation(
                left=resolve(base & (root_array == "L")),
                right=resolve(base & (root_array == "R")),
            )

        super_array = np.asarray(superclass, dtype=object)
        soma_array = np.asarray(soma_side, dtype=object)
        descending = traced & (super_array == "descending_neuron")
        type_array = np.asarray(neuron_type, dtype=object)
        olfactory = traced & (class_array == "olfactory")
        sensory_types = {
            name: BilateralPopulation(
                left=resolve(olfactory & (type_array == name) & (root_array == "L")),
                right=resolve(olfactory & (type_array == name) & (root_array == "R")),
            )
            for name in sorted(
                {str(value) for value in type_array[olfactory] if value is not None}
            )
        }
        descending_types = {
            name: BilateralPopulation(
                left=resolve(descending & (type_array == name) & (soma_array == "L")),
                right=resolve(descending & (type_array == name) & (soma_array == "R")),
            )
            for name in ("DNa01", "DNa02", "DNg13", "MDN", "DNp17")
        }
        return cls(
            sensory=modalities,
            sensory_types=sensory_types,
            descending=BilateralPopulation(
                left=resolve(descending & (soma_array == "L")),
                right=resolve(descending & (soma_array == "R")),
            ),
            descending_types=descending_types,
        )


class BilateralNeuralInterface:
    """Encode bilateral sensation and decode raw descending population activity."""

    def __init__(self, graph: Connectome, populations: MaleCNSPopulations):
        self.graph = graph
        self.populations = populations

    def encode(
        self,
        modality: str,
        left: float,
        right: float,
        amplitude: float = 1.2,
    ) -> np.ndarray:
        if modality not in self.populations.sensory:
            raise KeyError(f"unknown sensory modality: {modality}")
        population = self.populations.sensory[modality]
        current = np.zeros(self.graph.neuron_count, dtype=np.float32)
        if population.left.size:
            current[population.left] = max(0.0, float(left)) * amplitude
        if population.right.size:
            current[population.right] = max(0.0, float(right)) * amplitude
        return current

    def encode_sensory_types(
        self,
        type_names: tuple[str, ...],
        left: float,
        right: float,
        amplitude: float = 1.2,
    ) -> np.ndarray:
        """Encode a cue into named receptor/glomerular sensory channels."""
        missing = [name for name in type_names if name not in self.populations.sensory_types]
        if missing:
            raise KeyError(f"unknown sensory types: {', '.join(missing)}")
        populations = [self.populations.sensory_types[name] for name in type_names]
        combined = BilateralPopulation(
            left=np.unique(np.concatenate([population.left for population in populations])),
            right=np.unique(np.concatenate([population.right for population in populations])),
        )
        current = np.zeros(self.graph.neuron_count, dtype=np.float32)
        current[combined.left] = max(0.0, float(left)) * amplitude
        current[combined.right] = max(0.0, float(right)) * amplitude
        return current

    def descending_activity(self, spikes: np.ndarray) -> np.ndarray:
        spikes = np.asarray(spikes, dtype=float)
        if spikes.shape != (self.graph.neuron_count,):
            raise ValueError("spikes do not match graph neuron count")
        return self.bilateral_activity(spikes, self.populations.descending)

    def bilateral_activity(
        self, spikes: np.ndarray, population: BilateralPopulation
    ) -> np.ndarray:
        """Return left/right firing fractions for a specified population."""
        spikes = np.asarray(spikes, dtype=float)
        if spikes.shape != (self.graph.neuron_count,):
            raise ValueError("spikes do not match graph neuron count")
        return np.asarray(
            [
                float(np.mean(spikes[population.left])) if population.left.size else 0.0,
                float(np.mean(spikes[population.right])) if population.right.size else 0.0,
            ],
            dtype=np.float32,
        )

    def descending_type_activity(self, spikes: np.ndarray, *type_names: str) -> np.ndarray:
        """Pool only explicitly named, functionally characterized DN types."""
        populations = [self.populations.descending_types[name] for name in type_names]
        combined = BilateralPopulation(
            left=np.unique(np.concatenate([population.left for population in populations])),
            right=np.unique(np.concatenate([population.right for population in populations])),
        )
        return self.bilateral_activity(spikes, combined)


class DescendingDriveDecoder:
    """Calibratable, explicitly provisional mapping to two locomotor drive channels."""

    def __init__(
        self,
        baseline: float = 0.4,
        span: float = 0.8,
        half_saturation: float = 0.01,
        smoothing: float = 0.9,
    ):
        self.baseline = baseline
        self.span = span
        self.half_saturation = half_saturation
        self.smoothing = smoothing
        self.activity = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self.activity.fill(0)

    def update(self, bilateral_activity: np.ndarray) -> np.ndarray:
        activity = np.asarray(bilateral_activity, dtype=np.float32)
        if activity.shape != (2,):
            raise ValueError("bilateral activity must have shape (2,)")
        self.activity = self.smoothing * self.activity + (1 - self.smoothing) * activity
        normalized = self.activity / (self.activity + self.half_saturation)
        return self.baseline + self.span * normalized


class SteeringDriveDecoder:
    """Readout based on steering DNs with experimentally supported roles.

    DNa01 and DNa02 activity is associated with ipsiversive steering; DNa02
    shortens strides on the inside of a turn. We therefore reduce ipsilateral
    locomotor drive rather than treating all DNs as interchangeable speed units.
    """

    def __init__(
        self,
        baseline: float = 0.4,
        forward_span: float = 0.8,
        steering_span: float = 0.35,
        forward_half_saturation: float = 0.01,
        steering_half_saturation: float = 0.05,
        smoothing: float = 0.9,
    ):
        self.baseline = baseline
        self.forward_span = forward_span
        self.steering_span = steering_span
        self.forward_half_saturation = forward_half_saturation
        self.steering_half_saturation = steering_half_saturation
        self.smoothing = smoothing
        self.forward_activity = 0.0
        self.steering_activity = np.zeros(2, dtype=np.float32)

    def update(
        self,
        global_activity: np.ndarray,
        steering_activity: np.ndarray,
    ) -> np.ndarray:
        global_activity = np.asarray(global_activity, dtype=np.float32)
        steering_activity = np.asarray(steering_activity, dtype=np.float32)
        if global_activity.shape != (2,) or steering_activity.shape != (2,):
            raise ValueError("global and steering activities must have shape (2,)")
        mean_global = float(np.mean(global_activity))
        self.forward_activity = (
            self.smoothing * self.forward_activity + (1 - self.smoothing) * mean_global
        )
        self.steering_activity = (
            self.smoothing * self.steering_activity + (1 - self.smoothing) * steering_activity
        )
        forward = self.baseline + self.forward_span * (
            self.forward_activity
            / (self.forward_activity + self.forward_half_saturation)
        )
        steering = self.steering_activity / (
            self.steering_activity + self.steering_half_saturation
        )
        return np.clip(forward - self.steering_span * steering, 0.2, 1.4)
