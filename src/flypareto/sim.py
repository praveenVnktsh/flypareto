"""Reference event-driven neural dynamics for the MaleCNS graph."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .graph import Connectome


@dataclass
class LIFConfig:
    dt_ms: float = 1.0
    tau_ms: float = 20.0
    threshold: float = 1.0
    reset: float = 0.0
    synaptic_scale: float = 0.01
    unknown_sign: float = 0.0
    refractory_ms: float = 2.0


@dataclass
class SimulationResult:
    spike_counts: np.ndarray
    population_spikes: np.ndarray
    final_voltage: np.ndarray


class EventDrivenLIF:
    """A transparent baseline; accelerator backends must preserve its semantics."""

    def __init__(self, graph: Connectome, config: LIFConfig | None = None):
        self.graph = graph
        self.config = config or LIFConfig()
        self.voltage = np.zeros(graph.neuron_count, dtype=np.float32)
        self.refractory = np.zeros(graph.neuron_count, dtype=np.int16)
        self.spikes = np.zeros(graph.neuron_count, dtype=bool)
        self.decay = np.float32(np.exp(-self.config.dt_ms / self.config.tau_ms))
        self.refractory_steps = max(
            0, round(self.config.refractory_ms / self.config.dt_ms)
        )

    def reset(self, initial_spikes: Iterable[int] | None = None) -> None:
        """Reset neural state and optionally seed a set of spiking neurons."""
        self.voltage.fill(0)
        self.refractory.fill(0)
        self.spikes.fill(False)
        if initial_spikes is not None:
            indices = np.asarray(list(initial_spikes), dtype=np.int64)
            if np.any((indices < 0) | (indices >= self.graph.neuron_count)):
                raise IndexError("initial spike index is outside the graph")
            self.spikes[indices] = True

    def step(self, external: np.ndarray | None = None) -> np.ndarray:
        """Advance one millisecond-scale neural step and return the spike mask."""
        n = self.graph.neuron_count
        synaptic = np.zeros(n, dtype=np.float32)
        for source in np.flatnonzero(self.spikes):
            begin = int(self.graph.indptr[source])
            end = int(self.graph.indptr[source + 1])
            if begin == end:
                continue
            sign = int(self.graph.signs[source])
            signed = self.config.unknown_sign if sign == 0 else float(sign)
            if signed:
                np.add.at(
                    synaptic,
                    self.graph.targets[begin:end],
                    self.graph.weights[begin:end]
                    * signed
                    * self.config.synaptic_scale,
                )

        self.voltage *= self.decay
        self.voltage += synaptic
        if external is not None:
            external = np.asarray(external, dtype=np.float32)
            if external.shape != (n,):
                raise ValueError(f"external step input must have shape ({n},)")
            self.voltage += external
        self.refractory = np.maximum(self.refractory - 1, 0)
        self.voltage[self.refractory > 0] = self.config.reset
        self.spikes = (self.voltage >= self.config.threshold) & (self.refractory == 0)
        self.voltage[self.spikes] = self.config.reset
        self.refractory[self.spikes] = self.refractory_steps
        return self.spikes

    def run(
        self,
        steps: int,
        external: np.ndarray | None = None,
        initial_spikes: Iterable[int] | None = None,
    ) -> SimulationResult:
        if steps <= 0:
            raise ValueError("steps must be positive")
        n = self.graph.neuron_count
        spike_counts = np.zeros(n, dtype=np.int32)
        population = np.zeros(steps, dtype=np.int32)
        self.reset(initial_spikes)
        if external is not None:
            external = np.asarray(external, dtype=np.float32)
            if external.shape not in {(steps, n), (n,)}:
                raise ValueError(f"external must have shape ({steps}, {n}) or ({n},)")

        for step in range(steps):
            step_external = None
            if external is not None:
                step_external = external if external.ndim == 1 else external[step]
            spikes = self.step(step_external)
            population[step] = int(np.count_nonzero(spikes))
            spike_counts += spikes

        return SimulationResult(spike_counts, population, self.voltage.copy())
