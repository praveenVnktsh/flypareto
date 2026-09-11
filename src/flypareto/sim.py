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

    def run(
        self,
        steps: int,
        external: np.ndarray | None = None,
        initial_spikes: Iterable[int] | None = None,
    ) -> SimulationResult:
        if steps <= 0:
            raise ValueError("steps must be positive")
        n = self.graph.neuron_count
        voltage = np.zeros(n, dtype=np.float32)
        refractory = np.zeros(n, dtype=np.int16)
        spike_counts = np.zeros(n, dtype=np.int32)
        population = np.zeros(steps, dtype=np.int32)
        spikes = np.zeros(n, dtype=bool)
        if initial_spikes is not None:
            indices = np.asarray(list(initial_spikes), dtype=np.int64)
            if np.any((indices < 0) | (indices >= n)):
                raise IndexError("initial spike index is outside the graph")
            spikes[indices] = True
        if external is not None:
            external = np.asarray(external, dtype=np.float32)
            if external.shape not in {(steps, n), (n,)}:
                raise ValueError(f"external must have shape ({steps}, {n}) or ({n},)")

        decay = np.float32(np.exp(-self.config.dt_ms / self.config.tau_ms))
        refractory_steps = max(0, round(self.config.refractory_ms / self.config.dt_ms))

        for step in range(steps):
            synaptic = np.zeros(n, dtype=np.float32)
            active = np.flatnonzero(spikes)
            for source in active:
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
                        self.graph.weights[begin:end] * signed * self.config.synaptic_scale,
                    )

            voltage *= decay
            voltage += synaptic
            if external is not None:
                voltage += external if external.ndim == 1 else external[step]
            refractory = np.maximum(refractory - 1, 0)
            voltage[refractory > 0] = self.config.reset
            spikes = (voltage >= self.config.threshold) & (refractory == 0)
            population[step] = int(np.count_nonzero(spikes))
            spike_counts += spikes
            voltage[spikes] = self.config.reset
            refractory[spikes] = refractory_steps

        return SimulationResult(spike_counts, population, voltage)
