import numpy as np

from flypareto.graph import Connectome
from flypareto.sim import EventDrivenLIF, LIFConfig


def test_spike_propagates_over_excitatory_edge():
    graph = Connectome(
        node_ids=np.array([10, 20]),
        indptr=np.array([0, 1, 1]),
        targets=np.array([1], dtype=np.int32),
        weights=np.array([2.0], dtype=np.float32),
        signs=np.array([1, -1], dtype=np.int8),
    )
    result = EventDrivenLIF(
        graph,
        LIFConfig(threshold=1.0, synaptic_scale=1.0, refractory_ms=0.0),
    ).run(steps=2, initial_spikes=[0])
    assert result.population_spikes.tolist() == [1, 0]
    assert result.spike_counts.tolist() == [0, 1]


def test_external_shape_is_validated():
    graph = Connectome(
        node_ids=np.array([1]),
        indptr=np.array([0, 0]),
        targets=np.array([], dtype=np.int32),
        weights=np.array([], dtype=np.float32),
        signs=np.array([1], dtype=np.int8),
    )
    try:
        EventDrivenLIF(graph).run(3, external=np.zeros((2, 1)))
    except ValueError as error:
        assert "external" in str(error)
    else:
        raise AssertionError("invalid external shape was accepted")


def test_stateful_step_matches_batch_run():
    graph = Connectome(
        node_ids=np.array([10, 20]),
        indptr=np.array([0, 1, 1]),
        targets=np.array([1], dtype=np.int32),
        weights=np.array([2.0], dtype=np.float32),
        signs=np.array([1, -1], dtype=np.int8),
    )
    config = LIFConfig(threshold=1.0, synaptic_scale=1.0, refractory_ms=0.0)
    stepped = EventDrivenLIF(graph, config)
    stepped.reset([0])
    masks = [stepped.step().copy(), stepped.step().copy()]
    batched = EventDrivenLIF(graph, config).run(steps=2, initial_spikes=[0])
    assert [int(mask.sum()) for mask in masks] == batched.population_spikes.tolist()
