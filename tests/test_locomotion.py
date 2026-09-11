import csv
from dataclasses import replace

import numpy as np

from flypareto.flygym_bridge import ClosedLoopSmokeResult
from flypareto.graph import Connectome
from flypareto.locomotion import run_locomotion_sweep

BASE_RESULT = ClosedLoopSmokeResult(
    neural_steps=10,
    physics_steps=100,
    total_spikes=100,
    descending_spikes=10,
    mean_drive=(1.0, 1.0),
    drive_range=(0.4, 1.0),
    mean_proprioception=(0.5, 0.5),
    displacement_mm=(1.0, 0.1, 0.0),
    min_thorax_height_mm=0.8,
    upright_fraction=1.0,
    neural_energy_proxy=100.0,
    mechanical_effort_proxy=20.0,
    locomotion_viable=True,
    finite=True,
)


def test_locomotion_sweep_gates_and_ranks(tmp_path):
    graph = Connectome(
        node_ids=np.array([1]),
        indptr=np.array([0, 0]),
        targets=np.array([], dtype=np.int32),
        weights=np.array([], dtype=np.float32),
        signs=np.array([1], dtype=np.int8),
    )

    def runner(*args, synaptic_scale, decoder_half_saturation, seed, **kwargs):
        del args, decoder_half_saturation, seed, kwargs
        if synaptic_scale == 3.0:
            return replace(BASE_RESULT, locomotion_viable=False, upright_fraction=0.5)
        if synaptic_scale == 2.0:
            return replace(
                BASE_RESULT,
                displacement_mm=(2.0, 0.2, 0.0),
                neural_energy_proxy=200.0,
                mechanical_effort_proxy=30.0,
            )
        return BASE_RESULT

    count, frontier = run_locomotion_sweep(
        graph,
        tmp_path / "unused.feather",
        tmp_path / "results",
        synaptic_scales=[1.0, 2.0, 3.0],
        decoder_half_saturations=[0.01],
        proprioceptive_speed_scales=[5.0],
        seeds=[0, 1],
        neural_steps=10,
        runner=runner,
    )
    assert (count, frontier) == (3, 2)
    with (tmp_path / "results" / "locomotion-points.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["feasible"] for row in rows] == ["True", "True", "False"]
    assert [row["frontier"] for row in rows] == ["True", "True", "False"]
