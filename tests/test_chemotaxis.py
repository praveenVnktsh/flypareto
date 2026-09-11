from dataclasses import replace

import numpy as np

from flypareto.chemotaxis import ChemotaxisResult, run_chemotaxis_sweep
from flypareto.graph import Connectome

BASE = ChemotaxisResult(
    neural_steps=10,
    physics_steps=100,
    target_xy_mm=(2.0, 0.75),
    initial_distance_mm=2.0,
    final_distance_mm=0.4,
    best_distance_mm=0.4,
    progress_mm=1.6,
    source_reached=True,
    total_spikes=100,
    descending_spikes=10,
    neural_energy_proxy=100.0,
    mechanical_effort_proxy=20.0,
    min_thorax_height_mm=0.8,
    upright_fraction=1.0,
    mean_odor_asymmetry=0.1,
    mean_descending_asymmetry=0.01,
    mean_drive_asymmetry=0.01,
    odor_drive_correlation=0.5,
    locomotion_viable=True,
    finite=True,
)


def test_chemotaxis_sweep_applies_behavioral_feasibility_floor(tmp_path):
    graph = Connectome(
        node_ids=np.array([1]),
        indptr=np.array([0, 0]),
        targets=np.array([], dtype=np.int32),
        weights=np.array([], dtype=np.float32),
        signs=np.array([1], dtype=np.int8),
    )

    def runner(*args, odor_gain, **kwargs):
        del args, kwargs
        return BASE if odor_gain else replace(BASE, source_reached=False, progress_mm=0.2)

    candidates, frontier = run_chemotaxis_sweep(
        graph,
        tmp_path / "unused.feather",
        tmp_path / "results",
        synaptic_scales=[0.008],
        thresholds=[1.0],
        decoder_half_saturations=[0.01],
        odor_gains=[0.0, 2.0],
        seeds=[0, 1],
        target_laterals_mm=[-0.75, 0.75],
        neural_steps=10,
        runner=runner,
    )
    assert (candidates, frontier) == (2, 1)
