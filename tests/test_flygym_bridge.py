import numpy as np
import pyarrow as pa
import pytest
from pyarrow import feather

from flypareto.flygym_bridge import run_closed_loop_smoke, run_physics_smoke
from flypareto.graph import Connectome


def test_physics_smoke_validates_arguments_before_import():
    with pytest.raises(ValueError, match="steps"):
        run_physics_smoke(steps=0)


def test_physics_smoke_runs_when_embodiment_extra_is_installed():
    pytest.importorskip("flygym")
    result = run_physics_smoke(steps=5)
    assert result.steps == 5
    assert result.finite
    assert np.all(np.isfinite(result.displacement_mm))


def test_closed_loop_smoke_propagates_to_descending_neurons(tmp_path):
    pytest.importorskip("flygym")
    graph = Connectome(
        node_ids=np.array([10, 20, 30, 40]),
        indptr=np.array([0, 1, 2, 2, 2]),
        targets=np.array([2, 3], dtype=np.int32),
        weights=np.array([2.0, 2.0], dtype=np.float32),
        signs=np.ones(4, dtype=np.int8),
    )
    annotations = tmp_path / "annotations.feather"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [10, 20, 30, 40],
                "status": ["Traced"] * 4,
                "superclass": [
                    "cb_sensory",
                    "cb_sensory",
                    "descending_neuron",
                    "descending_neuron",
                ],
                "class": ["olfactory", "olfactory", None, None],
                "rootSide": ["L", "R", None, None],
                "somaSide": [None, None, "L", "R"],
            }
        ),
        annotations,
    )
    result = run_closed_loop_smoke(
        graph,
        annotations,
        neural_steps=3,
        stimulus_steps=1,
        synaptic_scale=1.0,
    )
    assert result.descending_spikes == 2
    assert result.physics_steps == 30
    assert result.finite
