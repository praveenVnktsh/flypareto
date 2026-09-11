import csv

import numpy as np
import pyarrow as pa
from pyarrow import feather

from flypareto.calibration import run_intervention_calibration
from flypareto.graph import Connectome


def test_intervention_calibration_preserves_typed_response_matrix(tmp_path):
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
                "type": ["ORN_DM1", "ORN_DM1", "DNa01", "DNa01"],
                "rootSide": ["L", "R", None, None],
                "somaSide": [None, None, "L", "R"],
            }
        ),
        annotations,
    )
    candidates, frontier = run_intervention_calibration(
        graph,
        annotations,
        tmp_path / "results",
        synaptic_scales=[1.0],
        thresholds=[1.0],
        odor_channels=("ORN_DM1",),
        dn_types=("DNa01",),
        steps=6,
        stimulus_steps=2,
        maximum_peak_fraction=1.0,
        maximum_late_fraction=1.0,
    )
    assert (candidates, frontier) == (1, 1)
    with (tmp_path / "results" / "intervention-matrix.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(float(row["lateral_selectivity"]) == 1.0 for row in rows)
