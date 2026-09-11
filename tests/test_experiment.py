import pyarrow as pa
from pyarrow import feather

from flypareto.demo import synthetic_graph
from flypareto.experiment import run_operating_sweep


def test_operating_sweep_uses_annotated_inputs_and_outputs(tmp_path):
    graph = synthetic_graph(seed=2, neurons=64, degree=4)
    annotations = tmp_path / "annotations.feather"
    labels = ["ol_sensory"] * 8 + ["descending_neuron"] * 8 + ["cb_intrinsic"] * 48
    feather.write_feather(
        pa.table({"bodyId": list(range(64)), "superclass": labels}), annotations
    )
    output = tmp_path / "results"
    candidates, frontier = run_operating_sweep(
        graph,
        annotations,
        output,
        scales=[0.01, 0.02],
        thresholds=[1.0],
        seeds=[1, 2],
        steps=12,
        stimulus_count=4,
    )
    assert candidates == 2
    assert frontier > 0
    assert (output / "operating-points.csv").exists()
    assert (output / "frontier.csv").exists()

