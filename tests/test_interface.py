import numpy as np
import pyarrow as pa
from pyarrow import feather

from flypareto.graph import Connectome
from flypareto.interface import (
    BilateralNeuralInterface,
    DescendingDriveDecoder,
    MaleCNSPopulations,
    SteeringDriveDecoder,
)


def test_annotations_map_sensation_and_descending_activity(tmp_path):
    graph = Connectome(
        node_ids=np.array([10, 20, 30, 40]),
        indptr=np.zeros(5, dtype=np.int64),
        targets=np.array([], dtype=np.int32),
        weights=np.array([], dtype=np.float32),
        signs=np.ones(4, dtype=np.int8),
    )
    annotations = tmp_path / "annotations.feather"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [10, 20, 30, 40],
                "status": ["Traced"] * 4,
                "superclass": ["ol_sensory", "ol_sensory", "descending_neuron", "descending_neuron"],
                "class": ["visual", "visual", None, None],
                "rootSide": ["L", "R", None, None],
                "somaSide": [None, None, "L", "R"],
            }
        ),
        annotations,
    )
    populations = MaleCNSPopulations.from_annotations(graph, annotations)
    interface = BilateralNeuralInterface(graph, populations)
    encoded = interface.encode("visual", 0.5, 1.0, amplitude=2.0)
    assert encoded.tolist() == [1.0, 2.0, 0.0, 0.0]
    activity = interface.descending_activity(np.array([False, False, True, False]))
    assert activity.tolist() == [1.0, 0.0]


def test_drive_decoder_is_bounded_and_stateful():
    decoder = DescendingDriveDecoder(smoothing=0.0)
    assert np.isclose(decoder.update(np.array([0.0, 1.0]))[0], 0.4)
    assert np.all(decoder.update(np.array([100.0, 100.0])) < 1.21)


def test_steering_decoder_reduces_ipsilateral_drive():
    decoder = SteeringDriveDecoder(smoothing=0.0)
    drive = decoder.update(np.array([0.1, 0.1]), np.array([1.0, 0.0]))
    assert drive[0] < drive[1]
