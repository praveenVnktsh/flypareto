import numpy as np
import pyarrow as pa
from pyarrow import feather

from flypareto.graph import Connectome, build_connectome


def test_build_connectome_filters_to_traced_nodes(tmp_path):
    annotations = tmp_path / "annotations.feather"
    transmitters = tmp_path / "transmitters.feather"
    connections = tmp_path / "connections.feather"
    output = tmp_path / "graph"
    feather.write_feather(
        pa.table({"bodyId": [10, 20, 30], "status": ["Traced", "Traced", "Glia"]}),
        annotations,
        chunksize=2,
    )
    feather.write_feather(
        pa.table({"body": [10, 20, 30], "predicted_nt": ["acetylcholine", "gaba", "gaba"]}),
        transmitters,
        chunksize=2,
    )
    feather.write_feather(
        pa.table(
            {
                "body_pre": [10, 10, 20, 30],
                "body_post": [20, 30, 10, 10],
                "weight": [4, 9, 3, 8],
            }
        ),
        connections,
        chunksize=2,
    )

    graph, metadata = build_connectome(
        annotations, transmitters, connections, output_path=output
    )
    assert graph.node_ids.tolist() == [10, 20]
    assert graph.indptr.tolist() == [0, 1, 2]
    assert graph.targets.tolist() == [1, 0]
    assert graph.weights.tolist() == [4.0, 3.0]
    assert graph.signs.tolist() == [1, -1]
    assert metadata["included_connection_rows"] == 2

    loaded = Connectome.load(output)
    assert isinstance(loaded.targets, np.memmap)
    assert loaded.targets.tolist() == [1, 0]

