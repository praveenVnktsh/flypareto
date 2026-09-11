"""Build and load a compact outgoing graph for full-connectome simulation."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import feather, ipc

from .data import first_column

BODY_ALIASES = ("body", "bodyid", "body_id", "root_id", "segment_id")
PRE_ALIASES = ("body_pre", "bodypre", "pre", "source", "source_id")
POST_ALIASES = ("body_post", "bodypost", "post", "target", "target_id")
WEIGHT_ALIASES = ("weight", "syn_count", "synapse_count", "count")
NT_ALIASES = ("predicted_nt", "celltype_predicted_nt", "nt", "neurotransmitter")

EXCITATORY = {"acetylcholine", "ach"}
INHIBITORY = {"gaba", "glutamate", "glut", "histamine"}
MODULATORY = {"dopamine", "da", "octopamine", "oct", "serotonin", "ser"}


@dataclass(frozen=True)
class Connectome:
    """Outgoing compressed sparse rows: one row per presynaptic neuron."""

    node_ids: np.ndarray
    indptr: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    signs: np.ndarray

    @property
    def neuron_count(self) -> int:
        return int(self.node_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.targets.size)

    def save(self, path: Path, metadata: dict[str, object] | None = None) -> None:
        """Save as separate arrays so the full graph remains memory-mappable."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing graph: {path}")
        path.mkdir(parents=True)
        np.save(path / "node_ids.npy", self.node_ids)
        np.save(path / "indptr.npy", self.indptr)
        np.save(path / "targets.npy", self.targets)
        np.save(path / "weights.npy", self.weights)
        np.save(path / "signs.npy", self.signs)
        details = {
            "neuron_count": self.neuron_count,
            "edge_count": self.edge_count,
            **(metadata or {}),
        }
        (path / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path, mmap_mode: str | None = "r") -> Connectome:
        path = Path(path)
        return cls(
            node_ids=np.load(path / "node_ids.npy", mmap_mode=mmap_mode, allow_pickle=False),
            indptr=np.load(path / "indptr.npy", mmap_mode=mmap_mode, allow_pickle=False),
            targets=np.load(path / "targets.npy", mmap_mode=mmap_mode, allow_pickle=False),
            weights=np.load(path / "weights.npy", mmap_mode=mmap_mode, allow_pickle=False),
            signs=np.load(path / "signs.npy", mmap_mode=mmap_mode, allow_pickle=False),
        )


def _required(columns: Iterable[str], aliases: Iterable[str], label: str) -> str:
    value = first_column(columns, aliases)
    if value is None:
        raise ValueError(f"Could not find {label} column. Available columns: {list(columns)}")
    return value


def _nt_sign(value: object) -> int:
    label = "" if value is None else str(value).strip().lower()
    if label in EXCITATORY:
        return 1
    if label in INHIBITORY:
        return -1
    if label in MODULATORY:
        return 0
    return 0


def _ipc_reader(path: Path) -> ipc.RecordBatchFileReader:
    return ipc.open_file(pa.memory_map(str(path), "r"))


def _mapped_batch(
    batch: pa.RecordBatch,
    node_ids: np.ndarray,
    pre_col: str,
    post_col: str,
    weight_col: str,
    min_weight: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_ids = np.asarray(batch.column(batch.schema.get_field_index(pre_col)))
    post_ids = np.asarray(batch.column(batch.schema.get_field_index(post_col)))
    raw_weights = np.asarray(batch.column(batch.schema.get_field_index(weight_col)))
    pre_index = np.searchsorted(node_ids, pre_ids)
    post_index = np.searchsorted(node_ids, post_ids)
    valid_pre = pre_index < node_ids.size
    valid_post = post_index < node_ids.size
    valid_pre[valid_pre] &= node_ids[pre_index[valid_pre]] == pre_ids[valid_pre]
    valid_post[valid_post] &= node_ids[post_index[valid_post]] == post_ids[valid_post]
    keep = valid_pre & valid_post & (raw_weights >= min_weight)
    return (
        pre_index[keep].astype(np.int32, copy=False),
        post_index[keep].astype(np.int32, copy=False),
        raw_weights[keep].astype(np.float32, copy=False),
    )


def build_connectome(
    annotations_path: Path,
    neurotransmitters_path: Path,
    connections_path: Path,
    min_weight: int = 1,
    output_path: Path | None = None,
) -> tuple[Connectome, dict[str, object]]:
    """Build the full graph induced by all annotated MaleCNS neurons."""
    annotations = feather.read_table(annotations_path, memory_map=True)
    ann_body = _required(annotations.column_names, BODY_ALIASES, "annotation body ID")
    status_col = first_column(annotations.column_names, ("status",))
    if status_col is None:
        raise ValueError("annotation table has no status column; cannot identify traced neurons")
    statuses = np.asarray(annotations[status_col].combine_chunks().to_pylist(), dtype=object)
    all_annotation_ids = np.asarray(annotations[ann_body].combine_chunks()).astype(
        np.int64, copy=False
    )
    node_ids = all_annotation_ids[statuses == "Traced"]
    node_ids = np.unique(node_ids)

    transmitters = feather.read_table(neurotransmitters_path, memory_map=True)
    nt_body = _required(transmitters.column_names, BODY_ALIASES, "transmitter body ID")
    nt_col = _required(transmitters.column_names, NT_ALIASES, "neurotransmitter")
    nt_ids = np.asarray(transmitters[nt_body].combine_chunks()).astype(np.int64, copy=False)
    nt_positions = np.searchsorted(node_ids, nt_ids)
    nt_valid = nt_positions < node_ids.size
    nt_valid[nt_valid] &= node_ids[nt_positions[nt_valid]] == nt_ids[nt_valid]
    selected_positions = nt_positions[nt_valid]
    nt_values = transmitters[nt_col].combine_chunks().filter(pa.array(nt_valid)).to_pylist()
    signs = np.zeros(node_ids.size, dtype=np.int8)
    for index, transmitter in zip(selected_positions, nt_values):
        signs[index] = _nt_sign(transmitter)

    reader = _ipc_reader(connections_path)
    columns = reader.schema.names
    pre_col = _required(columns, PRE_ALIASES, "presynaptic body ID")
    post_col = _required(columns, POST_ALIASES, "postsynaptic body ID")
    weight_col = _required(columns, WEIGHT_ALIASES, "connection weight")

    # Pass one counts retained outgoing edges without materializing the 151M-row table.
    counts = np.zeros(node_ids.size, dtype=np.int64)
    raw_connection_rows = 0
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        raw_connection_rows += batch.num_rows
        pre_index, _, _ = _mapped_batch(
            batch, node_ids, pre_col, post_col, weight_col, min_weight
        )
        counts += np.bincount(pre_index, minlength=node_ids.size)
    indptr = np.empty(node_ids.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    edge_count = int(indptr[-1])
    if output_path is None:
        targets = np.empty(edge_count, dtype=np.int32)
        weights = np.empty(edge_count, dtype=np.float32)
        working_dir = None
    else:
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite existing graph: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        working_dir = Path(tempfile.mkdtemp(prefix=f".{output_path.name}-", dir=output_path.parent))
        targets = np.lib.format.open_memmap(
            working_dir / "targets.npy", mode="w+", dtype=np.int32, shape=(edge_count,)
        )
        weights = np.lib.format.open_memmap(
            working_dir / "weights.npy", mode="w+", dtype=np.float32, shape=(edge_count,)
        )

    # Pass two fills each outgoing row directly, avoiding a global 151M-row argsort.
    cursor = indptr[:-1].copy()
    for batch_index in range(reader.num_record_batches):
        pre_index, post_index, batch_weights = _mapped_batch(
            reader.get_batch(batch_index), node_ids, pre_col, post_col, weight_col, min_weight
        )
        if not pre_index.size:
            continue
        order = np.argsort(pre_index, kind="stable")
        sorted_pre = pre_index[order]
        unique_pre, starts, batch_counts = np.unique(
            sorted_pre, return_index=True, return_counts=True
        )
        group_starts = np.repeat(starts, batch_counts)
        positions = cursor[sorted_pre] + np.arange(sorted_pre.size) - group_starts
        targets[positions] = post_index[order]
        weights[positions] = batch_weights[order]
        cursor[unique_pre] += batch_counts

    if not np.array_equal(cursor, indptr[1:]):
        raise RuntimeError("graph construction did not fill every allocated edge")

    graph = Connectome(node_ids, indptr, targets, weights, signs)
    metadata: dict[str, object] = {
        "dataset": "male-cns:v1.0",
        "min_weight": int(min_weight),
        "raw_connection_rows": int(raw_connection_rows),
        "included_connection_rows": graph.edge_count,
        "known_excitatory_neurons": int(np.count_nonzero(signs == 1)),
        "known_inhibitory_neurons": int(np.count_nonzero(signs == -1)),
        "modulatory_or_unknown_neurons": int(np.count_nonzero(signs == 0)),
    }
    if working_dir is not None and output_path is not None:
        targets.flush()
        weights.flush()
        np.save(working_dir / "node_ids.npy", node_ids)
        np.save(working_dir / "indptr.npy", indptr)
        np.save(working_dir / "signs.npy", signs)
        (working_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        working_dir.replace(output_path)
        graph = Connectome.load(output_path)
    return graph, metadata
