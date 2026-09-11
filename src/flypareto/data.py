"""Official MaleCNS data acquisition and tabular schema helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path

MALECNS_VERSION = "v1.0"
BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome"
)

DEFAULT_FILES: Mapping[str, str] = {
    "body-annotations.feather": (
        f"{BASE_URL}/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    ),
    "body-neurotransmitters.feather": (
        f"{BASE_URL}/body-neurotransmitters-male-cns-v1.0.feather"
    ),
    "connectome-weights.feather": (
        f"{BASE_URL}/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    ),
}

EXPECTED_SHA256: Mapping[str, str] = {
    "body-annotations.feather": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "body-neurotransmitters.feather": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    "connectome-weights.feather": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
}


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, chunk_size: int = 8 << 20) -> Path:
    """Download *url* atomically, resuming a partial transfer when supported."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:  # nosec: fixed official URLs
        status = getattr(response, "status", 200)
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as target:
            shutil.copyfileobj(response, target, length=chunk_size)
    partial.replace(destination)
    return destination


def download_default_dataset(data_dir: Path) -> dict[str, Path]:
    """Download the three files needed for full-graph simulation."""
    outputs: dict[str, Path] = {}
    for name, url in DEFAULT_FILES.items():
        destination = Path(data_dir) / name
        if not destination.exists():
            download_file(url, destination)
        actual_hash = sha256_file(destination)
        if actual_hash != EXPECTED_SHA256[name]:
            raise ValueError(
                f"SHA-256 mismatch for {destination}: expected {EXPECTED_SHA256[name]}, "
                f"found {actual_hash}"
            )
        outputs[name] = destination
    manifest = {
        "dataset": "male-cns",
        "version": MALECNS_VERSION,
        "source": "Google Research / HHMI Janelia MaleCNS",
        "files": {name: {"url": DEFAULT_FILES[name], "bytes": path.stat().st_size,
                         "sha256": EXPECTED_SHA256[name]}
                  for name, path in outputs.items()},
    }
    (Path(data_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def first_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    """Return the first present alias, comparing case-insensitively."""
    lookup = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None
