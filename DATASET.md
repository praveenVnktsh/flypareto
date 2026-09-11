# Dataset provenance

FlyPareto uses MaleCNS v1.0, a complete adult male *Drosophila melanogaster*
central-nervous-system connectome produced by HHMI Janelia, Google Research,
Cambridge, MRC LMB, and collaborators.

Official landing page: <https://male-cns.janelia.org/>

Paper DOI: <https://doi.org/10.1016/j.cell.2026.08.015>

## Verified bulk files

| File | Bytes | SHA-256 |
|---|---:|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

These hashes describe the official files retrieved on 2026-09-10. The downloader
verifies them before allowing preprocessing.

## Processed graph definition

The primary graph contains every neuron whose annotation `status` is `Traced`, plus
every row in the official neuron-pair connection table for which both endpoints are
traced. No minimum synaptic-weight threshold is applied beyond the source release's
own minimum confidence of 0.5.

Verified result:

- 165,122 neurons
- 25,563,197 directed neuron-pair connections
- 94,946 neurons assigned an excitatory transmitter
- 50,299 neurons assigned an inhibitory transmitter
- 19,877 modulatory or unresolved neurons

The distinction matters: the annotation file has 211,577 segments and the raw
connection table has 151,856,684 rows, but these include non-neuronal and incomplete
objects that are not part of the finished traced CNS.

MaleCNS data is CC BY and is not redistributed by this repository. Users are
responsible for following the dataset's attribution and citation requirements.

