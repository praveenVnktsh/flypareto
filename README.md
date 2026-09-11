# FlyPareto

FlyPareto is an open research implementation for measuring the intelligence–fitness
frontier of an embodied, connectome-constrained fruit fly. The authoritative neural
substrate is the complete Google Research / HHMI Janelia **MaleCNS v1.0** graph:
more than 166,000 neurons spanning the brain, optic lobes, neck connective, and
ventral nerve cord.

The project does **not** treat a connectome as a brain upload. A connectome specifies
wiring, while dynamics, receptor effects, neuromodulation, plasticity, sensory
calibration, and muscle control remain model assumptions. FlyPareto makes those
assumptions explicit and tests how conclusions change under uncertainty.

## Research question

> How much adaptive intelligence can the complete fly nervous-system architecture
> support while retaining survival, reproductive fitness, robustness, energetic
> efficiency, and statistically fly-like behavior?

Survival-critical behavior is a feasibility constraint. Among feasible controllers,
we estimate the nondominated frontier over:

- held-out learning and transfer performance;
- expected lifetime reproductive fitness;
- robustness under perturbation;
- neural activity and plasticity cost; and
- behavioral distance from measured flies.

## Current status

This first executable release provides:

- resumable bulk download of the official MaleCNS annotation, neurotransmitter,
  and complete neuron-pair connectivity tables;
- construction of an indexed, full-neuron outgoing sparse graph;
- a signed leaky-integrate-and-fire reference simulator that traverses active
  outgoing edges;
- reusable nondominated sorting and survival-gated Pareto analysis;
- a deterministic end-to-end demo and tests.

The embodiment, ecological benchmark suite, accelerator backend, optimizer, and
male/female comparative analysis are active roadmap items. See [ROADMAP.md](ROADMAP.md).

## Install

Python 3.10+ is supported.

```bash
uv venv
uv pip install '.[analysis,dev]'
```

## Run the verified demo

```bash
uv run flypareto demo --output results/demo
uv run pytest
```

The demo constructs a deterministic synthetic connectome, simulates a population of
controllers, applies survival floors, and writes `candidates.csv` and `frontier.csv`.
It validates the pipeline mechanics; it is not a biological result.

## Download the complete MaleCNS graph

```bash
uv run flypareto download --data-dir data/raw
```

This downloads approximately 1.2 GB: curated neuron annotations, aggregate
neurotransmitter predictions, and the official 1.1 GB full connection graph. Raw
synapse locations (more than 20 GB across optional tables) are intentionally excluded
from the default because neuron-pair weights are sufficient for the initial dynamics.

Then build the simulation graph:

```bash
uv run flypareto inspect data/raw/connectome-weights.feather
uv run flypareto build \
  --annotations data/raw/body-annotations.feather \
  --neurotransmitters data/raw/body-neurotransmitters.feather \
  --connections data/raw/connectome-weights.feather \
  --output data/processed/malecns-v1.0
```

By default, `build` includes every annotated neuron and every connection between
annotated neurons. `--min-weight` can be used only for sensitivity experiments; it
is not the default scientific condition.

Run a short stability probe on the complete graph:

```bash
uv run flypareto simulate data/processed/malecns-v1.0 \
  --steps 100 --stimulus-count 100 --seed 7
```

Generate the first full-connectome neural operating frontier:

```bash
uv run flypareto sweep data/processed/malecns-v1.0 \
  --annotations data/raw/body-annotations.feather \
  --output results/operating-frontier
```

This stimulates actual annotated sensory neurons and measures recruitment of actual
descending and motor populations across neural operating points. It is a systems
calibration frontier—not yet an intelligence–fitness result. Intelligence and survival
labels are reserved for the embodied, behaviorally validated experiments.

The processed graph is a directory of memory-mappable NumPy arrays. In the verified
v1.0 build, 25,563,197 directed connections link 165,122 officially traced neurons;
the source table's remaining rows involve orphan, glial, unimportant, untraced, or
out-of-scope segments. The two-pass builder scans every source row without loading or
duplicating the 151,856,684-row table in RAM.

## Scientific guardrails

1. The complete neuron graph is used as the primary substrate. Reduced graphs may
   accelerate debugging but cannot establish the main result.
2. Parameters are initially shared by biological group rather than independently
   fitted per synapse.
3. Evaluation uses held-out environments and repeated random seeds.
4. Survival floors are enforced before Pareto ranking.
5. Structural, physiological, embodiment, and measurement uncertainty are reported.
6. Claims about evolutionary pressure require comparative or intervention evidence;
   optimization alone establishes possibility, not evolutionary history.

## Data and attribution

MaleCNS data is distributed separately under CC BY. This repository does not
redistribute the dataset. Download locations and provenance are recorded in
`src/flypareto/data.py`.

- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [MaleCNS project](https://male-cns.janelia.org/)
- [MaleCNS dataset documentation](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- Paper: *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, Cell (2026), DOI `10.1016/j.cell.2026.08.015`.

## License

FlyPareto code is MIT licensed. MaleCNS data retains its original CC BY license and
citation requirements.
