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
- biological MaleCNS sensory and descending-population resolution;
- a pinned FlyGym 2.1 bridge with bilateral proprioceptive feedback;
- a replicated, survival-gated locomotion-viability Pareto sweep;
- a mirrored, receptor-specific chemotaxis assay with no-odor controls;
- a deterministic end-to-end demo and tests.

The embodiment, ecological benchmark suite, accelerator backend, optimizer, and
male/female comparative analysis are active roadmap items. See [ROADMAP.md](ROADMAP.md).

The verified full-data embodied smoke run advanced 165,122 neural states for 100
neural steps and FlyGym for 1,000 physics steps. It produced 245,758 total spikes,
4,104 descending-neuron spikes, finite physical state, and measurable displacement.
This establishes software integration, not biological calibration.

## Install

Python 3.12–3.14 is supported.

```bash
uv venv
uv pip install '.[analysis,dev]'
```

For FlyGym 2.1 embodiment:

```bash
uv pip install '.[analysis,embodiment,dev]'
uv run flypareto physics-smoke --steps 250
```

After building the full graph, run the first connectome-to-body integration test:

```bash
uv run flypareto embodied-smoke data/processed/malecns-v1.0 \
  --annotations data/raw/body-annotations.feather \
  --modality olfactory --neural-steps 100
```

This advances every traced MaleCNS neuron at 1 ms resolution, decodes bilateral
activity from all annotated descending neurons, and applies it to FlyGym at ten
physics steps per neural update. Left/right FlyGym joint velocities are returned to
the MaleCNS proprioceptive populations on every neural step, closing the first
brain–body loop. The aggregate sensory and descending calibrations are explicitly
provisional and must not be interpreted as validated fly behavior.

Run the first replicated locomotion-viability frontier:

```bash
uv run flypareto locomotion-sweep data/processed/malecns-v1.0 \
  --annotations data/raw/body-annotations.feather \
  --output results/locomotion-frontier
```

Each candidate is repeated across three FlyGym controller seeds. A candidate is
feasible only if every replicate stays finite, maintains at least 0.35 mm thorax
height, and remains within 60 degrees of its initial orientation for at least 90%
of sampled steps. Feasible candidates are Pareto-ranked for forward displacement,
upright time, lateral deviation, neural spikes, and mechanical effort. The first
verified 3 × 3 sweep found all 9 candidates feasible and nondominated: increased
propagation bought displacement at an activity and effort cost. Because the assay
lasts only 100 ms and uses a provisional aggregate descending decoder, this is a
locomotion-viability frontier—not yet the intelligence–fitness frontier.

The next assay tests odor-source localization with the odor field sampled at the
left and right funiculi. It stimulates only MaleCNS `ORN_DM1` and `ORN_VA2`
channels, which are implicated in innate vinegar attraction, and uses the
experimentally characterized DNa01/DNa02 steering pairs for turning readout:

```bash
uv run flypareto chemotaxis-sweep data/processed/malecns-v1.0 \
  --annotations data/raw/body-annotations.feather \
  --output results/chemotaxis-frontier
```

The first matched experiment used mirrored targets, two controller seeds, and a
no-odor control. Both odor and control conditions reached only one of four sources,
below the 50% feasibility floor. Odor modestly increased mean closest approach but
increased neural activity by 53% and mechanical effort by 12%. This is a useful
negative baseline: the homogeneous LIF model plus current readout does not yet
support robust chemotaxis, and a lucky trajectory toward one side is explicitly
rejected as locomotor handedness rather than intelligence.

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
- Steering calibration: [Rayshubskiy et al., eLife](https://doi.org/10.7554/eLife.102230)
  and [Braun et al., Cell](https://doi.org/10.1016/j.cell.2024.08.033).
- Attractive olfactory channels: [Semmelhack and Wang, Nature](https://doi.org/10.1038/nature07983).

## License

FlyPareto code is MIT licensed. MaleCNS data retains its original CC BY license and
citation requirements.
