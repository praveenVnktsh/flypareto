# Research roadmap

## R0 — Reproducible substrate

- [x] Official bulk-data downloader
- [x] Full annotated-neuron graph builder
- [x] Neurotransmitter-signed reference dynamics
- [x] Pareto and survival-gate primitives
- [x] End-to-end synthetic smoke experiment
- [x] Record release hashes and dataset summary from the full local download
- [x] Validate a 100-step stimulated full-graph execution
- [x] Add a replicated full-graph sensory-to-motor operating frontier
- [ ] Calibrate full-graph spontaneous and stimulated stability biologically

## R1 — Biological calibration

- [ ] Infer cell-type parameter groups and uncertainty priors
- [ ] Reproduce published sensory-to-descending-neuron interventions
- [ ] Calibrate activity scale against available recordings
- [ ] Quantify sensitivity to transmitter sign and missing receptor information

## R2 — Embodiment

- [x] Pin and smoke-test the current FlyGym 2.1 / MuJoCo physics stack
- [x] Resolve bilateral sensory modalities and descending populations from MaleCNS
- [x] Add a stateful neural-step interface for future closed-loop coupling
- [x] Close the first MaleCNS–FlyGym loop with bilateral joint-velocity feedback
- [x] Establish a replicated, survival-gated locomotion-viability frontier
- [ ] Replace aggregate descending activity with cell-type-specific motor calibration
- [ ] Map visual, olfactory, gustatory, and mechanosensory inputs
- [ ] Map descending and motor populations to action primitives
- [ ] Biologically validate locomotion, looming escape, and food-search survival floors
- [ ] Compare trajectories and ethograms with measured fly distributions

## R3 — Latent-capacity frontier

- [ ] Fix connectome topology
- [ ] Optimize cell-type neural dynamics and calibrated sensory/motor gains
- [ ] Evaluate reversal learning, delayed choice, few-shot adaptation, and transfer
- [ ] Estimate intelligence–fitness–energy–robustness Pareto fronts
- [ ] Re-evaluate every frontier member across held-out environments and seeds

## R4 — Plastic-capacity frontier

- [ ] Add local, dopamine-gated plasticity in known learning circuits
- [ ] Charge explicit activity and plasticity costs
- [ ] Compare short- and long-term memory strategies

## R5 — Evolutionary counterfactuals

- [ ] Add constrained duplication, pruning, and rewiring operators
- [ ] Repeat selection across predation, resource volatility, scarcity, terrain,
  social density, and reproductive-horizon regimes
- [ ] Use female BANC comparisons to separate conserved, dimorphic, and variable wiring
- [ ] Generate preregistered intervention predictions

## Completion criteria

The full project is complete only when the complete MaleCNS is embodied, baseline
survival behavior is validated, a replicated held-out Pareto frontier is produced,
environmental pressure sweeps and uncertainty analyses are complete, and the code,
data provenance, configurations, and results required to reproduce every figure are
public.
