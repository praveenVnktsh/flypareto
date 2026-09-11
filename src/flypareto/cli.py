"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pyarrow import feather

from .calibration import plot_calibration_frontier, run_intervention_calibration
from .chemotaxis import run_chemotaxis_sweep, run_chemotaxis_trial
from .data import download_default_dataset
from .demo import run_demo
from .experiment import run_operating_sweep
from .flygym_bridge import run_closed_loop_smoke, run_physics_smoke
from .graph import Connectome, build_connectome
from .locomotion import run_locomotion_sweep
from .sim import EventDrivenLIF


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flypareto")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download official MaleCNS tables")
    download.add_argument("--data-dir", type=Path, default=Path("data/raw"))

    inspect = commands.add_parser("inspect", help="inspect a Feather table")
    inspect.add_argument("path", type=Path)

    build = commands.add_parser("build", help="build the full simulation graph")
    build.add_argument("--annotations", type=Path, required=True)
    build.add_argument("--neurotransmitters", type=Path, required=True)
    build.add_argument("--connections", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--min-weight", type=int, default=1)

    simulate = commands.add_parser("simulate", help="run a full-graph stability probe")
    simulate.add_argument("graph", type=Path)
    simulate.add_argument("--steps", type=int, default=100)
    simulate.add_argument("--stimulus-count", type=int, default=100)
    simulate.add_argument("--seed", type=int, default=7)

    demo = commands.add_parser("demo", help="run the end-to-end synthetic smoke experiment")
    demo.add_argument("--output", type=Path, default=Path("results/demo"))
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--candidates", type=int, default=48)

    sweep = commands.add_parser(
        "sweep", help="run a neural operating-point frontier on the full connectome"
    )
    sweep.add_argument("graph", type=Path)
    sweep.add_argument("--annotations", type=Path, required=True)
    sweep.add_argument("--output", type=Path, default=Path("results/operating-frontier"))
    sweep.add_argument("--scales", default="0.002,0.004,0.008,0.016")
    sweep.add_argument("--thresholds", default="0.8,1.0,1.2")
    sweep.add_argument("--seeds", default="3,5,7")
    sweep.add_argument("--steps", type=int, default=100)
    sweep.add_argument("--stimulus-count", type=int, default=100)
    sweep.add_argument("--stimulus-amplitude", type=float, default=1.2)

    physics = commands.add_parser(
        "physics-smoke", help="run a headless FlyGym 2.1 locomotion smoke test"
    )
    physics.add_argument("--steps", type=int, default=250)
    physics.add_argument("--left-drive", type=float, default=1.0)
    physics.add_argument("--right-drive", type=float, default=1.0)

    embodied = commands.add_parser(
        "embodied-smoke", help="couple full MaleCNS activity to FlyGym locomotion"
    )
    embodied.add_argument("graph", type=Path)
    embodied.add_argument("--annotations", type=Path, required=True)
    embodied.add_argument("--neural-steps", type=int, default=100)
    embodied.add_argument("--modality", default="olfactory")
    embodied.add_argument("--left-stimulus", type=float, default=1.0)
    embodied.add_argument("--right-stimulus", type=float, default=1.0)
    embodied.add_argument("--stimulus-steps", type=int, default=3)
    embodied.add_argument("--synaptic-scale", type=float, default=0.008)
    embodied.add_argument("--proprioceptive-speed-scale", type=float, default=5.0)
    embodied.add_argument("--decoder-half-saturation", type=float, default=0.01)
    embodied.add_argument("--decoder-smoothing", type=float, default=0.9)
    embodied.add_argument("--seed", type=int, default=0)

    locomotion = commands.add_parser(
        "locomotion-sweep",
        help="run a replicated, survival-gated full-connectome locomotion frontier",
    )
    locomotion.add_argument("graph", type=Path)
    locomotion.add_argument("--annotations", type=Path, required=True)
    locomotion.add_argument("--output", type=Path, default=Path("results/locomotion-frontier"))
    locomotion.add_argument("--synaptic-scales", default="0.004,0.008,0.016")
    locomotion.add_argument("--decoder-half-saturations", default="0.005,0.01,0.02")
    locomotion.add_argument("--proprioceptive-speed-scales", default="5.0")
    locomotion.add_argument("--seeds", default="0,1,2")
    locomotion.add_argument("--neural-steps", type=int, default=100)
    locomotion.add_argument("--modality", default="olfactory")
    locomotion.add_argument("--stimulus-steps", type=int, default=3)

    chemotaxis = commands.add_parser(
        "chemotaxis", help="run closed-loop whole-connectome odor-source localization"
    )
    chemotaxis.add_argument("graph", type=Path)
    chemotaxis.add_argument("--annotations", type=Path, required=True)
    chemotaxis.add_argument("--neural-steps", type=int, default=500)
    chemotaxis.add_argument("--target-forward", type=float, default=2.0)
    chemotaxis.add_argument("--target-lateral", type=float, default=0.75)
    chemotaxis.add_argument("--source-radius", type=float, default=0.5)
    chemotaxis.add_argument("--odor-sigma", type=float, default=1.5)
    chemotaxis.add_argument("--odor-gain", type=float, default=2.0)
    chemotaxis.add_argument("--odor-channels", default="ORN_DM1,ORN_VA2")
    chemotaxis.add_argument("--synaptic-scale", type=float, default=0.008)
    chemotaxis.add_argument("--neural-threshold", type=float, default=1.0)
    chemotaxis.add_argument("--proprioceptive-speed-scale", type=float, default=5.0)
    chemotaxis.add_argument("--decoder-half-saturation", type=float, default=0.01)
    chemotaxis.add_argument("--decoder-smoothing", type=float, default=0.9)
    chemotaxis.add_argument("--decoder-mode", choices=("aggregate", "steering"), default="steering")
    chemotaxis.add_argument("--seed", type=int, default=0)

    chemotaxis_sweep = commands.add_parser(
        "chemotaxis-sweep", help="run a mirrored, replicated odor-source frontier"
    )
    chemotaxis_sweep.add_argument("graph", type=Path)
    chemotaxis_sweep.add_argument("--annotations", type=Path, required=True)
    chemotaxis_sweep.add_argument(
        "--output", type=Path, default=Path("results/chemotaxis-frontier")
    )
    chemotaxis_sweep.add_argument("--synaptic-scales", default="0.004,0.008")
    chemotaxis_sweep.add_argument("--thresholds", default="0.8,1.0")
    chemotaxis_sweep.add_argument("--decoder-half-saturations", default="0.01,0.02")
    chemotaxis_sweep.add_argument("--odor-gains", default="0.0,2.0")
    chemotaxis_sweep.add_argument("--odor-channels", default="ORN_DM1,ORN_VA2")
    chemotaxis_sweep.add_argument("--seeds", default="0,1")
    chemotaxis_sweep.add_argument("--target-laterals", default="-0.75,0.75")
    chemotaxis_sweep.add_argument("--neural-steps", type=int, default=500)
    chemotaxis_sweep.add_argument("--target-forward", type=float, default=2.0)
    chemotaxis_sweep.add_argument("--source-radius", type=float, default=0.5)
    chemotaxis_sweep.add_argument("--odor-sigma", type=float, default=1.5)
    chemotaxis_sweep.add_argument("--minimum-reach-fraction", type=float, default=0.5)
    chemotaxis_sweep.add_argument(
        "--decoder-mode", choices=("aggregate", "steering"), default="steering"
    )

    calibration = commands.add_parser(
        "calibrate-steering", help="build a receptor-to-steering intervention frontier"
    )
    calibration.add_argument("graph", type=Path)
    calibration.add_argument("--annotations", type=Path, required=True)
    calibration.add_argument(
        "--output", type=Path, default=Path("results/steering-calibration")
    )
    calibration.add_argument("--synaptic-scales", default="0.002,0.004,0.008,0.016")
    calibration.add_argument("--thresholds", default="0.8,1.0,1.2")
    calibration.add_argument("--odor-channels", default="ORN_DM1,ORN_VA2")
    calibration.add_argument("--dn-types", default="DNa01,DNa02,DNg13")
    calibration.add_argument("--steps", type=int, default=100)
    calibration.add_argument("--stimulus-steps", type=int, default=10)
    calibration.add_argument("--stimulus-amplitude", type=float, default=1.2)
    calibration.add_argument("--maximum-peak-fraction", type=float, default=0.25)
    calibration.add_argument("--maximum-late-fraction", type=float, default=0.75)

    calibration_plot = commands.add_parser(
        "plot-calibration", help="plot a calibration-points CSV"
    )
    calibration_plot.add_argument("points", type=Path)
    calibration_plot.add_argument("--output", type=Path, required=True)
    return parser


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download":
        outputs = download_default_dataset(args.data_dir)
        print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    elif args.command == "inspect":
        table = feather.read_table(args.path, memory_map=True)
        print(table.schema)
        print(f"rows: {table.num_rows:,}")
    elif args.command == "build":
        graph, metadata = build_connectome(
            args.annotations,
            args.neurotransmitters,
            args.connections,
            min_weight=args.min_weight,
            output_path=args.output,
        )
        print(json.dumps(metadata, indent=2))
    elif args.command == "simulate":
        graph = Connectome.load(args.graph)
        if args.stimulus_count < 1 or args.stimulus_count > graph.neuron_count:
            raise SystemExit("stimulus-count must be between 1 and the neuron count")
        rng = np.random.default_rng(args.seed)
        stimulus_nodes = rng.choice(graph.neuron_count, args.stimulus_count, replace=False)
        external = np.zeros((args.steps, graph.neuron_count), dtype=np.float32)
        external[:3, stimulus_nodes] = 1.2
        result = EventDrivenLIF(graph).run(args.steps, external)
        summary = {
            "neurons": graph.neuron_count,
            "edges": graph.edge_count,
            "steps": args.steps,
            "total_spikes": int(result.population_spikes.sum()),
            "active_neurons": int(np.count_nonzero(result.spike_counts)),
            "peak_population_spikes": int(result.population_spikes.max()),
        }
        print(json.dumps(summary, indent=2))
    elif args.command == "demo":
        feasible, frontier = run_demo(args.output, args.seed, args.candidates)
        print(json.dumps({"feasible": feasible, "frontier": frontier,
                          "output": str(args.output)}, indent=2))
    elif args.command == "sweep":
        graph = Connectome.load(args.graph)
        candidates, frontier = run_operating_sweep(
            graph,
            args.annotations,
            args.output,
            scales=_floats(args.scales),
            thresholds=_floats(args.thresholds),
            seeds=_ints(args.seeds),
            steps=args.steps,
            stimulus_count=args.stimulus_count,
            stimulus_amplitude=args.stimulus_amplitude,
        )
        print(json.dumps({"candidates": candidates, "frontier": frontier,
                          "output": str(args.output)}, indent=2))
    elif args.command == "physics-smoke":
        result = run_physics_smoke(
            steps=args.steps,
            descending_drive=(args.left_drive, args.right_drive),
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif args.command == "embodied-smoke":
        graph = Connectome.load(args.graph)
        result = run_closed_loop_smoke(
            graph,
            args.annotations,
            neural_steps=args.neural_steps,
            modality=args.modality,
            left_stimulus=args.left_stimulus,
            right_stimulus=args.right_stimulus,
            stimulus_steps=args.stimulus_steps,
            synaptic_scale=args.synaptic_scale,
            proprioceptive_speed_scale=args.proprioceptive_speed_scale,
            decoder_half_saturation=args.decoder_half_saturation,
            decoder_smoothing=args.decoder_smoothing,
            seed=args.seed,
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif args.command == "locomotion-sweep":
        graph = Connectome.load(args.graph)
        candidates, frontier = run_locomotion_sweep(
            graph,
            args.annotations,
            args.output,
            synaptic_scales=_floats(args.synaptic_scales),
            decoder_half_saturations=_floats(args.decoder_half_saturations),
            proprioceptive_speed_scales=_floats(args.proprioceptive_speed_scales),
            seeds=_ints(args.seeds),
            neural_steps=args.neural_steps,
            modality=args.modality,
            stimulus_steps=args.stimulus_steps,
        )
        print(json.dumps({"candidates": candidates, "frontier": frontier,
                          "output": str(args.output)}, indent=2))
    elif args.command == "chemotaxis":
        graph = Connectome.load(args.graph)
        result = run_chemotaxis_trial(
            graph,
            args.annotations,
            neural_steps=args.neural_steps,
            target_offset_mm=(args.target_forward, args.target_lateral),
            source_radius_mm=args.source_radius,
            odor_sigma_mm=args.odor_sigma,
            odor_gain=args.odor_gain,
            odor_channels=tuple(item for item in args.odor_channels.split(",") if item),
            synaptic_scale=args.synaptic_scale,
            neural_threshold=args.neural_threshold,
            proprioceptive_speed_scale=args.proprioceptive_speed_scale,
            decoder_half_saturation=args.decoder_half_saturation,
            decoder_smoothing=args.decoder_smoothing,
            decoder_mode=args.decoder_mode,
            seed=args.seed,
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif args.command == "chemotaxis-sweep":
        graph = Connectome.load(args.graph)
        candidates, frontier = run_chemotaxis_sweep(
            graph,
            args.annotations,
            args.output,
            synaptic_scales=_floats(args.synaptic_scales),
            thresholds=_floats(args.thresholds),
            decoder_half_saturations=_floats(args.decoder_half_saturations),
            odor_gains=_floats(args.odor_gains),
            seeds=_ints(args.seeds),
            target_laterals_mm=_floats(args.target_laterals),
            neural_steps=args.neural_steps,
            target_forward_mm=args.target_forward,
            source_radius_mm=args.source_radius,
            odor_sigma_mm=args.odor_sigma,
            odor_channels=tuple(item for item in args.odor_channels.split(",") if item),
            minimum_reach_fraction=args.minimum_reach_fraction,
            decoder_mode=args.decoder_mode,
        )
        print(json.dumps({"candidates": candidates, "frontier": frontier,
                          "output": str(args.output)}, indent=2))
    elif args.command == "calibrate-steering":
        graph = Connectome.load(args.graph)
        candidates, frontier = run_intervention_calibration(
            graph,
            args.annotations,
            args.output,
            synaptic_scales=_floats(args.synaptic_scales),
            thresholds=_floats(args.thresholds),
            odor_channels=tuple(item for item in args.odor_channels.split(",") if item),
            dn_types=tuple(item for item in args.dn_types.split(",") if item),
            steps=args.steps,
            stimulus_steps=args.stimulus_steps,
            stimulus_amplitude=args.stimulus_amplitude,
            maximum_peak_fraction=args.maximum_peak_fraction,
            maximum_late_fraction=args.maximum_late_fraction,
        )
        print(json.dumps({"candidates": candidates, "frontier": frontier,
                          "output": str(args.output)}, indent=2))
    elif args.command == "plot-calibration":
        plot_calibration_frontier(args.points, args.output)
        print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
