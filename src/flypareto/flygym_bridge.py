"""FlyGym 2.1 embodiment bridge.

Imports are local so the core connectome tooling remains usable without the optional
MuJoCo environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .graph import Connectome
from .interface import BilateralNeuralInterface, DescendingDriveDecoder, MaleCNSPopulations
from .sim import EventDrivenLIF, LIFConfig


@dataclass(frozen=True)
class PhysicsSmokeResult:
    steps: int
    timestep_seconds: float
    initial_thorax_mm: tuple[float, float, float]
    final_thorax_mm: tuple[float, float, float]
    displacement_mm: tuple[float, float, float]
    finite: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClosedLoopSmokeResult:
    neural_steps: int
    physics_steps: int
    total_spikes: int
    descending_spikes: int
    mean_drive: tuple[float, float]
    drive_range: tuple[float, float]
    mean_proprioception: tuple[float, float]
    displacement_mm: tuple[float, float, float]
    min_thorax_height_mm: float
    upright_fraction: float
    neural_energy_proxy: float
    mechanical_effort_proxy: float
    locomotion_viable: bool
    finite: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _FlyGymPhysics:
    """Small internal wrapper around the version-pinned FlyGym 2.1 API."""

    def __init__(self, seed: int = 0) -> None:
        try:
            from flygym import Simulation
            from flygym.anatomy import BodySegment, ContactBodiesPreset
            from flygym.compose import FlatGroundWorld
            from flygym.compose.fly import ActuatorType
            from flygym.utils.math import Rotation3D
            from flygym_demo.complex_terrain import (
                HybridControllerObservation,
                HybridTurningController,
                LocomotionAction,
                PreprogrammedSteps,
                apply_locomotion_action,
                make_locomotion_fly,
            )
        except ImportError as error:  # pragma: no cover - optional install
            raise ImportError(
                "FlyGym is optional; install FlyPareto with the 'embodiment' extra"
            ) from error

        self._observation_type = HybridControllerObservation
        self._apply_action = apply_locomotion_action
        self.fly = make_locomotion_fly(name="flypareto", add_adhesion=True, colorize=False)
        world = FlatGroundWorld()
        world.add_fly(
            self.fly,
            [0, 0, 0.8],
            Rotation3D("quat", [1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )
        self.simulation = Simulation(world)
        programmed_steps = PreprogrammedSteps()
        dof_order = self.fly.get_actuated_jointdofs_order("position")
        all_dof_order = self.fly.get_jointdofs_order()
        self._position_joint_indices = np.asarray(
            [all_dof_order.index(joint) for joint in dof_order], dtype=np.intp
        )
        self._left_joint_indices = np.asarray(
            [
                full_index
                for full_index, joint in zip(self._position_joint_indices, dof_order)
                if joint.child.name.rsplit("/", 1)[-1].startswith("l")
            ],
            dtype=np.intp,
        )
        self._right_joint_indices = np.asarray(
            [
                full_index
                for full_index, joint in zip(self._position_joint_indices, dof_order)
                if joint.child.name.rsplit("/", 1)[-1].startswith("r")
            ],
            dtype=np.intp,
        )
        if not self._left_joint_indices.size or not self._right_joint_indices.size:
            raise RuntimeError("could not resolve bilateral FlyGym joint indices")
        self.controller = HybridTurningController(
            timestep=self.simulation.timestep,
            preprogrammed_steps=programmed_steps,
            output_dof_order=dof_order,
        )
        self.simulation.reset()
        self.controller.reset(seed=seed)
        initial_action = LocomotionAction(
            joint_angles=programmed_steps.default_pose_by_dof_order(dof_order),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        self._apply_action(self.simulation, self.fly.name, initial_action)
        self.simulation.warmup()
        body_order = self.fly.get_bodysegs_order()
        self._thorax_index = body_order.index(BodySegment("c_thorax"))
        self._antenna_indices = np.asarray(
            [
                body_order.index(BodySegment("l_funiculus")),
                body_order.index(BodySegment("r_funiculus")),
            ],
            dtype=np.intp,
        )
        self._position_actuator_type = ActuatorType.POSITION
        self._reference_thorax_quaternion = self.thorax_quaternion

    @property
    def timestep(self) -> float:
        return float(self.simulation.timestep)

    @property
    def thorax_position(self) -> np.ndarray:
        return self.simulation.get_body_positions(self.fly.name)[self._thorax_index].astype(float)

    @property
    def antenna_positions(self) -> np.ndarray:
        """World positions of the left and right funiculi in millimeters."""
        return self.simulation.get_body_positions(self.fly.name)[self._antenna_indices].astype(float)

    @property
    def bilateral_joint_speed(self) -> np.ndarray:
        """Mean absolute angular velocity of left and right actuated joints."""
        velocities = np.abs(self.simulation.get_joint_velocities(self.fly.name))
        return np.asarray(
            [
                np.mean(velocities[self._left_joint_indices]),
                np.mean(velocities[self._right_joint_indices]),
            ],
            dtype=np.float32,
        )

    @property
    def thorax_quaternion(self) -> np.ndarray:
        return self.simulation.get_body_rotations(self.fly.name)[self._thorax_index].astype(float)

    @property
    def orientation_deviation_radians(self) -> float:
        quaternion = self.thorax_quaternion
        similarity = np.clip(abs(np.dot(quaternion, self._reference_thorax_quaternion)), 0, 1)
        return float(2 * np.arccos(similarity))

    def step(self, drive: np.ndarray) -> float:
        observation = self._observation_type.from_sim(self.simulation, self.fly.name)
        action = self.controller.step(drive, observation)
        self._apply_action(self.simulation, self.fly.name, action)
        force = self.simulation.get_actuator_forces(
            self.fly.name, self._position_actuator_type
        )
        velocity = self.simulation.get_joint_velocities(self.fly.name)[
            self._position_joint_indices
        ]
        effort = float(np.sum(np.abs(force * velocity)) * self.timestep)
        self.simulation.step()
        return effort


def run_physics_smoke(
    steps: int = 250,
    descending_drive: tuple[float, float] = (1.0, 1.0),
) -> PhysicsSmokeResult:
    """Run the official FlyGym 2.1 hybrid controller without rendering."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    physics = _FlyGymPhysics()
    initial = physics.thorax_position
    drive = np.asarray(descending_drive, dtype=float)
    if drive.shape != (2,) or not np.all(np.isfinite(drive)):
        raise ValueError("descending_drive must contain two finite values")

    for _ in range(steps):
        physics.step(drive)

    final = physics.thorax_position
    displacement = final - initial
    return PhysicsSmokeResult(
        steps=steps,
        timestep_seconds=physics.timestep,
        initial_thorax_mm=tuple(initial.tolist()),
        final_thorax_mm=tuple(final.tolist()),
        displacement_mm=tuple(displacement.tolist()),
        finite=bool(np.all(np.isfinite(final))),
    )


def run_closed_loop_smoke(
    graph: Connectome,
    annotations_path: Path,
    neural_steps: int = 100,
    modality: str = "olfactory",
    left_stimulus: float = 1.0,
    right_stimulus: float = 1.0,
    stimulus_steps: int = 3,
    synaptic_scale: float = 0.008,
    proprioceptive_speed_scale: float = 5.0,
    decoder_half_saturation: float = 0.01,
    decoder_smoothing: float = 0.9,
    seed: int = 0,
) -> ClosedLoopSmokeResult:
    """Drive FlyGym locomotion from full-MaleCNS descending activity.

    This is an integration test, not a validated behavior model. The mapping from
    aggregate descending firing to FlyGym's two drive channels remains provisional.
    """
    if neural_steps <= 0:
        raise ValueError("neural_steps must be positive")
    if proprioceptive_speed_scale <= 0:
        raise ValueError("proprioceptive_speed_scale must be positive")
    physics = _FlyGymPhysics(seed=seed)
    populations = MaleCNSPopulations.from_annotations(graph, annotations_path)
    interface = BilateralNeuralInterface(graph, populations)
    decoder = DescendingDriveDecoder(
        half_saturation=decoder_half_saturation,
        smoothing=decoder_smoothing,
    )
    neural = EventDrivenLIF(graph, LIFConfig(synaptic_scale=synaptic_scale))
    neural.reset()
    physics_per_neural = max(1, round((neural.config.dt_ms / 1000) / physics.timestep))
    initial = physics.thorax_position
    drives: list[np.ndarray] = []
    proprioception: list[np.ndarray] = []
    total_spikes = 0
    descending_spikes = 0
    mechanical_effort = 0.0
    heights: list[float] = []
    upright: list[bool] = []

    for neural_step in range(neural_steps):
        joint_speed = physics.bilateral_joint_speed
        normalized_proprioception = np.tanh(joint_speed / proprioceptive_speed_scale)
        proprioception.append(normalized_proprioception)
        external = interface.encode(
            "mechanosensory_proprioceptive",
            float(normalized_proprioception[0]),
            float(normalized_proprioception[1]),
        )
        if neural_step < stimulus_steps:
            external += interface.encode(modality, left_stimulus, right_stimulus)
        spikes = neural.step(external)
        total_spikes += int(np.count_nonzero(spikes))
        bilateral = interface.descending_activity(spikes)
        descending_spikes += int(np.count_nonzero(spikes[populations.descending.left]))
        descending_spikes += int(np.count_nonzero(spikes[populations.descending.right]))
        drive = decoder.update(bilateral)
        drives.append(drive.copy())
        for _ in range(physics_per_neural):
            mechanical_effort += physics.step(drive)
        heights.append(float(physics.thorax_position[2]))
        upright.append(physics.orientation_deviation_radians < np.deg2rad(60))

    final = physics.thorax_position
    drive_array = np.asarray(drives)
    proprioception_array = np.asarray(proprioception)
    finite = bool(
        np.all(np.isfinite(final))
        and np.all(np.isfinite(drive_array))
        and np.isfinite(mechanical_effort)
    )
    min_height = min(heights)
    upright_fraction = float(np.mean(upright))
    locomotion_viable = finite and min_height >= 0.35 and upright_fraction >= 0.9
    return ClosedLoopSmokeResult(
        neural_steps=neural_steps,
        physics_steps=neural_steps * physics_per_neural,
        total_spikes=total_spikes,
        descending_spikes=descending_spikes,
        mean_drive=tuple(np.mean(drive_array, axis=0).tolist()),
        drive_range=(float(np.min(drive_array)), float(np.max(drive_array))),
        mean_proprioception=tuple(np.mean(proprioception_array, axis=0).tolist()),
        displacement_mm=tuple((final - initial).tolist()),
        min_thorax_height_mm=min_height,
        upright_fraction=upright_fraction,
        neural_energy_proxy=float(total_spikes),
        mechanical_effort_proxy=mechanical_effort,
        locomotion_viable=locomotion_viable,
        finite=finite,
    )
