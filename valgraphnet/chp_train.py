"""GPU-only long-horizon training and validation for CHP-GNS."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from valgraphnet.chp_model import (
    CHPGNS,
    CHPState,
    CHPStatic,
    PhysicalStep,
    build_case_hierarchy,
    build_chp_static,
    radius_contact_pairs,
)
from valgraphnet.config import get_cfg
from valgraphnet.data import ValveGraphDataset
from valgraphnet.data.case import ValveCase
from valgraphnet.mechanics import deformation_gradient, invariants
from valgraphnet.physical_evaluation import validate_reference_protocol
from valgraphnet.stress_transform import AsinhStressTransform, robust_stress_loss


ROLLOUT_METRIC_KEYS = (
    "moving_displacement_relative_rmse",
    "final_displacement_relative_rmse",
    "stress_relative_rmse",
    "stress_p95_relative_rmse",
)


@dataclass
class CHPNormalizers:
    """Physical scales used by the state and heavy-tailed stress losses."""

    displacement_scale: torch.Tensor
    velocity_scale: torch.Tensor
    stress: AsinhStressTransform
    acceleration_scale: torch.Tensor | None = None
    cell_stress: AsinhStressTransform | None = None

    def __post_init__(self) -> None:
        if self.acceleration_scale is None:
            self.acceleration_scale = self.velocity_scale.detach().clone()

    def to(self, device: torch.device | str) -> "CHPNormalizers":
        return CHPNormalizers(
            displacement_scale=self.displacement_scale.to(device),
            velocity_scale=self.velocity_scale.to(device),
            stress=self.stress.to(device),
            acceleration_scale=self.acceleration_scale.to(device),
            cell_stress=(
                self.cell_stress.to(device) if self.cell_stress is not None else None
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "displacement_scale": self.displacement_scale.cpu(),
            "velocity_scale": self.velocity_scale.cpu(),
            "acceleration_scale": self.acceleration_scale.cpu(),
            "stress": self.stress.state_dict(),
            "cell_stress": (
                self.cell_stress.state_dict() if self.cell_stress is not None else None
            ),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CHPNormalizers":
        velocity_scale = state["velocity_scale"].float()
        return cls(
            displacement_scale=state["displacement_scale"].float(),
            velocity_scale=velocity_scale,
            stress=AsinhStressTransform.from_state_dict(state["stress"]),
            acceleration_scale=state.get(
                "acceleration_scale", velocity_scale
            ).float(),
            cell_stress=(
                AsinhStressTransform.from_state_dict(state["cell_stress"])
                if state.get("cell_stress") is not None
                else None
            ),
        )


@dataclass
class CHPDeviceCase:
    """One complete trajectory and its mechanics tensors resident on CUDA."""

    static: CHPStatic
    times: torch.Tensor
    pressure: torch.Tensor
    displacement: torch.Tensor
    velocity: torch.Tensor
    stress: torch.Tensor
    cell_stress: torch.Tensor
    normals: torch.Tensor
    nodal_area: torch.Tensor
    pressure_mask: torch.Tensor
    time_fraction: torch.Tensor


class CHPCaseCache:
    """Small LRU of full trajectories with a reusable CPU topology cache."""

    def __init__(
        self,
        cases: list[ValveCase],
        device: torch.device,
        *,
        material_dim: int,
        cache_size: int = 3,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CHPCaseCache is GPU-only")
        self.cases = cases
        self.device = device
        self.material_dim = int(material_dim)
        self.cache_size = max(int(cache_size), 1)
        self._hierarchies: dict[int, Any] = {}
        self._gpu: OrderedDict[tuple[int, int | None, int | None], CHPDeviceCase] = (
            OrderedDict()
        )

    @staticmethod
    def _tensor(value: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.array(value, copy=True)).to(
            device=device, dtype=dtype, non_blocking=True
        )

    def get(self, case_index: int) -> CHPDeviceCase:
        return self._get(case_index, None, None)

    def get_slice(self, case_index: int, start: int, stop: int) -> CHPDeviceCase:
        """Copy only the K+1 frames needed by one training rollout."""

        return self._get(case_index, int(start), int(stop))

    def _get(
        self, case_index: int, start: int | None, stop: int | None
    ) -> CHPDeviceCase:
        case_index = int(case_index)
        key = (case_index, start, stop)
        cached = self._gpu.pop(key, None)
        if cached is not None:
            self._gpu[key] = cached
            return cached
        case = self.cases[case_index]
        hierarchy = self._hierarchies.get(case_index)
        if hierarchy is None:
            hierarchy = build_case_hierarchy(case)
            self._hierarchies[case_index] = hierarchy
        static = build_chp_static(
            case,
            self.device,
            hierarchy=hierarchy,
            material_dim=self.material_dim,
        )
        frame_slice = slice(start, stop)
        times = np.asarray(case.times[frame_slice], dtype=np.float32)
        span = max(float(case.times[-1] - case.times[0]), 1.0e-8)
        phase = (times - float(case.times[0])) / span
        cached = CHPDeviceCase(
            static=static,
            times=self._tensor(times, self.device, torch.float32),
            pressure=self._tensor(case.pressure[frame_slice], self.device, torch.float32),
            displacement=self._tensor(
                case.displacement[frame_slice], self.device, torch.float32
            ),
            velocity=self._tensor(case.velocity[frame_slice], self.device, torch.float32),
            stress=self._tensor(case.stress[frame_slice], self.device, torch.float32),
            cell_stress=self._tensor(
                case.cell_stress[frame_slice], self.device, torch.float32
            ),
            normals=self._tensor(case.normals, self.device, torch.float32),
            nodal_area=self._tensor(case.nodal_area, self.device, torch.float32),
            pressure_mask=self._tensor(case.pressure_mask, self.device, torch.bool),
            time_fraction=self._tensor(phase, self.device, torch.float32),
        )
        self._gpu[key] = cached
        while len(self._gpu) > self.cache_size:
            self._gpu.popitem(last=False)
        return cached

    def clear_gpu(self) -> None:
        self._gpu.clear()
        torch.cuda.empty_cache()


def curriculum_horizon(epoch: int, stages: Iterable[dict[str, int]] | None = None) -> int:
    """Return the fixed K=1,2,4,8,16 curriculum horizon for an epoch."""

    if epoch < 1:
        raise ValueError("epoch must be positive")
    schedule = list(stages or (
        {"horizon": 1, "epochs": 4},
        {"horizon": 2, "epochs": 3},
        {"horizon": 4, "epochs": 3},
        {"horizon": 8, "epochs": 3},
        {"horizon": 16, "epochs": 3},
    ))
    end = 0
    for stage in schedule:
        end += int(stage["epochs"])
        if epoch <= end:
            return int(stage["horizon"])
    return int(schedule[-1]["horizon"])


def select_rollout_start(
    num_steps: int,
    horizon: int,
    stress_scores: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, str]:
    """Select 50% uniform, 25% late, or 25% high-stress starts."""

    max_start = int(num_steps) - 1 - int(horizon)
    if max_start < 0:
        raise ValueError("trajectory is shorter than the requested horizon")
    draw = float(rng.random())
    if draw < 0.50:
        return int(rng.integers(0, max_start + 1)), "uniform"
    if draw < 0.75:
        lower = max_start // 2
        return int(rng.integers(lower, max_start + 1)), "late"
    valid_scores = np.asarray(stress_scores[: max_start + 1], dtype=np.float64)
    finite = np.isfinite(valid_scores)
    if not finite.any():
        return int(rng.integers(0, max_start + 1)), "uniform-fallback"
    threshold = np.quantile(valid_scores[finite], 0.90)
    candidates = np.flatnonzero(finite & (valid_scores >= threshold))
    if candidates.size == 0:
        return int(rng.integers(0, max_start + 1)), "uniform-fallback"
    return int(rng.choice(candidates)), "stress"


def fit_chp_normalizers(
    cases: list[ValveCase],
    *,
    max_cases: int = 128,
    frames_per_case: int = 8,
    nodes_per_frame: int = 256,
) -> CHPNormalizers:
    """Fit deterministic bounded statistics without clipping physical labels."""

    if not cases:
        raise ValueError("normalizer fitting requires at least one case")
    case_count = min(len(cases), max(int(max_cases), 1))
    case_indices = np.linspace(0, len(cases) - 1, case_count).round().astype(int)
    delta_samples: list[torch.Tensor] = []
    velocity_samples: list[torch.Tensor] = []
    acceleration_samples: list[torch.Tensor] = []
    stress_samples: list[torch.Tensor] = []
    cell_stress_samples: list[torch.Tensor] = []
    for case_index in case_indices:
        case = cases[int(case_index)]
        frame_count = min(max(int(frames_per_case), 1), case.num_steps - 1)
        frames = np.linspace(0, case.num_steps - 2, frame_count).round().astype(int)
        node_count = min(max(int(nodes_per_frame), 1), case.num_nodes)
        nodes = np.linspace(0, case.num_nodes - 1, node_count).round().astype(int)
        raw_cell_stress = np.asarray(
            getattr(case, "cell_stress", np.empty((case.num_steps, 0, 0))),
            dtype=np.float32,
        )
        has_cell_stress = _case_has_cell_stress_tensor(case)
        if has_cell_stress:
            cell_count = min(max(int(nodes_per_frame), 1), raw_cell_stress.shape[1])
            cells = (
                np.linspace(0, raw_cell_stress.shape[1] - 1, cell_count)
                .round()
                .astype(int)
            )
        moving = ~(
            np.asarray(
                getattr(case, "fixed_mask", np.zeros(case.num_nodes, dtype=bool)),
                dtype=bool,
            )
            | np.asarray(
                getattr(
                    case,
                    "prescribed_mask",
                    np.zeros(case.num_nodes, dtype=bool),
                ),
                dtype=bool,
            )
        )
        free_nodes = np.flatnonzero(moving)
        acceleration_node_count = min(node_count, free_nodes.size)
        acceleration_nodes = (
            free_nodes[
                np.linspace(0, free_nodes.size - 1, acceleration_node_count)
                .round()
                .astype(int)
            ]
            if acceleration_node_count
            else np.empty(0, dtype=np.int64)
        )
        for frame in frames:
            delta = case.displacement[frame + 1, nodes] - case.displacement[frame, nodes]
            delta_samples.append(torch.from_numpy(np.array(delta, copy=True)).float())
            velocity_samples.append(
                torch.from_numpy(np.array(case.velocity[frame + 1, nodes], copy=True)).float()
            )
            if acceleration_nodes.size:
                if hasattr(case, "times"):
                    dt = float(case.times[frame + 1] - case.times[frame])
                else:
                    dt = 1.0
                if not math.isfinite(dt) or dt <= 0.0:
                    raise ValueError("trajectory times must be finite and increasing")
                acceleration_samples.append(
                    torch.from_numpy(
                        np.array(
                            (
                                case.velocity[frame + 1, acceleration_nodes]
                                - case.velocity[frame, acceleration_nodes]
                            )
                            / dt,
                            copy=True,
                        )
                    ).float()
                )
            if case.stress_dim:
                stress_samples.append(
                    torch.from_numpy(
                        np.array(case.stress[frame + 1, nodes, :1], copy=True)
                    ).float()
                )
            if has_cell_stress:
                cell_stress_samples.append(
                    torch.from_numpy(
                        np.array(raw_cell_stress[frame + 1, cells, :], copy=True)
                    ).float()
                )
    if not stress_samples:
        raise ValueError("CHP-GNS requires at least one nodal stress label channel")
    if not acceleration_samples:
        raise ValueError("CHP-GNS requires at least one moving acceleration sample")
    delta_values = torch.cat(delta_samples, dim=0)
    velocity_values = torch.cat(velocity_samples, dim=0)
    acceleration_values = torch.cat(acceleration_samples, dim=0)
    displacement_scale = delta_values.square().mean(0).sqrt().clamp_min(1.0e-6)
    velocity_scale = velocity_values.square().mean(0).sqrt().clamp_min(1.0e-6)
    # A scalar scale preserves rotational invariance of the acceleration loss.
    acceleration_scale = acceleration_values.square().mean().sqrt().clamp_min(1.0e-8)
    stress = AsinhStressTransform.fit(stress_samples)
    cell_stress = (
        AsinhStressTransform.fit(cell_stress_samples)
        if cell_stress_samples
        else None
    )
    return CHPNormalizers(
        displacement_scale=displacement_scale,
        velocity_scale=velocity_scale,
        stress=stress,
        acceleration_scale=acceleration_scale,
        cell_stress=cell_stress,
    )


def _case_has_cell_stress_tensor(case: ValveCase | Any) -> bool:
    """Return whether a case carries canonical ``[T, M, 6]`` stress labels."""

    cell_stress = np.asarray(
        getattr(case, "cell_stress", np.empty((0, 0, 0))),
    )
    num_steps = int(getattr(case, "num_steps", cell_stress.shape[0] or 0))
    if hasattr(case, "num_cells"):
        num_cells = int(case.num_cells)
    else:
        cells = np.asarray(getattr(case, "cells", np.empty((0, 4))))
        num_cells = int(cells.shape[0])
    return cell_stress.shape == (num_steps, num_cells, 6) and num_cells > 0


def _trajectory_has_cell_stress_tensor(trajectory: CHPDeviceCase) -> bool:
    return (
        trajectory.cell_stress.ndim == 3
        and trajectory.cell_stress.shape[0] == trajectory.times.shape[0]
        and trajectory.cell_stress.shape[1] == trajectory.static.cells.shape[0]
        and trajectory.cell_stress.shape[2] == 6
        and trajectory.cell_stress.shape[1] > 0
    )


def stress_frame_scores(case: ValveCase) -> np.ndarray:
    """RMS physical stress by frame for curriculum start stratification."""

    if not case.stress_dim:
        return np.zeros(case.num_steps, dtype=np.float32)
    prescribed = getattr(
        case, "prescribed_mask", np.zeros(case.num_nodes, dtype=bool)
    )
    stress_mask = ~np.asarray(prescribed, dtype=bool)
    if not stress_mask.any():
        stress_mask = np.ones(case.num_nodes, dtype=bool)
    stress = np.asarray(case.stress[:, stress_mask, :1], dtype=np.float64)
    return np.sqrt(np.mean(np.square(stress), axis=(1, 2))).astype(np.float32)


def acceleration_frame_scores(case: ValveCase) -> np.ndarray:
    """Free-node finite-difference acceleration RMS for active-frame sampling."""

    moving = ~(
        np.asarray(case.fixed_mask, dtype=bool)
        | np.asarray(case.prescribed_mask, dtype=bool)
    )
    scores = np.zeros(case.num_steps, dtype=np.float32)
    if not moving.any() or case.num_steps < 2:
        return scores
    dt = np.diff(np.asarray(case.times, dtype=np.float64))
    if not np.isfinite(dt).all() or np.any(dt <= 0.0):
        raise ValueError("trajectory times must be finite and increasing")
    acceleration = np.diff(
        np.asarray(case.velocity[:, moving], dtype=np.float64), axis=0
    ) / dt[:, None, None]
    scores[:-1] = np.sqrt(np.mean(np.square(acceleration), axis=(1, 2))).astype(
        np.float32
    )
    return scores


def build_acceleration_sampling_scores(
    cases: list[ValveCase], cache_path: str | Path | None = None
) -> list[np.ndarray]:
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("case_ids") == [case.case_id for case in cases]:
                return [np.asarray(value, dtype=np.float32) for value in saved["scores"]]
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
    scores = [
        acceleration_frame_scores(case)
        for case in tqdm(cases, desc="acceleration-index")
    ]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"case_ids": [case.case_id for case in cases], "scores": scores}, path
        )
    return scores


@torch.no_grad()
def geometry_safe_state_noise(
    static: CHPStatic,
    position: torch.Tensor,
    standard_deviation: float,
    *,
    smoothing_steps: int = 4,
    minimum_j: float = 0.2,
    max_backtracks: int = 8,
    maximum_invariant_growth: float = 2.0,
) -> tuple[torch.Tensor, float]:
    """Project 0.003 GPU noise onto the orientation-preserving state domain."""

    moving = ~(static.fixed_mask | static.prescribed_mask)
    if standard_deviation <= 0.0 or not bool(moving.any().item()):
        return torch.zeros_like(position), 0.0
    noise = torch.randn_like(position)
    src, dst = static.mesh_edge_index.long()
    for _ in range(max(int(smoothing_steps), 0)):
        aggregate = torch.zeros_like(noise)
        aggregate.index_add_(0, dst, noise[src])
        degree = torch.bincount(dst, minlength=position.shape[0]).to(noise.dtype)
        neighbor_mean = aggregate / degree.clamp_min(1.0)[:, None]
        noise = 0.25 * noise + 0.75 * neighbor_mean
        noise = noise * moving[:, None]
    rms = noise[moving].square().mean().sqrt().clamp_min(1.0e-12)
    noise = noise * (float(standard_deviation) / rms)
    base_state = invariants(
        deformation_gradient(position, static.cells, static.dm_inv)
    )
    base_j = base_state.j.min()
    admissible_j = min(
        float(minimum_j),
        max(0.5 * float(base_j.item()), 1.0e-4),
    )
    admissible_i1 = max(
        float(base_state.i1_bar.max().item()) * float(maximum_invariant_growth),
        100.0,
    )
    admissible_i2 = max(
        float(base_state.i2_bar.max().item()) * float(maximum_invariant_growth),
        1_000.0,
    )
    scale = 1.0
    for _ in range(max(int(max_backtracks), 0) + 1):
        candidate = position + scale * noise
        candidate_state = invariants(
            deformation_gradient(candidate, static.cells, static.dm_inv)
        )
        admissible = (
            (candidate_state.j.min() >= admissible_j)
            & (candidate_state.i1_bar.max() <= admissible_i1)
            & (candidate_state.i2_bar.max() <= admissible_i2)
        )
        if bool(admissible.item()):
            return scale * noise, float(scale)
        scale *= 0.5
    return torch.zeros_like(position), 0.0


@torch.no_grad()
def training_state_admissibility(
    static: CHPStatic,
    position: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[bool, float, float]:
    """Return potential-domain validity and the limiting invariants."""

    state = invariants(
        deformation_gradient(position, static.cells, static.dm_inv)
    )
    minimum_j = float(get_cfg(cfg, "training.minimum_start_j", 1.0e-2))
    maximum_i2 = float(get_cfg(cfg, "training.maximum_start_i2_bar", 1.0e5))
    minimum_observed_j = state.j.min()
    maximum_observed_i2 = state.i2_bar.max()
    valid = (
        torch.isfinite(state.j).all()
        & torch.isfinite(state.i2_bar).all()
        & (minimum_observed_j >= minimum_j)
        & (maximum_observed_i2 <= maximum_i2)
    )
    return (
        bool(valid.item()),
        float(minimum_observed_j.item()),
        float(maximum_observed_i2.item()),
    )


@torch.no_grad()
def is_admissible_training_state(
    static: CHPStatic,
    position: torch.Tensor,
    cfg: dict[str, Any],
) -> bool:
    """Reject source frames that already violate the potential domain."""

    return training_state_admissibility(static, position, cfg)[0]


@torch.no_grad()
def is_admissible_training_window(
    trajectory: CHPDeviceCase,
    cfg: dict[str, Any],
) -> bool:
    """Require every exact state in a rollout window to lie in GL+(3)."""

    position = (
        trajectory.static.reference_position[None]
        + trajectory.displacement
    )
    return training_state_admissibility(trajectory.static, position, cfg)[0]


def build_sampling_scores(
    cases: list[ValveCase], cache_path: str | Path | None = None
) -> list[np.ndarray]:
    """Build or restore per-frame stress scores with case-id validation."""

    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("case_ids") == [case.case_id for case in cases]:
                return [np.asarray(value, dtype=np.float32) for value in saved["scores"]]
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
    scores = [stress_frame_scores(case) for case in tqdm(cases, desc="stress-index")]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"case_ids": [case.case_id for case in cases], "scores": scores}, path
        )
    return scores


def external_force_at(
    trajectory: CHPDeviceCase,
    step: int,
    pressure_sign: float,
) -> torch.Tensor:
    pressure = float(pressure_sign) * trajectory.pressure[int(step)]
    return (
        pressure
        * trajectory.normals
        * trajectory.nodal_area[:, None]
        * trajectory.pressure_mask[:, None].to(trajectory.normals.dtype)
    )


def contact_pairs_at(
    trajectory: CHPDeviceCase,
    state: CHPState,
    cfg: dict[str, Any],
) -> torch.Tensor:
    if not bool(get_cfg(cfg, "contact.enabled", True)):
        return torch.zeros((2, 0), dtype=torch.long, device=state.position.device)
    return radius_contact_pairs(
        state.position,
        trajectory.static.mesh_edge_index,
        trajectory.static.fixed_mask,
        float(get_cfg(cfg, "contact.radius", 0.03)),
        max_neighbors=int(get_cfg(cfg, "contact.max_neighbors", 32)),
        prescribed_mask=trajectory.static.prescribed_mask,
        surface_mask=trajectory.static.contact_surface_mask,
    )


def chp_step_loss(
    output: PhysicalStep,
    input_state: CHPState,
    trajectory: CHPDeviceCase,
    next_step: int,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dimensionless state, stress, work, contact, residual, and J loss."""

    static = trajectory.static
    exact_position = static.reference_position + trajectory.displacement[next_step]
    step_dt = trajectory.times[next_step] - trajectory.times[next_step - 1]
    target_velocity, _ = integration_consistent_targets(
        input_state,
        exact_position,
        step_dt,
    )
    moving = ~(static.fixed_mask | static.prescribed_mask)
    if not bool(moving.any().item()):
        moving = ~static.fixed_mask
    predicted_position = output.next_position[moving]
    predicted_velocity = output.next_velocity[moving]
    target_position = exact_position[moving]
    target_velocity = target_velocity[moving]
    position_loss = F.huber_loss(
        (predicted_position - target_position) / normalizers.displacement_scale,
        torch.zeros_like(predicted_position),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    velocity_loss = F.huber_loss(
        (predicted_velocity - target_velocity) / normalizers.velocity_scale,
        torch.zeros_like(predicted_velocity),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    stress_mask = ~static.prescribed_mask
    if not bool(stress_mask.any().item()):
        stress_mask = torch.ones_like(static.prescribed_mask)
    stress_loss, stress_parts = _supervised_constitutive_loss(
        output.nodal_stress,
        output.cell_stress_tensor,
        trajectory,
        next_step,
        normalizers,
        cfg,
        nodal_mask=stress_mask,
    )

    diagnostics = output.energy_diagnostics
    work_scale = (
        diagnostics["kinetic"].abs()
        + diagnostics["kinetic_after"].abs()
        + diagnostics["potential"].abs()
        + diagnostics["potential_after"].abs()
        + diagnostics["external_work"].abs()
        + diagnostics["boundary_work"].abs()
        + diagnostics["residual_work"].abs()
        + diagnostics["projection_dissipation"].abs()
    ).detach().clamp_min(1.0e-8)
    work_loss = F.huber_loss(
        diagnostics["work_energy_balance"] / work_scale,
        torch.zeros_like(diagnostics["work_energy_balance"]),
        delta=1.0,
    )
    projection_free = (
        diagnostics["integration_update_scale"] >= 1.0 - 1.0e-7
    ).to(work_loss.dtype)
    work_loss = work_loss * projection_free * diagnostics["integration_valid"]
    penetration_loss = diagnostics["max_penetration"].square()
    residual_loss = (
        diagnostics["residual_norm"]
        / diagnostics["residual_reference"].clamp_min(1.0e-12)
    ).square()
    negative_j_loss = diagnostics["negative_j"]

    total = (
        float(get_cfg(cfg, "loss.position", 1.0)) * position_loss
        + float(get_cfg(cfg, "loss.velocity", 1.0)) * velocity_loss
        + float(get_cfg(cfg, "loss.stress", 1.0)) * stress_loss
        + float(get_cfg(cfg, "loss.work_energy", 0.1)) * work_loss
        + float(get_cfg(cfg, "loss.penetration", 1.0)) * penetration_loss
        + float(get_cfg(cfg, "loss.residual", 0.01)) * residual_loss
        + float(get_cfg(cfg, "loss.negative_j", 10.0)) * negative_j_loss
    )
    return total, {
        "loss": total.detach(),
        "position": position_loss.detach(),
        "velocity": velocity_loss.detach(),
        "stress": stress_loss.detach(),
        "stress_base": stress_parts["stress_base"],
        "stress_peak": stress_parts["stress_peak"],
        "stress_physical": stress_parts["stress_physical"],
        "stress_tensor_supervision": stress_parts["stress_tensor_supervision"],
        "stress_tensor_relative_rmse": stress_parts.get(
            "stress_tensor_relative_rmse", stress_loss.new_zeros(())
        ),
        "stress_tensor_vm_relative_rmse": stress_parts.get(
            "stress_tensor_vm_relative_rmse", stress_loss.new_zeros(())
        ),
        "work_energy": work_loss.detach(),
        "penetration": penetration_loss.detach(),
        "residual": residual_loss.detach(),
        "negative_j": negative_j_loss.detach(),
        "integration_update_scale": diagnostics[
            "integration_update_scale"
        ].detach(),
        "integration_backtrack": (
            diagnostics["integration_update_scale"] < 1.0 - 1.0e-7
        ).to(total.dtype).detach(),
        "integration_failure": (
            diagnostics["integration_valid"] < 0.5
        ).to(total.dtype).detach(),
        "integration_domain": diagnostics[
            "integration_domain_penalty"
        ].detach(),
    }


def integration_consistent_targets(
    input_state: CHPState,
    exact_next_position: torch.Tensor,
    dt: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply state-noise correction under the public full-step convention."""

    step_dt = torch.as_tensor(
        dt,
        device=input_state.position.device,
        dtype=input_state.position.dtype,
    )
    target_velocity = (exact_next_position - input_state.position) / step_dt
    target_acceleration = (target_velocity - input_state.velocity) / step_dt
    return target_velocity, target_acceleration


def minimax_checkpoint_score(
    metrics: dict[str, float],
    native_reference: dict[str, float] | None,
) -> float:
    """Worst relative degradation; no metric can be traded for another."""

    references = native_reference or {}
    ratios = []
    for key in ROLLOUT_METRIC_KEYS:
        value = float(metrics[key])
        reference = float(references.get(key, 1.0))
        if not math.isfinite(value) or reference <= 0.0:
            return float("inf")
        ratios.append(value / reference)
    return max(ratios)


def load_native_reference(cfg: dict[str, Any]) -> dict[str, float] | None:
    inline = get_cfg(cfg, "validation.native_reference", None)
    path = get_cfg(cfg, "validation.native_reference_file", None)
    reference_mode = str(
        get_cfg(cfg, "validation.checkpoint_reference_mode", "auto")
    ).lower()
    if reference_mode == "absolute_validation":
        if inline or path:
            raise ValueError(
                "absolute_validation checkpoint selection forbids native references"
            )
        return None
    if reference_mode not in {"auto", "native_validation"}:
        raise ValueError(
            "validation.checkpoint_reference_mode must be auto, "
            "absolute_validation, or native_validation"
        )
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        inline = payload
    if not inline:
        if reference_mode == "native_validation":
            raise ValueError(
                "native_validation checkpoint selection requires a validation reference artifact"
            )
        return None
    source_payload = inline
    if bool(get_cfg(cfg, "validation.require_native_reference_provenance", False)):
        steps = get_cfg(cfg, "validation.steps", None)
        validate_reference_protocol(
            source_payload,
            split_file=get_cfg(cfg, "data.split_file"),
            split=str(
                get_cfg(
                    cfg,
                    "validation.native_reference_split",
                    get_cfg(cfg, "data.val_split", "val"),
                )
            ),
            case_count=int(get_cfg(cfg, "validation.cases", 20)),
            frame_count=None if steps is None else int(steps) + 1,
            case_selection=str(
                get_cfg(cfg, "validation.native_reference_case_selection", "even")
            ),
        )
    inline = source_payload
    for container in ("rollout", "summary", "aggregate"):
        if container in inline and isinstance(inline[container], dict):
            inline = inline[container]
            break
    missing = [key for key in ROLLOUT_METRIC_KEYS if key not in inline]
    if missing:
        raise ValueError(f"native checkpoint reference is missing: {missing}")
    return {key: float(inline[key]) for key in ROLLOUT_METRIC_KEYS}


def _constitutive_parameters(model: CHPGNS) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = [
        model.log_stress_scale,
        *model.potential.parameters(),
    ]
    if model.material_potential is not None:
        parameters.extend(model.material_potential.parameters())
    return parameters


@torch.no_grad()
def calibrate_constitutive_modulus(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit the train-only physical modulus by a closed-form least-squares step."""

    model.eval()
    case_count = min(
        int(get_cfg(cfg, "constitutive_pretraining.calibration_cases", 32)),
        len(cases),
    )
    frames_per_case = max(
        int(get_cfg(cfg, "constitutive_pretraining.calibration_frames", 8)), 1
    )
    case_indices = np.linspace(0, len(cases) - 1, case_count).round().astype(int)
    prediction_square = torch.zeros((), device=cache.device)
    prediction_target = torch.zeros((), device=cache.device)
    target_square = torch.zeros((), device=cache.device)
    frame_moments: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    admissible_frames = 0
    requested_frames = 0
    tensor_frames = 0
    scalar_frames = 0
    for case_index_value in tqdm(
        case_indices, desc="constitutive-scale", leave=False
    ):
        case_index = int(case_index_value)
        case = cases[case_index]
        trajectory = cache.get(case_index)
        frames = np.linspace(
            1, case.num_steps - 1, min(frames_per_case, case.num_steps - 1)
        )
        for frame_value in np.unique(frames.round().astype(int)):
            requested_frames += 1
            frame = int(frame_value)
            exact_position = (
                trajectory.static.reference_position
                + trajectory.displacement[frame]
            )
            if not is_admissible_training_state(
                trajectory.static, exact_position, cfg
            ):
                continue
            nodal_prediction, cell_prediction = model.nodal_stress_at(
                trajectory.static, exact_position
            )
            mask = ~trajectory.static.prescribed_mask
            if not bool(mask.any().item()):
                mask = torch.ones_like(mask)
            if _trajectory_has_cell_stress_tensor(trajectory):
                prediction = _cauchy_to_tensor6(cell_prediction).float()
                target = trajectory.cell_stress[frame].float()
                tensor_frames += 1
            else:
                prediction = nodal_prediction[mask, :1].float()
                target = trajectory.stress[frame, mask, :1].float()
                scalar_frames += 1
            if not bool(
                (torch.isfinite(prediction).all() & torch.isfinite(target).all()).item()
            ):
                continue
            frame_prediction_square = prediction.square().sum()
            frame_prediction_target = (prediction * target).sum()
            frame_target_square = target.square().sum()
            prediction_square += frame_prediction_square
            prediction_target += frame_prediction_target
            target_square += frame_target_square
            if bool(
                (
                    (frame_prediction_square > 1.0e-12)
                    & (frame_prediction_target > 0.0)
                    & (frame_target_square > 1.0e-12)
                ).item()
            ):
                frame_moments.append(
                    (
                        frame_prediction_square,
                        frame_prediction_target,
                        frame_target_square,
                    )
                )
            admissible_frames += 1
    if not frame_moments:
        raise RuntimeError("no admissible non-zero frames for constitutive calibration")
    # A pooled least-squares factor is dominated by a handful of nearly
    # singular cells.  The median of per-frame optima is a robust train-only
    # modulus estimator and is stable across mesh resolutions.
    frame_factors = torch.stack(
        [cross / square for square, cross, _ in frame_moments]
    )
    factor = frame_factors.median().clamp(
        min=float(get_cfg(cfg, "constitutive_pretraining.minimum_scale_factor", 1.0e-3)),
        max=float(get_cfg(cfg, "constitutive_pretraining.maximum_scale_factor", 1.0e4)),
    )
    before_error = prediction_square - 2.0 * prediction_target + target_square
    after_error = (
        factor.square() * prediction_square
        - 2.0 * factor * prediction_target
        + target_square
    )
    log_factor = factor.log()
    model.log_stress_scale.add_(log_factor)
    if bool(
        get_cfg(cfg, "constitutive_pretraining.preserve_force_mass_ratio", True)
    ):
        model.log_mass_scale.add_(log_factor)
    frame_relative_after = torch.stack(
        [
            torch.sqrt(
                (
                    factor.square() * square
                    - 2.0 * factor * cross
                    + reference
                ).clamp_min(0.0)
                / reference.clamp_min(1.0e-12)
            )
            for square, cross, reference in frame_moments
        ]
    )
    return {
        "scale_factor": float(factor.item()),
        "physical_modulus": float(model.log_stress_scale.exp().item()),
        "mass_scale": float(model.log_mass_scale.exp().item()),
        "relative_rmse_before": float(
            torch.sqrt(before_error.clamp_min(0.0) / target_square.clamp_min(1.0e-12)).item()
        ),
        "relative_rmse_after": float(
            torch.sqrt(after_error.clamp_min(0.0) / target_square.clamp_min(1.0e-12)).item()
        ),
        "median_frame_relative_rmse_after": float(
            frame_relative_after.median().item()
        ),
        "frame_scale_q05": float(torch.quantile(frame_factors, 0.05).item()),
        "frame_scale_median": float(frame_factors.median().item()),
        "frame_scale_q95": float(torch.quantile(frame_factors, 0.95).item()),
        "admissible_frames": float(admissible_frames),
        "requested_frames": float(requested_frames),
        "admissible_coverage": float(admissible_frames / max(requested_frames, 1)),
        "stress_source": (
            "cell_tensor" if tensor_frames else "nodal_scalar_vm_fallback"
        ),
        "cell_tensor_frames": float(tensor_frames),
        "nodal_scalar_frames": float(scalar_frames),
        "cell_tensor_coverage": float(tensor_frames / max(admissible_frames, 1)),
    }


def _exact_constitutive_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    transformed_prediction = normalizers.stress.transform(prediction)
    transformed_target = normalizers.stress.transform(target)
    transformed, parts = robust_stress_loss(
        transformed_prediction,
        transformed_target,
        ranking_target=target,
        peak_fraction=float(get_cfg(cfg, "loss.peak_fraction", 0.1)),
        peak_weight=float(get_cfg(cfg, "loss.peak_weight", 0.5)),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    scale = normalizers.stress.reference_scale.to(
        prediction.device, prediction.dtype
    ).clamp_min(1.0e-8)
    physical = F.huber_loss(
        (prediction - target) / scale,
        torch.zeros_like(prediction),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    total = transformed + float(
        get_cfg(cfg, "loss.stress_physical_weight", 0.25)
    ) * physical
    return total, {
        "loss": total.detach(),
        "stress_transformed": transformed.detach(),
        "stress_base": parts["stress_base"],
        "stress_physical": physical.detach(),
        "stress_peak": parts["stress_peak"],
        "stress_tensor_supervision": total.new_zeros(()).detach(),
    }


def _cauchy_to_tensor6(stress: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., 3, 3]`` stress to canonical 11,22,33,12,13,23 order."""

    if stress.shape[-2:] != (3, 3):
        raise ValueError(f"cell stress must end in [3, 3], got {tuple(stress.shape)}")
    return torch.stack(
        (
            stress[..., 0, 0],
            stress[..., 1, 1],
            stress[..., 2, 2],
            0.5 * (stress[..., 0, 1] + stress[..., 1, 0]),
            0.5 * (stress[..., 0, 2] + stress[..., 2, 0]),
            0.5 * (stress[..., 1, 2] + stress[..., 2, 1]),
        ),
        dim=-1,
    )


def _tensor6_von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Von-Mises stress for canonical 11,22,33,12,13,23 tensors."""

    if stress.shape[-1] != 6:
        raise ValueError(f"canonical cell stress must end in 6, got {tuple(stress.shape)}")
    s11, s22, s33, s12, s13, s23 = stress.unbind(dim=-1)
    squared = (
        0.5
        * ((s11 - s22).square() + (s22 - s33).square() + (s33 - s11).square())
        + 3.0 * (s12.square() + s13.square() + s23.square())
    )
    squared = squared.clamp_min(0.0)
    eps = squared.new_tensor(torch.finfo(squared.dtype).eps)
    return torch.sqrt(squared + eps) - torch.sqrt(eps)


def _cell_tensor_constitutive_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Signed tensor loss with a derived, constitutively consistent VM auxiliary."""

    if normalizers.cell_stress is None:
        raise ValueError("cell tensor labels require a fitted cell-stress normalizer")
    prediction6 = _cauchy_to_tensor6(prediction)
    if target.shape != prediction6.shape:
        raise ValueError(
            "cell tensor target must match the predicted canonical tensor: "
            f"{tuple(target.shape)} != {tuple(prediction6.shape)}"
        )
    target = target.to(prediction6.dtype)
    target_vm = _tensor6_von_mises(target)
    prediction_vm = _tensor6_von_mises(prediction6)
    tensor_transform = normalizers.cell_stress
    transformed_prediction = tensor_transform.transform(prediction6)
    transformed_target = tensor_transform.transform(target)
    huber_delta = float(get_cfg(cfg, "loss.huber_delta", 1.0))
    tensor_base = F.huber_loss(
        transformed_prediction,
        transformed_target,
        delta=huber_delta,
    )
    peak_fraction = float(get_cfg(cfg, "loss.peak_fraction", 0.1))
    peak_weight = float(get_cfg(cfg, "loss.peak_weight", 0.5))
    if target.shape[0] and peak_fraction > 0.0 and peak_weight > 0.0:
        peak_count = max(1, min(target.shape[0], round(target.shape[0] * peak_fraction)))
        peak_cells = target_vm.detach().abs().topk(int(peak_count)).indices
        tensor_peak = F.huber_loss(
            transformed_prediction[peak_cells],
            transformed_target[peak_cells],
            delta=huber_delta,
        )
    else:
        tensor_peak = tensor_base.new_zeros(())
    tensor_transformed = tensor_base + peak_weight * tensor_peak
    tensor_scale = tensor_transform.reference_scale.to(
        prediction6.device, prediction6.dtype
    ).clamp_min(1.0e-8)
    tensor_physical = F.huber_loss(
        (prediction6 - target) / tensor_scale,
        torch.zeros_like(prediction6),
        delta=huber_delta,
    )
    tensor_loss = tensor_transformed + float(
        get_cfg(cfg, "loss.stress_physical_weight", 0.25)
    ) * tensor_physical

    vm_prediction = prediction_vm[:, None]
    vm_target = target_vm[:, None]
    vm_loss, _ = robust_stress_loss(
        normalizers.stress.transform(vm_prediction),
        normalizers.stress.transform(vm_target),
        ranking_target=vm_target,
        peak_fraction=float(get_cfg(cfg, "loss.peak_fraction", 0.1)),
        peak_weight=float(get_cfg(cfg, "loss.peak_weight", 0.5)),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    total = tensor_loss + float(get_cfg(cfg, "loss.cell_vm_weight", 0.25)) * vm_loss
    eps = prediction6.new_tensor(1.0e-12)
    tensor_relative = torch.sqrt(
        (prediction6 - target).square().sum() / target.square().sum().clamp_min(eps)
    )
    vm_relative = torch.sqrt(
        (prediction_vm - target_vm).square().sum()
        / target_vm.square().sum().clamp_min(eps)
    )
    return total, {
        "loss": total.detach(),
        "stress_transformed": tensor_transformed.detach(),
        "stress_base": tensor_base.detach(),
        "stress_physical": tensor_physical.detach(),
        "stress_peak": tensor_peak.detach(),
        "stress_tensor": tensor_loss.detach(),
        "stress_tensor_vm": vm_loss.detach(),
        "stress_tensor_relative_rmse": tensor_relative.detach(),
        "stress_tensor_vm_relative_rmse": vm_relative.detach(),
        "stress_tensor_supervision": total.new_ones(()).detach(),
    }


def _supervised_constitutive_loss(
    nodal_prediction: torch.Tensor,
    cell_prediction: torch.Tensor,
    trajectory: CHPDeviceCase,
    frame: int,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
    *,
    nodal_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer complete cell tensors, with an explicit nodal-VM fallback."""

    if _trajectory_has_cell_stress_tensor(trajectory):
        return _cell_tensor_constitutive_loss(
            cell_prediction,
            trajectory.cell_stress[int(frame)],
            normalizers,
            cfg,
        )
    return _exact_constitutive_loss(
        nodal_prediction[nodal_mask, :1],
        trajectory.stress[int(frame), nodal_mask, :1],
        normalizers,
        cfg,
    )


def train_constitutive_epoch(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    normalizers: CHPNormalizers,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    *,
    epoch: int,
) -> dict[str, float]:
    """Train only the shared potential on exact, admissible geometries."""

    model.train()
    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 7919 * int(epoch))
    case_count = min(
        int(get_cfg(cfg, "constitutive_pretraining.cases_per_epoch", 128)),
        len(cases),
    )
    frames_per_case = max(
        int(get_cfg(cfg, "constitutive_pretraining.frames_per_case", 8)), 1
    )
    case_indices = rng.choice(len(cases), size=case_count, replace=False)
    parameters = _constitutive_parameters(model)
    totals: dict[str, float] = {}
    used_frames = 0
    skipped_frames = 0
    for case_index_value in tqdm(
        case_indices, desc="constitutive-pretrain", leave=False
    ):
        case_index = int(case_index_value)
        case = cases[case_index]
        trajectory = cache.get(case_index)
        frame_candidates = rng.permutation(np.arange(1, case.num_steps))
        losses: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)
        for frame_value in frame_candidates:
            frame = int(frame_value)
            exact_position = (
                trajectory.static.reference_position
                + trajectory.displacement[frame]
            )
            if not is_admissible_training_state(
                trajectory.static, exact_position, cfg
            ):
                skipped_frames += 1
                continue
            nodal_prediction, cell_prediction = model.nodal_stress_at(
                trajectory.static, exact_position
            )
            mask = ~trajectory.static.prescribed_mask
            if not bool(mask.any().item()):
                mask = torch.ones_like(mask)
            loss, metrics = _supervised_constitutive_loss(
                nodal_prediction,
                cell_prediction,
                trajectory,
                frame,
                normalizers,
                cfg,
                nodal_mask=mask,
            )
            if not bool(torch.isfinite(loss).item()):
                skipped_frames += 1
                continue
            losses.append(loss)
            _accumulate(totals, metrics)
            used_frames += 1
            if len(losses) >= frames_per_case:
                break
        if not losses:
            continue
        torch.stack(losses).mean().backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(get_cfg(cfg, "constitutive_pretraining.grad_clip_norm", 10.0)),
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("non-finite constitutive pretraining gradient")
        optimizer.step()
    averaged = {key: value / max(used_frames, 1) for key, value in totals.items()}
    averaged.update(
        {
            "used_frames": float(used_frames),
            "skipped_frames": float(skipped_frames),
            "physical_modulus": float(model.log_stress_scale.detach().exp().item()),
        }
    )
    return averaged


def run_constitutive_pretraining(
    model: CHPGNS,
    train_cases: list[ValveCase],
    val_cases: list[ValveCase] | None,
    train_cache: CHPCaseCache,
    val_cache: CHPCaseCache | None,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    """Calibrate and pretrain the potential before coupled rollouts."""

    if not bool(get_cfg(cfg, "constitutive_pretraining.enabled", True)):
        return {"enabled": False}
    history: dict[str, Any] = {
        "enabled": True,
        "calibration": calibrate_constitutive_modulus(
            model, train_cases, train_cache, cfg
        ),
        "epochs": [],
    }
    initial_validation = (
        evaluate_teacher_forced_stress(
            model, val_cases, val_cache, cfg, amp_dtype=amp_dtype
        )
        if val_cases is not None and val_cache is not None
        else {}
    )
    history["initial_validation"] = initial_validation
    best_metric = float(
        initial_validation.get("teacher_stress_relative_rmse", float("inf"))
    )
    best_validation = dict(initial_validation)
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    optimizer = torch.optim.Adam(
        _constitutive_parameters(model),
        lr=float(get_cfg(cfg, "constitutive_pretraining.lr", 1.0e-3)),
        weight_decay=0.0,
    )
    epochs = max(int(get_cfg(cfg, "constitutive_pretraining.epochs", 2)), 0)
    for epoch in range(1, epochs + 1):
        train_metrics = train_constitutive_epoch(
            model,
            train_cases,
            train_cache,
            normalizers,
            optimizer,
            cfg,
            epoch=epoch,
        )
        validation = (
            evaluate_teacher_forced_stress(
                model, val_cases, val_cache, cfg, amp_dtype=amp_dtype
            )
            if val_cases is not None and val_cache is not None
            else {}
        )
        metric = float(validation.get("teacher_stress_relative_rmse", float("inf")))
        if metric < best_metric:
            best_metric = metric
            best_validation = dict(validation)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        history["epochs"].append(
            {"epoch": epoch, "train": train_metrics, "validation": validation}
        )
    model.load_state_dict(best_state)
    history["selected_teacher_stress_relative_rmse"] = best_metric
    history["selected_teacher_stress_source"] = best_validation.get(
        "teacher_stress_source", "unavailable"
    )
    history["selected_teacher_stress_label_coverage"] = best_validation.get(
        "teacher_stress_label_coverage", 0.0
    )
    history["selected_validation"] = best_validation
    _save_json(output_dir / "constitutive_pretraining.json", history)
    return history


def exact_dynamics_loss(
    output: PhysicalStep,
    input_state: CHPState,
    trajectory: CHPDeviceCase,
    next_step: int,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
    *,
    contact_pairs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the force-driven update on an exact, noise-free state."""

    static = trajectory.static
    exact_position = static.reference_position + trajectory.displacement[next_step]
    dt = trajectory.times[next_step] - trajectory.times[next_step - 1]
    target_velocity, target_acceleration = integration_consistent_targets(
        input_state, exact_position, dt
    )
    moving = ~(static.fixed_mask | static.prescribed_mask)
    if not bool(moving.any().item()):
        raise ValueError("exact dynamics pretraining requires at least one moving node")
    target_acceleration_moving = target_acceleration[moving]
    global_scale = normalizers.acceleration_scale.clamp_min(1.0e-8)
    frame_scale = torch.maximum(
        global_scale,
        target_acceleration_moving.square().mean().sqrt().detach(),
    )
    position_loss = F.huber_loss(
        (output.next_position[moving] - exact_position[moving])
        / (dt.abs().square() * frame_scale).clamp_min(1.0e-8),
        torch.zeros_like(exact_position[moving]),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    velocity_loss = F.huber_loss(
        (output.next_velocity[moving] - target_velocity[moving])
        / (dt.abs() * frame_scale).clamp_min(1.0e-8),
        torch.zeros_like(target_velocity[moving]),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    acceleration_element = F.huber_loss(
        (output.acceleration[moving] - target_acceleration_moving) / frame_scale,
        torch.zeros_like(target_acceleration_moving),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
        reduction="none",
    )
    acceleration_base = acceleration_element.mean()
    target_norm = torch.linalg.vector_norm(target_acceleration_moving, dim=1)
    peak_fraction = float(
        get_cfg(cfg, "dynamics_pretraining.active_node_fraction", 0.2)
    )
    peak_count = max(int(math.ceil(target_norm.numel() * peak_fraction)), 1)
    peak_nodes = torch.topk(target_norm, k=peak_count, sorted=False).indices
    acceleration_peak = acceleration_element[peak_nodes].mean()
    acceleration_loss = 0.5 * acceleration_base + 0.5 * acceleration_peak
    stress_mask = ~static.prescribed_mask
    if not bool(stress_mask.any().item()):
        stress_mask = torch.ones_like(static.prescribed_mask)
    stress_loss, stress_metrics = _supervised_constitutive_loss(
        output.nodal_stress,
        output.cell_stress_tensor,
        trajectory,
        next_step,
        normalizers,
        cfg,
        nodal_mask=stress_mask,
    )
    projection_free = (
        (output.energy_diagnostics["integration_update_scale"] >= 1.0 - 1.0e-7)
        & (output.energy_diagnostics["integration_valid"] >= 0.5)
    ).to(acceleration_loss.dtype)
    effective_mass = (
        static.lumped_mass.reshape(-1, 1)
        * output.energy_diagnostics["mass_scale"]
    )
    residual_acceleration = output.residual_force / effective_mass
    residual_moving = residual_acceleration[moving]
    correction_target = (
        target_acceleration_moving
        - (output.acceleration[moving] - residual_moving).detach()
    ).detach()
    correction_norm = torch.linalg.vector_norm(correction_target, dim=1)
    direction_mask = correction_norm > (
        float(get_cfg(cfg, "dynamics_pretraining.active_threshold", 0.05))
        * global_scale
    )
    if contact_pairs is not None and contact_pairs.numel():
        contact_nodes = torch.zeros_like(moving)
        contact_nodes[contact_pairs.reshape(-1).long()] = True
        direction_mask = direction_mask & ~contact_nodes[moving]
    if bool(direction_mask.any().item()):
        selected_correction = correction_target[direction_mask].float()
        correction_unit = selected_correction / torch.linalg.vector_norm(
            selected_correction, dim=1, keepdim=True
        ).clamp_min(global_scale.detach().float() * 0.05)
        directional_projection = (
            residual_moving[direction_mask].float()
            / frame_scale.detach().float()
            * correction_unit
        ).sum(dim=1)
        direction_loss = (1.0 - directional_projection).mean()
    else:
        direction_loss = acceleration_loss.new_zeros(())
    quiet_mask = target_norm <= (
        float(get_cfg(cfg, "dynamics_pretraining.active_threshold", 0.05))
        * global_scale
    )
    quiet_loss = (
        (residual_moving[quiet_mask] / global_scale).square().mean()
        if bool(quiet_mask.any().item())
        else acceleration_loss.new_zeros(())
    )
    state_loss = projection_free * (
        float(get_cfg(cfg, "dynamics_pretraining.position_weight", 0.25))
        * position_loss
        + float(get_cfg(cfg, "dynamics_pretraining.velocity_weight", 0.25))
        * velocity_loss
        + float(get_cfg(cfg, "dynamics_pretraining.acceleration_weight", 1.0))
        * acceleration_loss
        + float(get_cfg(cfg, "dynamics_pretraining.direction_weight", 0.25))
        * direction_loss
        + float(get_cfg(cfg, "dynamics_pretraining.quiet_weight", 0.05))
        * quiet_loss
        + float(get_cfg(cfg, "dynamics_pretraining.stress_weight", 0.1))
        * stress_loss
    )
    residual_loss = (
        output.energy_diagnostics["residual_norm"]
        / output.energy_diagnostics["residual_reference"].clamp_min(1.0e-12)
    ).square()
    total = (
        state_loss
        + float(get_cfg(cfg, "dynamics_pretraining.residual_weight", 1.0e-3))
        * residual_loss
        + float(get_cfg(cfg, "loss.negative_j", 10.0))
        * output.energy_diagnostics["negative_j"]
    )
    return total, {
        "loss": total.detach(),
        "position": position_loss.detach(),
        "velocity": velocity_loss.detach(),
        "acceleration": acceleration_loss.detach(),
        "acceleration_base": acceleration_base.detach(),
        "acceleration_peak": acceleration_peak.detach(),
        "direction": direction_loss.detach(),
        "quiet": quiet_loss.detach(),
        "stress": stress_loss.detach(),
        "stress_tensor_supervision": stress_metrics[
            "stress_tensor_supervision"
        ],
        "residual": residual_loss.detach(),
        "integration_backtrack": (1.0 - projection_free).detach(),
    }


@torch.no_grad()
def evaluate_teacher_forced_dynamics(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    cfg: dict[str, Any],
    normalizers: CHPNormalizers,
    *,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """Measure exact-state one-step dynamics without touching the test split."""

    model.eval()
    case_count = min(
        int(get_cfg(cfg, "dynamics_pretraining.validation_cases", 10)),
        len(cases),
    )
    frames_per_case = max(
        int(get_cfg(cfg, "dynamics_pretraining.validation_frames", 8)), 1
    )
    case_indices = np.linspace(0, len(cases) - 1, case_count).round().astype(int)
    device = cache.device
    totals = {
        key: torch.zeros((), device=device, dtype=torch.float64)
        for key in (
            "a_error",
            "a_reference",
            "a_prediction",
            "a_cross",
            "active_error",
            "active_reference",
            "active_prediction",
            "active_cross",
            "delta_error",
            "delta_reference",
            "stress_error",
            "stress_reference",
            "residual_square",
            "residual_count",
            "saturated",
            "node_count",
            "backtrack",
            "failure",
        )
    }
    used_frames = 0
    pressure_sign = float(get_cfg(cfg, "data.pressure_sign", 1.0))
    for case_index_value in case_indices:
        case_index = int(case_index_value)
        case = cases[case_index]
        trajectory = cache.get(case_index)
        frames = np.linspace(
            0,
            case.num_steps - 2,
            min(frames_per_case, case.num_steps - 1),
        )
        for frame_value in np.unique(frames.round().astype(int)):
            frame = int(frame_value)
            state = CHPState(
                trajectory.static.reference_position
                + trajectory.displacement[frame],
                trajectory.velocity[frame],
            )
            pairs = contact_pairs_at(trajectory, state, cfg)
            dt = trajectory.times[frame + 1] - trajectory.times[frame]
            with _autocast(device, amp_dtype):
                output = model(
                    trajectory.static,
                    state,
                    contact_pairs=pairs,
                    external_force=external_force_at(
                        trajectory, frame, pressure_sign
                    ),
                    dt=dt,
                    time_fraction=trajectory.time_fraction[frame],
                    prescribed_position=(
                        trajectory.static.reference_position
                        + trajectory.displacement[frame + 1]
                    ),
                    prescribed_velocity=trajectory.velocity[frame + 1],
                )
            moving = ~(
                trajectory.static.fixed_mask
                | trajectory.static.prescribed_mask
            )
            if not bool(moving.any().item()):
                continue
            exact_next = (
                trajectory.static.reference_position
                + trajectory.displacement[frame + 1]
            )
            target_velocity, target_acceleration = integration_consistent_targets(
                state, exact_next, dt
            )
            prediction = output.acceleration[moving].double()
            target = target_acceleration[moving].double()
            error = prediction - target
            totals["a_error"] += error.square().sum()
            totals["a_reference"] += target.square().sum()
            totals["a_prediction"] += prediction.square().sum()
            totals["a_cross"] += (prediction * target).sum()
            active = torch.linalg.vector_norm(target, dim=1) > (
                float(get_cfg(cfg, "dynamics_pretraining.active_threshold", 0.05))
                * normalizers.acceleration_scale.double()
            )
            if bool(active.any().item()):
                totals["active_error"] += error[active].square().sum()
                totals["active_reference"] += target[active].square().sum()
                totals["active_prediction"] += prediction[active].square().sum()
                totals["active_cross"] += (prediction[active] * target[active]).sum()
            predicted_delta = output.next_position[moving] - state.position[moving]
            target_delta = exact_next[moving] - state.position[moving]
            totals["delta_error"] += (
                predicted_delta.double() - target_delta.double()
            ).square().sum()
            totals["delta_reference"] += target_delta.double().square().sum()
            stress_mask = ~trajectory.static.prescribed_mask
            stress_target = trajectory.stress[frame + 1, stress_mask, :1].double()
            stress_error = (
                output.nodal_stress[stress_mask, :1].double() - stress_target
            )
            totals["stress_error"] += stress_error.square().sum()
            totals["stress_reference"] += stress_target.square().sum()
            residual = (
                output.residual_force[moving]
                / (
                    trajectory.static.lumped_mass.reshape(-1, 1)[moving]
                    * output.energy_diagnostics["mass_scale"]
                )
            )
            residual_norm = torch.linalg.vector_norm(residual.double(), dim=1)
            totals["residual_square"] += residual_norm.square().sum()
            totals["residual_count"] += residual_norm.numel()
            totals["saturated"] += (
                residual_norm >= 0.95 * model.residual_acceleration_cap
            ).double().sum()
            totals["node_count"] += residual_norm.numel()
            totals["backtrack"] += (
                output.energy_diagnostics["integration_update_scale"]
                < 1.0 - 1.0e-7
            ).double()
            totals["failure"] += (
                output.energy_diagnostics["integration_valid"] < 0.5
            ).double()
            used_frames += 1
    eps = torch.tensor(1.0e-12, device=device, dtype=torch.float64)
    acceleration_relative = torch.sqrt(
        totals["a_error"] / totals["a_reference"].clamp_min(eps)
    )
    acceleration_cosine = totals["a_cross"] / torch.sqrt(
        totals["a_prediction"].clamp_min(eps)
        * totals["a_reference"].clamp_min(eps)
    )
    active_relative = torch.sqrt(
        totals["active_error"] / totals["active_reference"].clamp_min(eps)
    )
    active_cosine = totals["active_cross"] / torch.sqrt(
        totals["active_prediction"].clamp_min(eps)
        * totals["active_reference"].clamp_min(eps)
    )
    return {
        "acceleration_relative_rmse": float(acceleration_relative.item()),
        "acceleration_cosine": float(acceleration_cosine.item()),
        "active_acceleration_relative_rmse": float(active_relative.item()),
        "active_acceleration_cosine": float(active_cosine.item()),
        "position_increment_relative_rmse": float(
            torch.sqrt(
                totals["delta_error"]
                / totals["delta_reference"].clamp_min(eps)
            ).item()
        ),
        "one_step_stress_relative_rmse": float(
            torch.sqrt(
                totals["stress_error"]
                / totals["stress_reference"].clamp_min(eps)
            ).item()
        ),
        "residual_acceleration_rms": float(
            torch.sqrt(
                totals["residual_square"]
                / totals["residual_count"].clamp_min(1.0)
            ).item()
        ),
        "residual_saturation_fraction": float(
            (totals["saturated"] / totals["node_count"].clamp_min(1.0)).item()
        ),
        "integration_backtrack_fraction": float(
            (totals["backtrack"] / max(used_frames, 1)).item()
        ),
        "integration_failure_fraction": float(
            (totals["failure"] / max(used_frames, 1)).item()
        ),
        "frames": float(used_frames),
    }


def _dynamics_pretraining_parameters(
    model: CHPGNS,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    residual = list(model.residual_channel.parameters())
    residual_ids = {id(parameter) for parameter in residual}
    processor: list[torch.nn.Parameter] = []
    for module in (
        model.node_encoder,
        model.vector_encoder,
        model.processor,
    ):
        processor.extend(
            parameter
            for parameter in module.parameters()
            if id(parameter) not in residual_ids
        )
    return processor, residual


def train_exact_dynamics_epoch(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    sampling_scores: list[np.ndarray],
    normalizers: CHPNormalizers,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    *,
    epoch: int,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """Fit exact one-step acceleration before exposing the model to noise."""

    model.train()
    model.processor.activation_checkpointing = False
    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 7919 * int(epoch))
    indices = rng.permutation(len(cases))
    requested = int(
        get_cfg(cfg, "dynamics_pretraining.trajectories_per_epoch", len(cases))
    )
    indices = indices[: min(requested, len(indices))]
    pressure_sign = float(get_cfg(cfg, "data.pressure_sign", 1.0))
    totals: dict[str, float] = {}
    used = 0
    nonfinite_losses = 0
    nonfinite_gradients = 0
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    for case_index_value in tqdm(indices, desc="dynamics-pretrain", leave=False):
        case_index = int(case_index_value)
        case = cases[case_index]
        start, _ = select_rollout_start(
            case.num_steps, 1, sampling_scores[case_index], rng
        )
        trajectory = cache.get_slice(case_index, start, start + 2)
        if not is_admissible_training_window(trajectory, cfg):
            continue
        state = CHPState(
            trajectory.static.reference_position + trajectory.displacement[0],
            trajectory.velocity[0],
        )
        pairs = contact_pairs_at(trajectory, state, cfg)
        dt = trajectory.times[1] - trajectory.times[0]
        optimizer.zero_grad(set_to_none=True)
        with _autocast(cache.device, amp_dtype):
            output = model(
                trajectory.static,
                state,
                contact_pairs=pairs,
                external_force=external_force_at(trajectory, 0, pressure_sign),
                dt=dt,
                time_fraction=trajectory.time_fraction[0],
                prescribed_position=(
                    trajectory.static.reference_position
                    + trajectory.displacement[1]
                ),
                prescribed_velocity=trajectory.velocity[1],
                detach_pair_force_features=True,
            )
            loss, metrics = exact_dynamics_loss(
                output,
                state,
                trajectory,
                1,
                normalizers,
                cfg,
                contact_pairs=pairs,
            )
        if not bool(torch.isfinite(loss).item()):
            nonfinite_losses += 1
            continue
        loss.backward()
        finite_gradient = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in parameters
        )
        if not finite_gradient:
            nonfinite_gradients += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(get_cfg(cfg, "dynamics_pretraining.grad_clip_norm", 1.0)),
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            nonfinite_gradients += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        _accumulate(totals, metrics)
        used += 1
    averaged = {key: value / max(used, 1) for key, value in totals.items()}
    averaged["steps"] = float(used)
    averaged["nonfinite_losses"] = float(nonfinite_losses)
    averaged["nonfinite_gradients"] = float(nonfinite_gradients)
    return averaged


def run_dynamics_pretraining(
    model: CHPGNS,
    train_cases: list[ValveCase],
    val_cases: list[ValveCase] | None,
    train_cache: CHPCaseCache,
    val_cache: CHPCaseCache | None,
    sampling_scores: list[np.ndarray],
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    """Pretrain bounded residual dynamics on exact states and select on val."""

    if not bool(get_cfg(cfg, "dynamics_pretraining.enabled", True)):
        return {"enabled": False}
    if val_cases is None or val_cache is None:
        raise ValueError("dynamics pretraining requires a validation split")
    processor_parameters, residual_parameters = _dynamics_pretraining_parameters(model)
    selected_ids = {
        id(parameter)
        for parameter in (*processor_parameters, *residual_parameters)
    }
    original_requires_grad = {
        id(parameter): parameter.requires_grad for parameter in model.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    base_lr = float(get_cfg(cfg, "dynamics_pretraining.lr", 1.0e-4))
    head_warmup_lr = float(
        get_cfg(cfg, "dynamics_pretraining.head_warmup_lr", 5.0e-3)
    )
    joint_head_lr = float(
        get_cfg(
            cfg,
            "dynamics_pretraining.joint_head_lr",
            base_lr
            * float(
                get_cfg(cfg, "dynamics_pretraining.residual_lr_scale", 5.0)
            ),
        )
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": processor_parameters, "lr": 0.0},
            {
                "params": residual_parameters,
                "lr": head_warmup_lr,
            },
        ],
        weight_decay=float(
            get_cfg(cfg, "dynamics_pretraining.weight_decay", 1.0e-6)
        ),
    )
    history: dict[str, Any] = {"enabled": True, "epochs": []}
    try:
        initial = evaluate_teacher_forced_dynamics(
            model,
            val_cases,
            val_cache,
            cfg,
            normalizers,
            amp_dtype=amp_dtype,
        )
        history["initial_validation"] = initial
        best_score = max(
            initial["active_acceleration_relative_rmse"],
            initial["one_step_stress_relative_rmse"],
        )
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        epochs = max(int(get_cfg(cfg, "dynamics_pretraining.epochs", 2)), 0)
        head_warmup_epochs = max(
            int(get_cfg(cfg, "dynamics_pretraining.head_warmup_epochs", 1)), 0
        )
        for epoch in range(1, epochs + 1):
            if epoch <= head_warmup_epochs:
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = head_warmup_lr
            else:
                optimizer.param_groups[0]["lr"] = base_lr
                optimizer.param_groups[1]["lr"] = joint_head_lr
            train_metrics = train_exact_dynamics_epoch(
                model,
                train_cases,
                train_cache,
                sampling_scores,
                normalizers,
                optimizer,
                cfg,
                epoch=epoch,
                amp_dtype=amp_dtype,
            )
            validation = evaluate_teacher_forced_dynamics(
                model,
                val_cases,
                val_cache,
                cfg,
                normalizers,
                amp_dtype=amp_dtype,
            )
            score = max(
                validation["active_acceleration_relative_rmse"],
                validation["one_step_stress_relative_rmse"],
            )
            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            history["epochs"].append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation,
                    "score": score,
                    "processor_lr": optimizer.param_groups[0]["lr"],
                    "residual_head_lr": optimizer.param_groups[1]["lr"],
                }
            )
        model.load_state_dict(best_state)
        history["selected_score"] = best_score
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
    _save_json(output_dir / "dynamics_pretraining.json", history)
    return history


def run_chp_training(cfg: dict[str, Any]) -> Path:
    """Train CHP-GNS using BF16 neural blocks and FP32 mechanics on CUDA."""

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    device = _require_cuda(cfg)
    output_dir = Path(get_cfg(cfg, "training.output_dir", "outputs/chp_gns"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _assert_no_gate_failure_artifact(output_dir, context="training/resume")
    train_dataset, val_dataset = _build_chp_datasets(cfg)
    material_dim = max(
        case.material_features.shape[1]
        for case in train_dataset.cases + (val_dataset.cases if val_dataset else [])
    )
    requires_cell_stress_normalizer = any(
        _case_has_cell_stress_tensor(case) for case in train_dataset.cases
    )

    normalizer_path = output_dir / "normalizers.pt"
    if normalizer_path.exists():
        normalizer_state = _torch_load(normalizer_path, "cpu")
        normalizer_is_current = "acceleration_scale" in normalizer_state and (
            not requires_cell_stress_normalizer
            or normalizer_state.get("cell_stress") is not None
        )
        if normalizer_is_current:
            normalizers = CHPNormalizers.from_state_dict(normalizer_state)
        else:
            normalizers = fit_chp_normalizers(
                train_dataset.cases,
                max_cases=int(get_cfg(cfg, "training.normalizer_cases", 128)),
                frames_per_case=int(
                    get_cfg(cfg, "training.normalizer_frames", 8)
                ),
                nodes_per_frame=int(
                    get_cfg(cfg, "training.normalizer_nodes", 256)
                ),
            )
            torch.save(normalizers.state_dict(), normalizer_path)
    else:
        normalizers = fit_chp_normalizers(
            train_dataset.cases,
            max_cases=int(get_cfg(cfg, "training.normalizer_cases", 128)),
            frames_per_case=int(get_cfg(cfg, "training.normalizer_frames", 8)),
            nodes_per_frame=int(get_cfg(cfg, "training.normalizer_nodes", 256)),
        )
        torch.save(normalizers.state_dict(), normalizer_path)
    normalizers = normalizers.to(device)
    sampling_scores = build_sampling_scores(
        train_dataset.cases, output_dir / "stress_sampling_scores.pt"
    )

    model = CHPGNS(cfg, material_dim=material_dim).to(device)
    base_lr = float(get_cfg(cfg, "training.lr", 1.0e-4))
    weight_decay = float(get_cfg(cfg, "training.weight_decay", 1.0e-6))
    constitutive_parameters = _constitutive_parameters(model)
    constitutive_ids = {id(parameter) for parameter in constitutive_parameters}
    network_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in constitutive_ids
    ]
    optimizer_kwargs: dict[str, Any] = {
        "lr": base_lr,
        "weight_decay": weight_decay,
    }
    if bool(get_cfg(cfg, "training.fused_optimizer", True)):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [
            {
                "params": network_parameters,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "network",
            },
            {
                "params": constitutive_parameters,
                "lr": base_lr
                * float(get_cfg(cfg, "training.constitutive_lr_scale", 0.1)),
                "weight_decay": 0.0,
                "name": "constitutive",
            },
        ],
        **optimizer_kwargs,
    )
    stages = get_cfg(cfg, "training.curriculum", None)
    epochs = int(get_cfg(cfg, "training.epochs", 16))
    train_count = _limited_count(
        len(train_dataset.cases), get_cfg(cfg, "training.trajectories_per_epoch", None)
    )
    total_steps = max(epochs * train_count, 1)
    warmup_steps = min(int(get_cfg(cfg, "training.warmup_steps", 500)), total_steps // 4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _warmup_cosine(step, total_steps, warmup_steps),
    )
    amp_dtype = _amp_dtype(cfg)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_dtype == torch.float16
    )
    train_cache = CHPCaseCache(
        train_dataset.cases,
        device,
        material_dim=material_dim,
        cache_size=int(get_cfg(cfg, "training.gpu_case_cache_size", 3)),
    )
    val_cache = (
        CHPCaseCache(
            val_dataset.cases,
            device,
            material_dim=material_dim,
            cache_size=int(get_cfg(cfg, "validation.gpu_case_cache_size", 2)),
        )
        if val_dataset is not None
        else None
    )
    native_reference = load_native_reference(cfg)
    if native_reference is None:
        print("warning: no native reference supplied; minimax score uses unit references")
    print(
        f"CHP-GNS | device={device} | amp={str(amp_dtype).removeprefix('torch.')} "
        f"| train={len(train_dataset.cases)} | val={len(val_dataset.cases) if val_dataset else 0}"
    )

    best_score = float("inf")
    start_epoch = 1
    history = _load_json(output_dir / "history.json", default=[])
    resume = _resolve_resume(cfg, output_dir)
    if resume is None:
        pretraining = run_constitutive_pretraining(
            model,
            train_dataset.cases,
            val_dataset.cases if val_dataset is not None else None,
            train_cache,
            val_cache,
            normalizers,
            cfg,
            output_dir,
            amp_dtype=amp_dtype,
        )
        if pretraining.get("enabled"):
            selected = float(
                pretraining.get(
                    "selected_teacher_stress_relative_rmse", float("inf")
                )
            )
            selected_source = str(
                pretraining.get("selected_teacher_stress_source", "unavailable")
            )
            selected_coverage = float(
                pretraining.get("selected_teacher_stress_label_coverage", 0.0)
            )
            print(
                "constitutive pretraining selected teacher "
                f"rRMSE={selected:.4g} source={selected_source} "
                f"coverage={selected_coverage:.3f}"
            )
            teacher_threshold = float(
                get_cfg(cfg, "validation.teacher_stress_threshold", 0.50)
            )
            if (
                bool(
                    get_cfg(
                        cfg,
                        "validation.enforce_teacher_stress_gate",
                        True,
                    )
                )
                and selected >= teacher_threshold
            ):
                failure = {
                    "stage": "constitutive_pretraining",
                    "teacher_stress_relative_rmse": selected,
                    "teacher_stress_source": selected_source,
                    "teacher_stress_label_coverage": selected_coverage,
                    "threshold": teacher_threshold,
                    "action": "stop before dynamics and rollout training",
                }
                _save_json(
                    output_dir / "teacher_stress_gate_failure.json", failure
                )
                raise RuntimeError(
                    "pretrained teacher-forced stress gate failed: "
                    f"{selected:.4g} >= {teacher_threshold:.4g}"
                )
        dynamics_sampling_scores = build_acceleration_sampling_scores(
            train_dataset.cases,
            output_dir / "acceleration_sampling_scores.pt",
        )
        dynamics_pretraining = run_dynamics_pretraining(
            model,
            train_dataset.cases,
            val_dataset.cases if val_dataset is not None else None,
            train_cache,
            val_cache,
            dynamics_sampling_scores,
            normalizers,
            cfg,
            output_dir,
            amp_dtype=amp_dtype,
        )
        if dynamics_pretraining.get("enabled"):
            selected = float(
                dynamics_pretraining.get("selected_score", float("inf"))
            )
            print(f"dynamics pretraining selected minimax={selected:.4g}")
    if resume is not None:
        checkpoint = _torch_load(resume, device)
        validate_chp_checkpoint_semantics(checkpoint, source=resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", checkpoint.get("score", best_score)))
        print(f"resumed {resume} at epoch {start_epoch - 1}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        horizon = curriculum_horizon(epoch, stages)
        torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_chp_epoch(
            model,
            train_dataset.cases,
            train_cache,
            sampling_scores,
            normalizers,
            optimizer,
            scheduler,
            scaler,
            cfg,
            epoch=epoch,
            horizon=horizon,
            amp_dtype=amp_dtype,
        )
        stage_ends = _curriculum_stage_ends(stages)
        validation_every = int(get_cfg(cfg, "validation.every", 1))
        validate_now = (
            epoch == epochs
            or epoch in stage_ends
            or (
                not bool(get_cfg(cfg, "validation.stage_end_only", False))
                and validation_every > 0
                and epoch % validation_every == 0
            )
        )
        if validate_now and val_dataset is not None and val_cache is not None:
            rollout_metrics = evaluate_chp_rollouts(
                model, val_dataset.cases, val_cache, cfg, amp_dtype=amp_dtype
            )
            teacher_metrics = evaluate_teacher_forced_stress(
                model, val_dataset.cases, val_cache, cfg, amp_dtype=amp_dtype
            )
            rollout_metrics.update(teacher_metrics)
            score = minimax_checkpoint_score(rollout_metrics, native_reference)
        else:
            rollout_metrics = {}
            teacher_metrics = {}
            score = None
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        record = {
            "epoch": epoch,
            "horizon": horizon,
            "train": train_metrics,
            "rollout": rollout_metrics,
            "score": score,
            "peak_memory_gib": peak_gib,
            "seconds": time.perf_counter() - epoch_start,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history = [item for item in history if int(item["epoch"]) != epoch]
        history.append(record)
        _save_json(output_dir / "history.json", sorted(history, key=lambda x: x["epoch"]))
        score_label = f"{score:.4g}" if score is not None else "not-evaluated"
        print(
            f"epoch={epoch:02d} K={horizon:02d} loss={train_metrics['loss']:.5g} "
            f"score={score_label} peak={peak_gib:.2f}GiB "
            f"time={record['seconds']:.1f}s"
        )
        _enforce_scientific_gates(
            epoch,
            stages,
            teacher_metrics,
            rollout_metrics,
            cfg,
            output_dir,
        )
        gate_status = _scientific_gate_status(epoch, stages, cfg)
        checkpoint_score = float(score) if score is not None else 1.0e30
        checkpoint_best = min(best_score, checkpoint_score)
        if not math.isfinite(checkpoint_best):
            checkpoint_best = 1.0e30
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            normalizers,
            cfg,
            epoch,
            checkpoint_score,
            checkpoint_best,
            rollout_metrics,
            material_dim,
            scientific_gate_status=gate_status,
        )
        torch.save(payload, output_dir / "latest.pt")
        if (
            gate_status in {"passed", "not_required"}
            and score is not None
            and score < best_score
        ):
            best_score = score
            payload["best_score"] = best_score
            torch.save(payload, output_dir / "best.pt")
    if not (output_dir / "best.pt").exists():
        raise RuntimeError("CHP-GNS produced no scientifically eligible best checkpoint")
    return output_dir / "best.pt"


def train_chp_epoch(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    sampling_scores: list[np.ndarray],
    normalizers: CHPNormalizers,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    cfg: dict[str, Any],
    *,
    epoch: int,
    horizon: int,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.train()
    model.processor.activation_checkpointing = bool(
        get_cfg(cfg, "model.activation_checkpointing", True)
    ) and horizon >= int(get_cfg(cfg, "training.checkpoint_min_horizon", 8))
    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 1009 * int(epoch))
    indices = rng.permutation(len(cases))
    requested = get_cfg(cfg, "training.trajectories_per_epoch", None)
    if requested is not None:
        indices = indices[: int(requested)]
    totals: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    step_count = 0
    pressure_sign = float(get_cfg(cfg, "data.pressure_sign", 1.0))
    tbptt_chunk = int(get_cfg(cfg, "training.tbptt_chunk", 4)) if horizon >= 8 else horizon
    stop_gradient = bool(get_cfg(cfg, "training.pushforward_stop_gradient", True))
    noise_std = float(get_cfg(cfg, "data.noise_std", 0.003))

    for case_index_value in tqdm(indices, desc=f"train K={horizon}", leave=False):
        case_index = int(case_index_value)
        case = cases[case_index]
        if case.num_steps <= horizon:
            continue
        start, category = select_rollout_start(
            case.num_steps, horizon, sampling_scores[case_index], rng
        )
        resamples = 0
        maximum_resamples = int(get_cfg(cfg, "training.maximum_start_resamples", 16))
        while True:
            trajectory = cache.get_slice(case_index, start, start + horizon + 1)
            if is_admissible_training_window(trajectory, cfg):
                break
            resamples += 1
            if resamples > maximum_resamples:
                raise RuntimeError(
                    f"{case.case_id}: no admissible start found after "
                    f"{maximum_resamples} resamples"
                )
            start = int(rng.integers(0, case.num_steps - horizon))
            category = "admissible-resample"
        category_counts[category] = category_counts.get(category, 0) + 1
        totals["start_resamples"] = totals.get("start_resamples", 0.0) + (
            float(resamples) * horizon
        )
        position = trajectory.static.reference_position + trajectory.displacement[0]
        velocity = trajectory.velocity[0]
        moving = ~(trajectory.static.fixed_mask | trajectory.static.prescribed_mask)
        noise, noise_scale = geometry_safe_state_noise(
            trajectory.static,
            position,
            noise_std,
            smoothing_steps=int(get_cfg(cfg, "training.noise_smoothing_steps", 4)),
            minimum_j=float(get_cfg(cfg, "training.noise_min_j", 0.2)),
            max_backtracks=int(get_cfg(cfg, "training.noise_backtracks", 8)),
            maximum_invariant_growth=float(
                get_cfg(cfg, "training.noise_maximum_invariant_growth", 2.0)
            ),
        )
        position = position + noise
        totals["noise_scale"] = totals.get("noise_scale", 0.0) + noise_scale * horizon
        state = CHPState(position, velocity)
        optimizer.zero_grad(set_to_none=True)

        for chunk_start in range(0, horizon, max(tbptt_chunk, 1)):
            chunk_stop = min(horizon, chunk_start + max(tbptt_chunk, 1))
            chunk_loss = torch.zeros((), device=position.device)
            for offset in range(chunk_start, chunk_stop):
                step = offset
                dt = trajectory.times[step + 1] - trajectory.times[step]
                phase = trajectory.time_fraction[step]
                contact_pairs = contact_pairs_at(trajectory, state, cfg)
                external_force = external_force_at(trajectory, step, pressure_sign)
                prescribed_position = (
                    trajectory.static.reference_position
                    + trajectory.displacement[step + 1]
                )
                with _autocast(position.device, amp_dtype):
                    output = model(
                        trajectory.static,
                        state,
                        contact_pairs=contact_pairs,
                        external_force=external_force,
                        dt=dt,
                        time_fraction=phase,
                        prescribed_position=prescribed_position,
                        prescribed_velocity=trajectory.velocity[step + 1],
                    )
                    step_loss, metrics = chp_step_loss(
                        output, state, trajectory, step + 1, normalizers, cfg
                    )
                    exact_position = (
                        trajectory.static.reference_position
                        + trajectory.displacement[step + 1]
                    )
                    exact_nodal_stress, exact_cell_stress = model.nodal_stress_at(
                        trajectory.static, exact_position
                    )
                    exact_mask = ~trajectory.static.prescribed_mask
                    if not bool(exact_mask.any().item()):
                        exact_mask = torch.ones_like(exact_mask)
                    exact_constitutive, exact_metrics = _supervised_constitutive_loss(
                        exact_nodal_stress,
                        exact_cell_stress,
                        trajectory,
                        step + 1,
                        normalizers,
                        cfg,
                        nodal_mask=exact_mask,
                    )
                    step_loss = step_loss + float(
                        get_cfg(cfg, "loss.exact_constitutive", 0.5)
                    ) * exact_constitutive
                    metrics["exact_constitutive"] = exact_metrics["loss"]
                    metrics["exact_stress_physical"] = exact_metrics[
                        "stress_physical"
                    ]
                    metrics["exact_stress_tensor_supervision"] = exact_metrics[
                        "stress_tensor_supervision"
                    ]
                if not bool(torch.isfinite(step_loss).item()):
                    raise FloatingPointError(
                        f"non-finite CHP loss for {case.case_id} at frame {start + offset}"
                    )
                chunk_loss = chunk_loss + step_loss / float(horizon)
                _accumulate(totals, metrics)
                step_count += 1
                if stop_gradient:
                    state = CHPState(
                        output.next_position.detach(), output.next_velocity.detach()
                    )
                else:
                    state = CHPState(output.next_position, output.next_velocity)
            scaler.scale(chunk_loss).backward()
            state = CHPState(state.position.detach(), state.velocity.detach())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        clip = get_cfg(cfg, "training.grad_clip_norm", 1.0)
        if clip is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip)
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError(
                    f"non-finite CHP gradient for {case.case_id} at frame {start}"
                )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    averaged = {key: value / max(step_count, 1) for key, value in totals.items()}
    trajectory_count = max(sum(category_counts.values()), 1)
    for key, value in category_counts.items():
        averaged[f"sample_{key}"] = value / trajectory_count
    averaged["steps"] = float(step_count)
    averaged["horizon"] = float(horizon)
    return averaged


@torch.no_grad()
def evaluate_chp_rollouts(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    cfg: dict[str, Any],
    *,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Evaluate fixed, true autoregressive validation trajectories on CUDA."""

    model.eval()
    count = min(int(get_cfg(cfg, "validation.cases", 20)), len(cases))
    case_indices = np.linspace(0, len(cases) - 1, count).round().astype(int)
    requested_steps = get_cfg(cfg, "validation.steps", None)
    pressure_sign = float(get_cfg(cfg, "data.pressure_sign", 1.0))
    device = cache.device
    accum = {
        key: torch.zeros((), device=device, dtype=torch.float64)
        for key in (
            "u_error",
            "u_reference",
            "final_error",
            "final_reference",
            "stress_error",
            "stress_reference",
            "p95_error",
            "p95_reference",
            "cell_tensor_error",
            "cell_tensor_reference",
            "cell_tensor_p95_error",
            "cell_tensor_p95_reference",
            "cell_vm_error",
            "cell_vm_reference",
            "cell_vm_p95_error",
            "cell_vm_p95_reference",
            "penetration",
            "momentum",
            "energy",
            "backtrack",
            "integration_failure",
            "proposal_domain",
        )
    }
    evaluated_steps = 0
    attempted_steps = 0
    tensor_evaluated_steps = 0
    diverged = 0
    divergence_limit = float(get_cfg(cfg, "validation.divergence_position", 10.0))
    for case_index_value in tqdm(case_indices, desc="rollout-val", leave=False):
        case_index = int(case_index_value)
        case = cases[case_index]
        trajectory = cache.get(case_index)
        state = CHPState(
            trajectory.static.reference_position + trajectory.displacement[0],
            trajectory.velocity[0],
        )
        steps = case.num_steps - 1
        if requested_steps is not None:
            steps = min(steps, int(requested_steps))
        moving = ~(trajectory.static.fixed_mask | trajectory.static.prescribed_mask)
        if not bool(moving.any().item()):
            moving = ~trajectory.static.fixed_mask
        case_diverged = False
        for step in range(steps):
            attempted_steps += 1
            dt = trajectory.times[step + 1] - trajectory.times[step]
            span = (trajectory.times[-1] - trajectory.times[0]).clamp_min(1.0e-8)
            phase = (trajectory.times[step] - trajectory.times[0]) / span
            pairs = contact_pairs_at(trajectory, state, cfg)
            external = external_force_at(trajectory, step, pressure_sign)
            with _autocast(device, amp_dtype):
                output = model(
                    trajectory.static,
                    state,
                    contact_pairs=pairs,
                    external_force=external,
                    dt=dt,
                    time_fraction=phase,
                    prescribed_position=(
                        trajectory.static.reference_position
                        + trajectory.displacement[step + 1]
                    ),
                    prescribed_velocity=trajectory.velocity[step + 1],
                )
            state = CHPState(output.next_position, output.next_velocity)
            finite_step = (
                torch.isfinite(state.position).all()
                & torch.isfinite(state.velocity).all()
                & torch.isfinite(output.nodal_stress).all()
                & torch.isfinite(output.internal_force).all()
                & torch.isfinite(output.contact_force).all()
                & torch.isfinite(output.energy_diagnostics["work_energy_balance"])
                & (output.energy_diagnostics["integration_valid"] >= 0.5)
            )
            accum["backtrack"] += (
                output.energy_diagnostics["integration_update_scale"]
                < 1.0 - 1.0e-7
            ).double()
            accum["integration_failure"] += (
                output.energy_diagnostics["integration_valid"] < 0.5
            ).double()
            accum["proposal_domain"] += (
                output.energy_diagnostics["integration_domain_penalty"] > 0.0
            ).double()
            predicted_deformation = invariants(
                deformation_gradient(
                    state.position,
                    trajectory.static.cells,
                    trajectory.static.dm_inv,
                )
            )
            predicted_domain_valid = (
                torch.isfinite(predicted_deformation.j).all()
                & torch.isfinite(predicted_deformation.i2_bar).all()
                & (
                    predicted_deformation.j.min()
                    >= float(get_cfg(cfg, "validation.minimum_predicted_j", 1.0e-4))
                )
                & (
                    predicted_deformation.i2_bar.max()
                    <= float(
                        get_cfg(cfg, "validation.maximum_predicted_i2_bar", 1.0e6)
                    )
                )
            )
            if (
                not bool((finite_step & predicted_domain_valid).item())
                or float(state.position.abs().max().item()) > divergence_limit
            ):
                case_diverged = True
                state = CHPState(
                    torch.nan_to_num(
                        state.position,
                        nan=divergence_limit,
                        posinf=divergence_limit,
                        neginf=-divergence_limit,
                    ).clamp(-divergence_limit, divergence_limit),
                    torch.nan_to_num(
                        state.velocity,
                        nan=0.0,
                        posinf=divergence_limit,
                        neginf=-divergence_limit,
                    ).clamp(-divergence_limit, divergence_limit),
                )
                break
            exact_position = (
                trajectory.static.reference_position
                + trajectory.displacement[step + 1]
            )
            u_error = state.position[moving] - exact_position[moving]
            u_reference = trajectory.displacement[step + 1, moving]
            accum["u_error"] += u_error.double().square().sum()
            accum["u_reference"] += u_reference.double().square().sum()
            stress_mask = ~trajectory.static.prescribed_mask
            if not bool(stress_mask.any().item()):
                stress_mask = torch.ones_like(trajectory.static.prescribed_mask)
            target_stress = trajectory.stress[step + 1, stress_mask, :1]
            stress_error = output.nodal_stress[stress_mask, :1] - target_stress
            accum["stress_error"] += stress_error.double().square().sum()
            accum["stress_reference"] += target_stress.double().square().sum()
            threshold = torch.quantile(target_stress.abs().reshape(-1), 0.95)
            peak = target_stress.abs() >= threshold
            accum["p95_error"] += stress_error[peak].double().square().sum()
            accum["p95_reference"] += target_stress[peak].double().square().sum()
            if _trajectory_has_cell_stress_tensor(trajectory):
                predicted_tensor = _cauchy_to_tensor6(output.cell_stress_tensor)
                target_tensor = trajectory.cell_stress[step + 1].to(
                    predicted_tensor.dtype
                )
                tensor_error = predicted_tensor - target_tensor
                target_vm = _tensor6_von_mises(target_tensor)
                predicted_vm = _tensor6_von_mises(predicted_tensor)
                vm_error = predicted_vm - target_vm
                cell_threshold = torch.quantile(target_vm.abs(), 0.95)
                cell_peak = target_vm.abs() >= cell_threshold
                accum["cell_tensor_error"] += tensor_error.double().square().sum()
                accum["cell_tensor_reference"] += target_tensor.double().square().sum()
                accum["cell_tensor_p95_error"] += (
                    tensor_error[cell_peak].double().square().sum()
                )
                accum["cell_tensor_p95_reference"] += (
                    target_tensor[cell_peak].double().square().sum()
                )
                accum["cell_vm_error"] += vm_error.double().square().sum()
                accum["cell_vm_reference"] += target_vm.double().square().sum()
                accum["cell_vm_p95_error"] += (
                    vm_error[cell_peak].double().square().sum()
                )
                accum["cell_vm_p95_reference"] += (
                    target_vm[cell_peak].double().square().sum()
                )
                tensor_evaluated_steps += 1
            accum["penetration"] += output.energy_diagnostics[
                "max_penetration"
            ].double()
            resultant = output.internal_force + output.contact_force
            accum["momentum"] += torch.linalg.vector_norm(
                resultant.double().sum(0)
            )
            accum["energy"] += output.energy_diagnostics[
                "work_energy_balance"
            ].double().abs()
            evaluated_steps += 1
        exact_final = (
            trajectory.static.reference_position
            + trajectory.displacement[min(steps, case.num_steps - 1)]
        )
        final_error = state.position[moving] - exact_final[moving]
        accum["final_error"] += final_error.double().square().sum()
        accum["final_reference"] += trajectory.displacement[
            min(steps, case.num_steps - 1), moving
        ].double().square().sum()
        diverged += int(case_diverged)
    eps = torch.tensor(1.0e-12, device=device)
    result = {
        "moving_displacement_relative_rmse": float(
            torch.sqrt(accum["u_error"] / accum["u_reference"].clamp_min(eps)).item()
        ),
        "final_displacement_relative_rmse": float(
            torch.sqrt(
                accum["final_error"] / accum["final_reference"].clamp_min(eps)
            ).item()
        ),
        "stress_relative_rmse": float(
            torch.sqrt(
                accum["stress_error"] / accum["stress_reference"].clamp_min(eps)
            ).item()
        ),
        "stress_p95_relative_rmse": float(
            torch.sqrt(
                accum["p95_error"] / accum["p95_reference"].clamp_min(eps)
            ).item()
        ),
        "cell_stress_tensor_relative_rmse": (
            float(
                torch.sqrt(
                    accum["cell_tensor_error"]
                    / accum["cell_tensor_reference"].clamp_min(eps)
                ).item()
            )
            if tensor_evaluated_steps
            else float("inf")
        ),
        "cell_stress_tensor_p95_relative_rmse": (
            float(
                torch.sqrt(
                    accum["cell_tensor_p95_error"]
                    / accum["cell_tensor_p95_reference"].clamp_min(eps)
                ).item()
            )
            if tensor_evaluated_steps
            else float("inf")
        ),
        "cell_stress_vm_relative_rmse": (
            float(
                torch.sqrt(
                    accum["cell_vm_error"]
                    / accum["cell_vm_reference"].clamp_min(eps)
                ).item()
            )
            if tensor_evaluated_steps
            else float("inf")
        ),
        "cell_stress_vm_p95_relative_rmse": (
            float(
                torch.sqrt(
                    accum["cell_vm_p95_error"]
                    / accum["cell_vm_p95_reference"].clamp_min(eps)
                ).item()
            )
            if tensor_evaluated_steps
            else float("inf")
        ),
        "cell_stress_tensor_coverage": float(
            tensor_evaluated_steps / max(evaluated_steps, 1)
        ),
        "diverged_cases": float(diverged),
        "mean_penetration": float((accum["penetration"] / max(evaluated_steps, 1)).item()),
        "mean_momentum_residual": float((accum["momentum"] / max(evaluated_steps, 1)).item()),
        "mean_work_energy_error": float((accum["energy"] / max(evaluated_steps, 1)).item()),
        "integration_backtrack_fraction": float(
            (accum["backtrack"] / max(attempted_steps, 1)).item()
        ),
        "integration_failure_fraction": float(
            (accum["integration_failure"] / max(attempted_steps, 1)).item()
        ),
        "raw_proposal_domain_violation_fraction": float(
            (accum["proposal_domain"] / max(attempted_steps, 1)).item()
        ),
        "evaluated_steps": float(evaluated_steps),
    }
    if diverged:
        divergence_floor = 1.0e6 * diverged / max(count, 1)
        for key in ROLLOUT_METRIC_KEYS:
            value = float(result[key])
            result[key] = (
                max(value, divergence_floor) if math.isfinite(value) else divergence_floor
            )
    return result


@torch.no_grad()
def evaluate_teacher_forced_stress(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    cfg: dict[str, Any],
    *,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Evaluate the potential on exact geometry, preferring full cell tensors."""

    model.eval()
    case_count = min(
        int(get_cfg(cfg, "validation.teacher_stress_cases", 20)), len(cases)
    )
    frame_count = max(int(get_cfg(cfg, "validation.teacher_stress_frames", 16)), 1)
    case_indices = np.linspace(0, len(cases) - 1, case_count).round().astype(int)
    accum = {
        key: torch.zeros((), device=cache.device)
        for key in (
            "nodal_error",
            "nodal_reference",
            "nodal_peak_error",
            "nodal_peak_reference",
            "tensor_error",
            "tensor_reference",
            "tensor_peak_error",
            "tensor_peak_reference",
            "cell_vm_error",
            "cell_vm_reference",
            "cell_vm_peak_error",
            "cell_vm_peak_reference",
        )
    }
    requested_frames = 0
    admissible_frames = 0
    inverted_frames = 0
    near_singular_frames = 0
    extreme_i2_frames = 0
    tensor_label_frames = 0
    nodal_label_frames = 0
    for case_index_value in tqdm(
        case_indices, desc="teacher-stress", leave=False
    ):
        case_index = int(case_index_value)
        case = cases[case_index]
        trajectory = cache.get(case_index)
        frames = np.linspace(1, case.num_steps - 1, min(frame_count, case.num_steps - 1))
        for frame_value in np.unique(frames.round().astype(int)):
            frame = int(frame_value)
            requested_frames += 1
            exact_position = (
                trajectory.static.reference_position + trajectory.displacement[frame]
            )
            admissible, minimum_j, maximum_i2 = training_state_admissibility(
                trajectory.static, exact_position, cfg
            )
            if not admissible:
                if minimum_j <= 0.0:
                    inverted_frames += 1
                elif minimum_j < float(
                    get_cfg(cfg, "training.minimum_start_j", 1.0e-2)
                ):
                    near_singular_frames += 1
                if minimum_j > 0.0 and maximum_i2 > float(
                    get_cfg(cfg, "training.maximum_start_i2_bar", 1.0e5)
                ):
                    extreme_i2_frames += 1
                continue
            admissible_frames += 1
            with _autocast(cache.device, amp_dtype):
                nodal_prediction, cell_prediction = model.nodal_stress_at(
                    trajectory.static, exact_position
                )
            mask = ~trajectory.static.prescribed_mask
            if not bool(mask.any().item()):
                mask = torch.ones_like(mask)
            nodal_target = trajectory.stress[frame, mask, :1]
            nodal_residual = nodal_prediction[mask, :1] - nodal_target
            accum["nodal_error"] += nodal_residual.square().sum()
            accum["nodal_reference"] += nodal_target.square().sum()
            nodal_threshold = torch.quantile(nodal_target.abs().reshape(-1), 0.95)
            nodal_peak = nodal_target.abs() >= nodal_threshold
            accum["nodal_peak_error"] += nodal_residual[nodal_peak].square().sum()
            accum["nodal_peak_reference"] += nodal_target[nodal_peak].square().sum()
            nodal_label_frames += 1

            if _trajectory_has_cell_stress_tensor(trajectory):
                prediction6 = _cauchy_to_tensor6(cell_prediction)
                target6 = trajectory.cell_stress[frame].to(prediction6.dtype)
                residual6 = prediction6 - target6
                target_vm = _tensor6_von_mises(target6)
                prediction_vm = _tensor6_von_mises(prediction6)
                residual_vm = prediction_vm - target_vm
                peak_threshold = torch.quantile(target_vm.abs(), 0.95)
                cell_peak = target_vm.abs() >= peak_threshold
                accum["tensor_error"] += residual6.square().sum()
                accum["tensor_reference"] += target6.square().sum()
                accum["tensor_peak_error"] += residual6[cell_peak].square().sum()
                accum["tensor_peak_reference"] += target6[cell_peak].square().sum()
                accum["cell_vm_error"] += residual_vm.square().sum()
                accum["cell_vm_reference"] += target_vm.square().sum()
                accum["cell_vm_peak_error"] += residual_vm[cell_peak].square().sum()
                accum["cell_vm_peak_reference"] += target_vm[cell_peak].square().sum()
                tensor_label_frames += 1
    eps = torch.tensor(1.0e-12, device=cache.device)

    def relative(error_key: str, reference_key: str, frames: int) -> float:
        reference = accum[reference_key]
        if frames and bool((reference > eps).item()):
            return float(torch.sqrt(accum[error_key] / reference.clamp_min(eps)).item())
        return float("inf")

    nodal_relative = relative("nodal_error", "nodal_reference", nodal_label_frames)
    nodal_peak_relative = relative(
        "nodal_peak_error", "nodal_peak_reference", nodal_label_frames
    )
    tensor_relative = relative(
        "tensor_error", "tensor_reference", tensor_label_frames
    )
    tensor_peak_relative = relative(
        "tensor_peak_error", "tensor_peak_reference", tensor_label_frames
    )
    cell_vm_relative = relative(
        "cell_vm_error", "cell_vm_reference", tensor_label_frames
    )
    cell_vm_peak_relative = relative(
        "cell_vm_peak_error", "cell_vm_peak_reference", tensor_label_frames
    )
    if tensor_label_frames:
        source = "cell_tensor"
        primary_relative = tensor_relative
        primary_peak_relative = tensor_peak_relative
        label_frames = tensor_label_frames
    else:
        source = "nodal_scalar_vm_fallback"
        primary_relative = nodal_relative
        primary_peak_relative = nodal_peak_relative
        label_frames = nodal_label_frames
    return {
        # Stable API: these aliases always hold the preferred supervision source.
        "teacher_stress_relative_rmse": primary_relative,
        "teacher_stress_p95_relative_rmse": primary_peak_relative,
        "teacher_stress_source": source,
        "teacher_stress_label_coverage": float(
            label_frames / max(admissible_frames, 1)
        ),
        "teacher_cell_stress_tensor_relative_rmse": tensor_relative,
        "teacher_cell_stress_tensor_p95_relative_rmse": tensor_peak_relative,
        "teacher_cell_stress_vm_relative_rmse": cell_vm_relative,
        "teacher_cell_stress_vm_p95_relative_rmse": cell_vm_peak_relative,
        "teacher_cell_stress_tensor_frames": float(tensor_label_frames),
        "teacher_cell_stress_tensor_coverage": float(
            tensor_label_frames / max(admissible_frames, 1)
        ),
        "teacher_nodal_stress_relative_rmse": nodal_relative,
        "teacher_nodal_stress_p95_relative_rmse": nodal_peak_relative,
        "teacher_nodal_stress_frames": float(nodal_label_frames),
        "teacher_stress_admissible_coverage": float(
            admissible_frames / max(requested_frames, 1)
        ),
        "teacher_stress_requested_frames": float(requested_frames),
        "teacher_stress_admissible_frames": float(admissible_frames),
        "teacher_stress_inverted_frames": float(inverted_frames),
        "teacher_stress_near_singular_frames": float(near_singular_frames),
        "teacher_stress_extreme_i2_frames": float(extreme_i2_frames),
    }


def load_chp_checkpoint(
    path: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[CHPGNS, CHPNormalizers, dict[str, Any]]:
    """Load only schema-v2 checkpoints for GPU inference."""

    if device.type != "cuda":
        raise ValueError("CHP-GNS inference is GPU-only")
    checkpoint = _torch_load(path, device)
    validate_chp_checkpoint_semantics(
        checkpoint, source=path, require_scientific_gate=True
    )
    effective_cfg = checkpoint.get("config", cfg)
    model = CHPGNS(
        effective_cfg, material_dim=int(checkpoint.get("material_dim", 0))
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizers = CHPNormalizers.from_state_dict(checkpoint["normalizers"]).to(device)
    return model, normalizers, checkpoint


def validate_chp_checkpoint_semantics(
    checkpoint: dict[str, Any],
    *,
    source: str | Path = "checkpoint",
    require_scientific_gate: bool = False,
) -> None:
    """Reject schema-v2 files whose dynamics had older force semantics."""

    if int(checkpoint.get("schema_version", 0)) != CHPGNS.checkpoint_schema_version:
        raise ValueError(
            f"legacy checkpoint cannot populate the CHP physical decoder: {source}"
        )
    expected = {
        "dynamics_schema_version": CHPGNS.dynamics_schema_version,
        "residual_parameterization": CHPGNS.residual_parameterization,
        "residual_gate": CHPGNS.residual_gate,
    }
    mismatched = {
        key: checkpoint.get(key)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "checkpoint dynamics semantics are missing or incompatible: "
            f"{source}; found={mismatched}, expected={expected}. "
            "Start a fresh run instead of reinterpreting a legacy residual head."
        )
    if require_scientific_gate:
        source_path = Path(source)
        if source_path != Path("checkpoint"):
            _assert_no_gate_failure_artifact(
                source_path.resolve().parent,
                context="inference/evaluation",
            )
        status = str(checkpoint.get("scientific_gate_status", "missing"))
        if status not in {"passed", "not_required"}:
            raise ValueError(
                "checkpoint did not pass the configured scientific gates: "
                f"{source}; scientific_gate_status={status!r}"
            )


def _build_chp_datasets(
    cfg: dict[str, Any],
) -> tuple[ValveGraphDataset, ValveGraphDataset | None]:
    root = get_cfg(cfg, "data.root", get_cfg(cfg, "data.case_dir", None))
    if root is None:
        raise ValueError("data.root or data.case_dir is required")
    split_file = get_cfg(
        cfg, "data.split_file", get_cfg(cfg, "data.case_split_file", None)
    )
    train_split = str(
        get_cfg(cfg, "data.train_split", get_cfg(cfg, "data.train_case_split", "train"))
    )
    val_split = str(
        get_cfg(cfg, "data.val_split", get_cfg(cfg, "data.val_case_split", "val"))
    )
    if split_file and Path(split_file).exists():
        train = ValveGraphDataset(root, cfg, split=train_split, split_file=split_file)
        val = ValveGraphDataset(root, cfg, split=val_split, split_file=split_file)
        return train, val
    return ValveGraphDataset(root, cfg), None


def _checkpoint_payload(
    model: CHPGNS,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
    epoch: int,
    score: float,
    best_score: float,
    rollout_metrics: dict[str, float],
    material_dim: int,
    *,
    scientific_gate_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHPGNS.checkpoint_schema_version,
        "dynamics_schema_version": CHPGNS.dynamics_schema_version,
        "residual_parameterization": CHPGNS.residual_parameterization,
        "residual_gate": CHPGNS.residual_gate,
        "architecture": "CHP-GNS",
        "scientific_gate_status": str(scientific_gate_status),
        "epoch": int(epoch),
        "score": float(score),
        "best_score": float(best_score),
        "rollout_metrics": rollout_metrics,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "normalizers": normalizers.state_dict(),
        "material_dim": int(material_dim),
        "config": cfg,
    }


def _require_cuda(cfg: dict[str, Any]) -> torch.device:
    requested = str(get_cfg(cfg, "training.device", "cuda")).lower()
    if requested not in {"cuda", "auto"}:
        raise ValueError("CHP-GNS training is intentionally GPU-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CHP-GNS training and contact search")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    return device


def _amp_dtype(cfg: dict[str, Any]) -> torch.dtype:
    if not bool(get_cfg(cfg, "training.amp", True)):
        return torch.float32
    name = str(get_cfg(cfg, "training.amp_dtype", "bfloat16")).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported AMP dtype: {name}")


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _warmup_cosine(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1.0e-3)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _limited_count(total: int, requested: Any) -> int:
    return total if requested is None else min(total, max(int(requested), 1))


def _teacher_gate_epoch(stages: Iterable[dict[str, int]] | None) -> int:
    schedule = list(stages or (
        {"horizon": 1, "epochs": 4},
        {"horizon": 2, "epochs": 3},
        {"horizon": 4, "epochs": 3},
        {"horizon": 8, "epochs": 3},
        {"horizon": 16, "epochs": 3},
    ))
    total = 0
    for stage in schedule:
        if int(stage["horizon"]) != 1:
            break
        total += int(stage["epochs"])
    return max(total, 1)


def _scientific_gate_status(
    epoch: int,
    stages: Iterable[dict[str, int]] | None,
    cfg: dict[str, Any],
) -> str:
    required = bool(
        get_cfg(cfg, "validation.enforce_teacher_stress_gate", True)
    ) or bool(get_cfg(cfg, "validation.enforce_rollout_pilot_gate", False))
    if not required:
        return "not_required"
    return "passed" if int(epoch) >= _teacher_gate_epoch(stages) else "pending"


def _assert_no_gate_failure_artifact(
    directory: str | Path,
    *,
    context: str,
) -> None:
    root = Path(directory)
    markers = (
        root / "teacher_stress_gate_failure.json",
        root / "rollout_pilot_gate_failure.json",
    )
    failed = [marker for marker in markers if marker.is_file()]
    if failed:
        names = ", ".join(marker.name for marker in failed)
        raise RuntimeError(
            f"CHP-GNS scientific gate already failed; refusing {context}: {names}. "
            "Use a new output directory after revising the model or data."
        )


def _enforce_scientific_gates(
    epoch: int,
    stages: Iterable[dict[str, int]] | None,
    teacher_metrics: dict[str, Any],
    rollout_metrics: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
) -> None:
    gate_epoch = _teacher_gate_epoch(stages)
    if int(epoch) != gate_epoch:
        return
    teacher_relative = float(
        teacher_metrics.get("teacher_stress_relative_rmse", float("inf"))
    )
    teacher_threshold = float(
        get_cfg(cfg, "validation.teacher_stress_threshold", 0.50)
    )
    if (
        bool(get_cfg(cfg, "validation.enforce_teacher_stress_gate", True))
        and teacher_relative >= teacher_threshold
    ):
        teacher_source = str(
            teacher_metrics.get("teacher_stress_source", "unavailable")
        )
        failure = {
            "epoch": int(epoch),
            "teacher_stress_relative_rmse": teacher_relative,
            "teacher_stress_source": teacher_source,
            "teacher_stress_label_coverage": teacher_metrics.get(
                "teacher_stress_label_coverage", 0.0
            ),
            "threshold": teacher_threshold,
            "action": (
                "stop rollout curriculum and revise constitutive model/data"
                if teacher_source == "cell_tensor"
                else "stop rollout curriculum and add full tensor labels"
            ),
        }
        _save_json(output_dir / "teacher_stress_gate_failure.json", failure)
        raise RuntimeError(
            "teacher-forced stress gate failed: "
            f"{teacher_relative:.4g} >= {teacher_threshold:.4g}"
        )
    if not bool(get_cfg(cfg, "validation.enforce_rollout_pilot_gate", False)):
        return
    moving_relative = float(
        rollout_metrics.get("moving_displacement_relative_rmse", float("inf"))
    )
    stress_relative = float(
        rollout_metrics.get("stress_relative_rmse", float("inf"))
    )
    diverged_cases = float(rollout_metrics.get("diverged_cases", float("inf")))
    moving_threshold = float(
        get_cfg(cfg, "validation.pilot_moving_relative_rmse_threshold", 0.80)
    )
    stress_threshold = float(
        get_cfg(cfg, "validation.pilot_stress_relative_rmse_threshold", 0.65)
    )
    passed = (
        moving_relative < moving_threshold
        and stress_relative < stress_threshold
        and diverged_cases == 0.0
    )
    if not passed:
        failure = {
            "epoch": int(epoch),
            "moving_displacement_relative_rmse": moving_relative,
            "moving_threshold": moving_threshold,
            "stress_relative_rmse": stress_relative,
            "stress_threshold": stress_threshold,
            "diverged_cases": diverged_cases,
            "action": "stop before K=2 and revise force identifiability",
        }
        _save_json(output_dir / "rollout_pilot_gate_failure.json", failure)
        raise RuntimeError(
            "K=1 rollout pilot gate failed: "
            f"moving={moving_relative:.4g}, stress={stress_relative:.4g}, "
            f"diverged={diverged_cases:.0f}"
        )


def _curriculum_stage_ends(stages: Iterable[dict[str, int]] | None) -> set[int]:
    schedule = list(stages or (
        {"horizon": 1, "epochs": 4},
        {"horizon": 2, "epochs": 3},
        {"horizon": 4, "epochs": 3},
        {"horizon": 8, "epochs": 3},
        {"horizon": 16, "epochs": 3},
    ))
    total = 0
    ends = set()
    for stage in schedule:
        total += int(stage["epochs"])
        ends.add(total)
    return ends


def _accumulate(totals: dict[str, float], values: dict[str, torch.Tensor]) -> None:
    for key, value in values.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().item())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_resume(cfg: dict[str, Any], output_dir: Path) -> Path | None:
    value = get_cfg(cfg, "training.resume_from", None)
    if value in {None, False, "none"}:
        return None
    if str(value).lower() == "auto":
        path = output_dir / "latest.pt"
        return path if path.exists() else None
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _torch_load(path: str | Path, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)
