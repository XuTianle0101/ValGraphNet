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
from valgraphnet.mechanics import (
    deformation_gradient,
    invariants,
    project_cell_to_nodes,
    von_mises,
)
from valgraphnet.physical_evaluation import (
    CELL_TENSOR_STRESS_SOURCE,
    NODAL_STRESS_FALLBACK_SOURCE,
    validate_reference_protocol,
)
from valgraphnet.stress_transform import AsinhStressTransform, robust_stress_loss


ROLLOUT_METRIC_KEYS = (
    "moving_displacement_relative_rmse",
    "final_displacement_relative_rmse",
    "stress_relative_rmse",
    "stress_p95_relative_rmse",
)
FAILED_RELATIVE_METRIC = 1.0e30


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


def stable_clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter], max_norm: float
) -> torch.Tensor:
    """Clip finite gradients without overflowing the norm reduction.

    ``torch.nn.utils.clip_grad_norm_`` may report an infinite FP32 norm even
    when every gradient entry is finite.  Compute a dimensionless norm after
    scaling by the largest entry, accumulate the small list of per-tensor norms
    in FP64, and only then restore the physical magnitude.  Actual NaN/Inf
    entries are rejected before any in-place scaling, so numerical corruption
    is not confused with a merely large finite update.
    """

    limit = float(max_norm)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("max_norm must be finite and non-negative")
    parameter_list = list(parameters)
    gradients: list[torch.Tensor] = []
    gradient_indices: list[int] = []
    for index, parameter in enumerate(parameter_list):
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            raise TypeError("stable_clip_grad_norm_ does not support sparse gradients")
        if gradient.numel():
            gradients.append(gradient)
            gradient_indices.append(index)
    if not gradients:
        device = parameter_list[0].device if parameter_list else torch.device("cpu")
        return torch.zeros((), device=device, dtype=torch.float64)

    finite_flags = torch.stack(
        [torch.isfinite(gradient).all() for gradient in gradients]
    )
    if not bool(finite_flags.all().item()):
        failed = [
            gradient_indices[index]
            for index, finite in enumerate(finite_flags.detach().cpu().tolist())
            if not finite
        ]
        raise FloatingPointError(
            "individual non-finite gradient values in parameter indices "
            f"{failed}"
        )
    largest = torch.stack(
        [gradient.detach().abs().max() for gradient in gradients]
    ).max()
    safe_largest = largest.clamp_min(torch.finfo(largest.dtype).tiny)
    scaled_norms = torch.stack(
        [
            torch.linalg.vector_norm(gradient.detach() / safe_largest)
            for gradient in gradients
        ]
    ).double()
    total_norm = largest.double() * torch.linalg.vector_norm(scaled_norms)
    if not bool(torch.isfinite(total_norm).item()):
        raise FloatingPointError(
            "gradient norm reduction is non-finite although every individual "
            "gradient value is finite"
        )
    denominator = total_norm.clamp_min(torch.finfo(torch.float64).tiny)
    coefficient = torch.clamp(
        total_norm.new_tensor(limit) / denominator,
        max=1.0,
    )
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm


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
    solver_nodal_force: torch.Tensor
    has_solver_nodal_force: bool


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
            solver_nodal_force=self._tensor(
                case.solver_nodal_force[frame_slice], self.device, torch.float32
            ),
            has_solver_nodal_force=case.has_solver_nodal_force,
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
        fixed = np.asarray(
            getattr(case, "fixed_mask", np.zeros(case.num_nodes, dtype=bool)),
            dtype=bool,
        )
        prescribed = np.asarray(
            getattr(
                case,
                "prescribed_mask",
                np.zeros(case.num_nodes, dtype=bool),
            ),
            dtype=bool,
        )
        moving = ~(fixed | prescribed)
        free_nodes = np.flatnonzero(moving)
        supervised_stress_nodes = np.flatnonzero(~prescribed)
        stress_node_count = min(node_count, supervised_stress_nodes.size)
        stress_nodes = (
            supervised_stress_nodes[
                np.linspace(
                    0, supervised_stress_nodes.size - 1, stress_node_count
                )
                .round()
                .astype(int)
            ]
            if stress_node_count
            else nodes
        )
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
                        np.array(
                            case.stress[frame + 1, stress_nodes, :1], copy=True
                        )
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
    maximum_i1_bar: float | None = None,
    maximum_i2_bar: float | None = None,
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
    if float(maximum_invariant_growth) < 1.0:
        raise ValueError("maximum_invariant_growth must be at least one")
    base_deformation = deformation_gradient(position, static.cells, static.dm_inv)
    base_state = invariants(base_deformation)
    base_is_finite = (
        torch.isfinite(base_deformation).all()
        & torch.isfinite(base_state.c).all()
        & torch.isfinite(base_state.i1).all()
        & torch.isfinite(base_state.i1_bar).all()
        & torch.isfinite(base_state.i2).all()
        & torch.isfinite(base_state.i2_bar).all()
        & torch.isfinite(base_state.j).all()
    )
    if not bool(base_is_finite.item()):
        raise FloatingPointError("cannot add noise to a non-finite tetrahedral state")
    base_j = base_state.j.min()
    admissible_j = min(
        float(minimum_j),
        max(0.5 * float(base_j.item()), 1.0e-4),
    )
    admissible_i1 = float(base_state.i1_bar.max().item()) * float(
        maximum_invariant_growth
    )
    admissible_i2 = float(base_state.i2_bar.max().item()) * float(
        maximum_invariant_growth
    )
    if maximum_i1_bar is not None:
        if float(maximum_i1_bar) <= 0.0 or math.isnan(float(maximum_i1_bar)):
            raise ValueError("maximum_i1_bar must be positive or None")
        admissible_i1 = min(admissible_i1, float(maximum_i1_bar))
    if maximum_i2_bar is not None:
        if float(maximum_i2_bar) <= 0.0 or math.isnan(float(maximum_i2_bar)):
            raise ValueError("maximum_i2_bar must be positive or None")
        admissible_i2 = min(admissible_i2, float(maximum_i2_bar))
    scale = 1.0
    for _ in range(max(int(max_backtracks), 0) + 1):
        candidate = position + scale * noise
        candidate_deformation = deformation_gradient(
            candidate, static.cells, static.dm_inv
        )
        candidate_state = invariants(candidate_deformation)
        admissible = (
            torch.isfinite(candidate_deformation).all()
            & torch.isfinite(candidate_state.c).all()
            & torch.isfinite(candidate_state.i1).all()
            & torch.isfinite(candidate_state.i1_bar).all()
            & torch.isfinite(candidate_state.i2).all()
            & torch.isfinite(candidate_state.i2_bar).all()
            & torch.isfinite(candidate_state.j).all()
            & (candidate_state.j.min() >= admissible_j)
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

    deformation = deformation_gradient(position, static.cells, static.dm_inv)
    state = invariants(deformation)
    minimum_j = float(get_cfg(cfg, "training.minimum_start_j", 1.0e-2))
    maximum_i1 = float(
        get_cfg(cfg, "training.maximum_start_i1_bar", float("inf"))
    )
    maximum_i2 = float(get_cfg(cfg, "training.maximum_start_i2_bar", 1.0e5))
    minimum_observed_j = state.j.min()
    maximum_observed_i2 = state.i2_bar.max()
    valid = (
        torch.isfinite(deformation).all()
        & torch.isfinite(state.c).all()
        & torch.isfinite(state.j).all()
        & torch.isfinite(state.i1).all()
        & torch.isfinite(state.i1_bar).all()
        & torch.isfinite(state.i2).all()
        & torch.isfinite(state.i2_bar).all()
        & (minimum_observed_j >= minimum_j)
        & (state.i1_bar.max() <= maximum_i1)
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
        gate_mse_weight=float(
            get_cfg(cfg, "loss.rollout_stress_gate_mse_weight", 0.0)
        ),
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
        "stress_gate_mse": stress_parts["stress_gate_mse"],
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
    """Return targets for one global semi-implicit state update.

    The exported deforming_plate velocity is the backward position difference,
    so with ``dt=1`` this gives ``v*=V[t+1]`` and
    ``a*=V[t+1]-V[t]`` on an exact input state.  Contact iterations refine the
    force but never change this single-update kinematic contract.
    """

    step_dt = torch.as_tensor(
        dt,
        device=input_state.position.device,
        dtype=input_state.position.dtype,
    )
    if step_dt.numel() != 1 or not bool(torch.isfinite(step_dt).item()):
        raise ValueError("dt must be a finite scalar")
    if not bool((step_dt > 0.0).item()):
        raise ValueError("dt must be positive")
    target_velocity = (exact_next_position - input_state.position) / step_dt
    target_acceleration = (
        target_velocity - input_state.velocity
    ) / step_dt
    return target_velocity, target_acceleration


def minimax_checkpoint_score(
    metrics: dict[str, Any],
    native_reference: dict[str, Any] | None,
) -> float:
    """Worst relative degradation; no metric can be traded for another."""

    references = native_reference or {}
    metric_source = metrics.get("stress_metric_source")
    reference_source = references.get("stress_metric_source")
    allowed_sources = {
        CELL_TENSOR_STRESS_SOURCE,
        NODAL_STRESS_FALLBACK_SOURCE,
    }
    if metric_source is not None and metric_source not in allowed_sources:
        raise ValueError(f"unsupported rollout stress metric source: {metric_source!r}")
    if reference_source is not None and reference_source not in allowed_sources:
        raise ValueError(
            f"unsupported native stress metric source: {reference_source!r}"
        )
    if reference_source is not None and metric_source != reference_source:
        raise ValueError(
            "checkpoint and native reference use different stress metrics: "
            f"checkpoint={metric_source!r}, native={reference_source!r}"
        )
    if (
        native_reference is not None
        and metric_source == CELL_TENSOR_STRESS_SOURCE
        and reference_source is None
    ):
        raise ValueError(
            "cell-tensor checkpoint selection requires a source-compatible "
            "native reference; use absolute_validation or regenerate the reference"
        )
    ratios = []
    for key in ROLLOUT_METRIC_KEYS:
        value = float(metrics[key])
        reference = float(references.get(key, 1.0))
        if (
            not math.isfinite(value)
            or not math.isfinite(reference)
            or reference <= 0.0
        ):
            return FAILED_RELATIVE_METRIC
        ratios.append(value / reference)
    return max(ratios)


def load_native_reference(cfg: dict[str, Any]) -> dict[str, Any] | None:
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
    result: dict[str, Any] = {
        key: float(inline[key]) for key in ROLLOUT_METRIC_KEYS
    }
    source = inline.get("stress_metric_source")
    if source is None and isinstance(source_payload.get("metric_definition"), dict):
        source = source_payload["metric_definition"].get("stress_source")
    if source is not None:
        result["stress_metric_source"] = str(source)
    return result


def _constitutive_parameters(
    model: CHPGNS, *, include_stress_scale: bool = True
) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    if include_stress_scale:
        parameters.append(model.log_stress_scale)
    parameters.extend(model.potential.parameters())
    if model.material_potential is not None:
        parameters.extend(model.material_potential.parameters())
    return parameters


def pooled_positive_constitutive_scale(
    prediction_square: torch.Tensor,
    prediction_target: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Return the constrained scale minimizing the pooled squared error."""

    if float(minimum) <= 0.0 or float(maximum) < float(minimum):
        raise ValueError("constitutive scale bounds must satisfy 0 < minimum <= maximum")
    square = torch.as_tensor(prediction_square)
    cross = torch.as_tensor(
        prediction_target, device=square.device, dtype=square.dtype
    )
    if square.numel() != 1 or cross.numel() != 1:
        raise ValueError("pooled constitutive moments must be scalar")
    if not bool(torch.isfinite(square).item()) or float(square) <= 0.0:
        raise ValueError("pooled prediction square must be finite and positive")
    if not bool(torch.isfinite(cross).item()):
        raise ValueError("pooled prediction-target moment must be finite")
    return (cross / square).clamp(min=float(minimum), max=float(maximum))


def winsorized_relative_constitutive_scale(
    frame_moments: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    leverage_quantile: float,
    minimum: float,
    maximum: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit a positive scale without letting one predicted frame own the fit.

    Each frame is first normalized by its target energy, which removes mesh
    size and stress-amplitude weighting from this train-only calibration.
    Predicted leverage is then winsorized by downweighting frames above the
    requested quantile.  The reported scientific gate remains the untouched
    pooled physical rRMSE over every admissible validation label.
    """

    if not 0.0 < float(leverage_quantile) <= 1.0:
        raise ValueError("leverage_quantile must be in (0, 1]")
    if not frame_moments:
        raise ValueError("winsorized scale fitting requires frame moments")
    square = torch.stack([item[0] for item in frame_moments]).float()
    cross = torch.stack([item[1] for item in frame_moments]).float()
    reference = torch.stack([item[2] for item in frame_moments]).float()
    eps = torch.finfo(square.dtype).eps
    valid = (
        torch.isfinite(square)
        & torch.isfinite(cross)
        & torch.isfinite(reference)
        & (square > eps)
        & (reference > eps)
    )
    if not bool(valid.any().item()):
        raise ValueError("winsorized scale fitting has no finite non-zero frames")
    square = square[valid]
    cross = cross[valid]
    reference = reference[valid]
    relative_square = square / reference
    relative_cross = cross / reference
    leverage_cap = torch.quantile(relative_square, float(leverage_quantile))
    leverage_cap = leverage_cap.clamp_min(eps)
    weights = torch.minimum(
        torch.ones_like(relative_square),
        leverage_cap / relative_square.clamp_min(eps),
    )
    weighted_square = weights * relative_square
    weighted_cross = weights * relative_cross
    factor = pooled_positive_constitutive_scale(
        weighted_square.sum(),
        weighted_cross.sum(),
        minimum=minimum,
        maximum=maximum,
    )
    effective_frames = weights.sum().square() / weights.square().sum().clamp_min(eps)
    return factor, {
        "leverage_quantile": float(leverage_quantile),
        "leverage_cap": float(leverage_cap.item()),
        "maximum_raw_leverage_share": float(
            (relative_square.max() / relative_square.sum().clamp_min(eps)).item()
        ),
        "maximum_winsorized_leverage_share": float(
            (weighted_square.max() / weighted_square.sum().clamp_min(eps)).item()
        ),
        "effective_frames": float(effective_frames.item()),
    }


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
    positive_frame_moments: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
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
            exact_fields = model.constitutive_fields(
                trajectory.static, exact_position
            )
            cell_prediction = exact_fields.cauchy_stress
            nodal_prediction = _nodal_stress_from_cell_tensor(
                cell_prediction, trajectory.static
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
                if bool((frame_prediction_target > 0.0).item()):
                    positive_frame_moments.append(frame_moments[-1])
            admissible_frames += 1
    if not frame_moments:
        raise RuntimeError("no admissible non-zero frames for constitutive calibration")
    frame_factors = (
        torch.stack(
            [cross / square for square, cross, _ in positive_frame_moments]
        )
        if positive_frame_moments
        else None
    )
    minimum_factor = float(
        get_cfg(cfg, "constitutive_pretraining.minimum_scale_factor", 1.0e-3)
    )
    maximum_factor = float(
        get_cfg(cfg, "constitutive_pretraining.maximum_scale_factor", 1.0e4)
    )
    # The scientific gate remains an untouched pooled physical rRMSE.  The
    # train-only initializer may use a robust estimator so a single malformed
    # predicted frame cannot collapse the modulus for every other trajectory;
    # both pooled before/after errors are still recorded below.
    estimator = str(
        get_cfg(cfg, "constitutive_pretraining.scale_estimator", "pooled")
    ).lower()
    estimator_diagnostics: dict[str, float] = {}
    if estimator == "pooled":
        factor = pooled_positive_constitutive_scale(
            prediction_square,
            prediction_target,
            minimum=minimum_factor,
            maximum=maximum_factor,
        )
        estimator_name = "constrained_pooled_least_squares"
    elif estimator == "winsorized_relative":
        factor, estimator_diagnostics = winsorized_relative_constitutive_scale(
            frame_moments,
            leverage_quantile=float(
                get_cfg(
                    cfg,
                    "constitutive_pretraining.calibration_leverage_quantile",
                    0.99,
                )
            ),
            minimum=minimum_factor,
            maximum=maximum_factor,
        )
        estimator_name = "winsorized_frame_relative_least_squares"
    else:
        raise ValueError(f"unsupported constitutive scale estimator: {estimator!r}")
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
        "scale_estimator": estimator_name,
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
        "frame_scale_q05": (
            float(torch.quantile(frame_factors, 0.05).item())
            if frame_factors is not None
            else None
        ),
        "frame_scale_median": (
            float(frame_factors.median().item())
            if frame_factors is not None
            else None
        ),
        "frame_scale_q95": (
            float(torch.quantile(frame_factors, 0.95).item())
            if frame_factors is not None
            else None
        ),
        "admissible_frames": float(admissible_frames),
        "requested_frames": float(requested_frames),
        "admissible_coverage": float(admissible_frames / max(requested_frames, 1)),
        "stress_source": (
            "cell_tensor" if tensor_frames else "nodal_scalar_vm_fallback"
        ),
        "cell_tensor_frames": float(tensor_frames),
        "nodal_scalar_frames": float(scalar_frames),
        "cell_tensor_coverage": float(tensor_frames / max(admissible_frames, 1)),
        **estimator_diagnostics,
    }


def _exact_constitutive_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
    *,
    gate_mse_weight: float | None = None,
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
    normalized_residual = (prediction - target) / scale
    physical = F.huber_loss(
        normalized_residual,
        torch.zeros_like(prediction),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    # The scientific gate is a pooled physical rRMSE.  Its denominator is
    # fixed by the labels, so a non-saturating squared residual directly
    # targets its numerator under the training sampler.  Keep asinh+Huber for
    # the heavy-tailed bulk and add this term instead of replacing it.
    gate_mse = normalized_residual.square().mean()
    effective_gate_weight = (
        float(get_cfg(cfg, "loss.stress_gate_mse_weight", 0.0))
        if gate_mse_weight is None
        else float(gate_mse_weight)
    )
    total = transformed + float(
        get_cfg(cfg, "loss.stress_physical_weight", 0.25)
    ) * physical + effective_gate_weight * gate_mse
    return total, {
        "loss": total.detach(),
        "stress_transformed": transformed.detach(),
        "stress_base": parts["stress_base"],
        "stress_physical": physical.detach(),
        "stress_gate_mse": gate_mse.detach(),
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
    *,
    gate_mse_weight: float | None = None,
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
    tensor_normalized_residual = (prediction6 - target) / tensor_scale
    tensor_physical = F.huber_loss(
        tensor_normalized_residual,
        torch.zeros_like(prediction6),
        delta=huber_delta,
    )
    tensor_gate_mse = tensor_normalized_residual.square().mean()
    effective_gate_weight = (
        float(get_cfg(cfg, "loss.stress_gate_mse_weight", 0.0))
        if gate_mse_weight is None
        else float(gate_mse_weight)
    )
    tensor_loss = tensor_transformed + float(
        get_cfg(cfg, "loss.stress_physical_weight", 0.25)
    ) * tensor_physical + effective_gate_weight * tensor_gate_mse

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
        "stress_gate_mse": tensor_gate_mse.detach(),
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
    gate_mse_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer complete cell tensors, with an explicit nodal-VM fallback."""

    if _trajectory_has_cell_stress_tensor(trajectory):
        return _cell_tensor_constitutive_loss(
            cell_prediction,
            trajectory.cell_stress[int(frame)],
            normalizers,
            cfg,
            gate_mse_weight=gate_mse_weight,
        )
    return _exact_constitutive_loss(
        nodal_prediction[nodal_mask, :1],
        trajectory.stress[int(frame), nodal_mask, :1],
        normalizers,
        cfg,
        gate_mse_weight=gate_mse_weight,
    )


def _nodal_stress_from_cell_tensor(
    cell_stress: torch.Tensor,
    static: CHPStatic,
) -> torch.Tensor:
    return project_cell_to_nodes(
        von_mises(cell_stress)[:, None],
        static.cells,
        static.num_nodes,
        weights=static.volume,
    )


def reaction_equilibrium_loss(
    internal_force: torch.Tensor,
    solver_nodal_force: torch.Tensor,
    reaction_mask: torch.Tensor,
    *,
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dimensionless fixed-reaction equilibrium loss and physical rRMSE.

    CalculiX ``FORC`` in HyperContact contains support reactions on the fixed
    block surface.  With the internal-force sign convention used here, static
    equilibrium is ``f_internal + f_reaction = 0`` on those nodes.  Contact
    forces are intentionally not read as model inputs.
    """

    if internal_force.shape != solver_nodal_force.shape:
        raise ValueError("internal and solver nodal forces must have matching shapes")
    if reaction_mask.shape != internal_force.shape[:-1]:
        raise ValueError("reaction_mask must have shape [N]")
    if not bool(reaction_mask.any().item()):
        zero = internal_force.new_zeros(())
        return zero, zero
    target = solver_nodal_force[reaction_mask]
    residual = internal_force[reaction_mask] + target
    reference_square = target.square().sum()
    if not bool((reference_square > 1.0e-20).item()):
        zero = internal_force.new_zeros(())
        return zero, zero
    scale = target.square().mean().sqrt().detach().clamp_min(1.0e-12)
    loss = F.huber_loss(
        residual / scale,
        torch.zeros_like(residual),
        delta=float(huber_delta),
    )
    relative = torch.sqrt(
        residual.square().sum() / reference_square.clamp_min(1.0e-20)
    )
    return loss, relative


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
    closed_form_scale = bool(
        get_cfg(cfg, "constitutive_pretraining.closed_form_scale_each_epoch", False)
    )
    parameters = _constitutive_parameters(
        model, include_stress_scale=not closed_form_scale
    )
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
            # Evaluate the shared potential once: stress supervision and the
            # fixed-support equilibrium term must use the exact same
            # constitutive state (and autograd graph).
            exact_fields = model.constitutive_fields(
                trajectory.static, exact_position
            )
            cell_prediction = exact_fields.cauchy_stress
            nodal_prediction = _nodal_stress_from_cell_tensor(
                cell_prediction, trajectory.static
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
            reaction_weight = float(
                get_cfg(cfg, "loss.reaction_equilibrium", 0.0)
            )
            if reaction_weight > 0.0 and not trajectory.has_solver_nodal_force:
                raise ValueError(
                    "reaction-equilibrium supervision requires solver_nodal_force.npy"
                )
            if trajectory.has_solver_nodal_force:
                reaction_loss, reaction_relative = reaction_equilibrium_loss(
                    exact_fields.internal_force,
                    trajectory.solver_nodal_force[frame],
                    trajectory.static.fixed_mask,
                    huber_delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
                )
            else:
                reaction_loss = loss.new_zeros(())
                reaction_relative = loss.new_zeros(())
            loss = loss + reaction_weight * reaction_loss
            metrics["reaction_equilibrium"] = reaction_loss.detach()
            metrics["reaction_equilibrium_relative_rmse"] = (
                reaction_relative.detach()
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
        try:
            stable_clip_grad_norm_(
                parameters,
                float(get_cfg(cfg, "constitutive_pretraining.grad_clip_norm", 10.0)),
            )
        except FloatingPointError as exc:
            raise FloatingPointError(
                f"non-finite constitutive pretraining gradient: {exc}"
            ) from exc
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
    closed_form_scale = bool(
        get_cfg(cfg, "constitutive_pretraining.closed_form_scale_each_epoch", False)
    )
    optimizer = torch.optim.Adam(
        _constitutive_parameters(
            model, include_stress_scale=not closed_form_scale
        ),
        lr=float(get_cfg(cfg, "constitutive_pretraining.lr", 1.0e-3)),
        weight_decay=0.0,
    )
    epochs = max(int(get_cfg(cfg, "constitutive_pretraining.epochs", 2)), 0)
    original_scale_requires_grad = bool(model.log_stress_scale.requires_grad)
    if closed_form_scale:
        model.log_stress_scale.requires_grad_(False)
    try:
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
            epoch_calibration = None
            if closed_form_scale:
                epoch_calibration = calibrate_constitutive_modulus(
                    model, train_cases, train_cache, cfg
                )
                train_metrics["post_calibration_physical_modulus"] = float(
                    epoch_calibration["physical_modulus"]
                )
            validation = (
                evaluate_teacher_forced_stress(
                    model, val_cases, val_cache, cfg, amp_dtype=amp_dtype
                )
                if val_cases is not None and val_cache is not None
                else {}
            )
            metric = float(
                validation.get("teacher_stress_relative_rmse", float("inf"))
            )
            if metric < best_metric:
                best_metric = metric
                best_validation = dict(validation)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            record: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
            }
            if epoch_calibration is not None:
                record["calibration"] = epoch_calibration
            history["epochs"].append(record)
    finally:
        model.log_stress_scale.requires_grad_(original_scale_requires_grad)
    model.load_state_dict(best_state)
    history["selected_teacher_stress_relative_rmse"] = best_metric
    history["selected_teacher_stress_source"] = best_validation.get(
        "teacher_stress_source", "unavailable"
    )
    history["selected_teacher_stress_label_coverage"] = best_validation.get(
        "teacher_stress_label_coverage", 0.0
    )
    history["selected_teacher_stress_admissible_coverage"] = best_validation.get(
        "teacher_stress_admissible_coverage", 0.0
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
    residual_enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the force-driven update on an exact, noise-free state."""

    static = trajectory.static
    exact_position = static.reference_position + trajectory.displacement[next_step]
    dt = trajectory.times[next_step] - trajectory.times[next_step - 1]
    target_velocity, target_acceleration = integration_consistent_targets(
        input_state,
        exact_position,
        dt,
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
        gate_mse_weight=float(
            get_cfg(cfg, "dynamics_pretraining.stress_gate_mse_weight", 0.0)
        ),
    )
    projection_free = (
        (output.energy_diagnostics["integration_update_scale"] >= 1.0 - 1.0e-7)
        & (output.energy_diagnostics["integration_valid"] >= 0.5)
    ).to(acceleration_loss.dtype)
    del contact_pairs  # The cosine objective applies to the total acceleration.
    target_norm = torch.linalg.vector_norm(target_acceleration_moving, dim=1)
    direction_mask = target_norm > (
        float(get_cfg(cfg, "dynamics_pretraining.active_threshold", 0.05))
        * global_scale
    )
    if bool(direction_mask.any().item()):
        direction_loss = (
            1.0
            - F.cosine_similarity(
                output.acceleration[moving][direction_mask].float(),
                target_acceleration_moving[direction_mask].float(),
                dim=1,
                eps=1.0e-8,
            )
        ).mean()
    else:
        direction_loss = acceleration_loss.new_zeros(())
    quiet_mask = target_norm <= (
        float(get_cfg(cfg, "dynamics_pretraining.active_threshold", 0.05))
        * global_scale
    )
    effective_mass = (
        static.lumped_mass.reshape(-1, 1)
        * output.energy_diagnostics["mass_scale"]
    )
    residual_acceleration = output.residual_force / effective_mass
    residual_moving = residual_acceleration[moving]
    quiet_loss = acceleration_loss.new_zeros(())
    if residual_enabled and bool(quiet_mask.any().item()):
        quiet_loss = (residual_moving[quiet_mask] / global_scale).square().mean()
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
    residual_loss = acceleration_loss.new_zeros(())
    if residual_enabled:
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
        "stress_gate_mse": stress_metrics["stress_gate_mse"],
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
    residual_enabled: bool = True,
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
                    residual_enabled=residual_enabled,
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
                state,
                exact_next,
                dt,
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
) -> dict[str, list[torch.nn.Parameter]]:
    """Return a complete, mutually exclusive physical pretraining partition."""

    physical_graph: list[torch.nn.Parameter] = []
    for module in (
        model.node_encoder,
        model.vector_encoder,
        model.cell_encoder,
        model.cell_blocks,
        model.processor,
    ):
        physical_graph.extend(module.parameters())
    groups = {
        "inertia": [model.log_mass_scale],
        "force_scales": [model.log_contact_scale, model.log_damping_scale],
        "pair_heads": list(model.force_heads.parameters()),
        "physical_graph": physical_graph,
        "constitutive": _constitutive_parameters(model),
        "residual": list(model.residual_channel.parameters()),
    }
    assigned: dict[int, str] = {}
    for name, parameters in groups.items():
        for parameter in parameters:
            previous = assigned.setdefault(id(parameter), name)
            if previous != name:
                raise ValueError(
                    f"dynamics parameter appears in both {previous!r} and {name!r}"
                )
    model_ids = {id(parameter) for parameter in model.parameters()}
    missing_ids = model_ids.difference(assigned)
    extra_ids = set(assigned).difference(model_ids)
    if missing_ids or extra_ids:
        missing_names = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in missing_ids
        ]
        raise ValueError(
            "dynamics parameter partition must cover the model exactly; "
            f"missing={missing_names}, extra_count={len(extra_ids)}"
        )
    return groups


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
    phase: str = "joint",
    residual_enabled: bool = True,
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
                detach_pair_force_features=False,
                residual_enabled=residual_enabled,
            )
            loss, metrics = exact_dynamics_loss(
                output,
                state,
                trajectory,
                1,
                normalizers,
                cfg,
                contact_pairs=pairs,
                residual_enabled=residual_enabled,
            )
            exact_stress_weight = float(
                get_cfg(cfg, "dynamics_pretraining.exact_stress_weight", 0.1)
            )
            if phase in {"physics_only", "joint"} and exact_stress_weight > 0.0:
                exact_position = (
                    trajectory.static.reference_position
                    + trajectory.displacement[1]
                )
                exact_fields = model.constitutive_fields(
                    trajectory.static, exact_position
                )
                exact_nodal_stress = _nodal_stress_from_cell_tensor(
                    exact_fields.cauchy_stress, trajectory.static
                )
                stress_mask = ~trajectory.static.prescribed_mask
                if not bool(stress_mask.any().item()):
                    stress_mask = torch.ones_like(
                        trajectory.static.prescribed_mask
                    )
                exact_stress_loss, exact_stress_metrics = (
                    _supervised_constitutive_loss(
                        exact_nodal_stress,
                        exact_fields.cauchy_stress,
                        trajectory,
                        1,
                        normalizers,
                        cfg,
                        nodal_mask=stress_mask,
                        gate_mse_weight=1.0,
                    )
                )
                loss = loss + exact_stress_weight * exact_stress_loss
                metrics["exact_geometry_stress"] = exact_stress_loss.detach()
                metrics["exact_geometry_stress_gate_mse"] = (
                    exact_stress_metrics["stress_gate_mse"]
                )
        if not bool(torch.isfinite(loss).item()):
            nonfinite_losses += 1
            raise FloatingPointError(
                "non-finite dynamics-pretraining loss for "
                f"{case.case_id} at frame {start} during phase {phase!r}"
            )
        loss.backward()
        try:
            stable_clip_grad_norm_(
                parameters,
                float(get_cfg(cfg, "dynamics_pretraining.grad_clip_norm", 1.0)),
            )
        except FloatingPointError as exc:
            raise FloatingPointError(
                "non-finite dynamics-pretraining gradient for "
                f"{case.case_id} at frame {start}: {exc}"
            ) from exc
        optimizer.step()
        _accumulate(totals, metrics)
        used += 1
    averaged = {key: value / max(used, 1) for key, value in totals.items()}
    averaged["steps"] = float(used)
    averaged["nonfinite_losses"] = float(nonfinite_losses)
    averaged["nonfinite_gradients"] = float(nonfinite_gradients)
    if used == 0:
        raise RuntimeError(
            f"dynamics-pretraining phase {phase!r} produced no admissible steps"
        )
    return averaged


def _dynamics_pretraining_schedule(
    cfg: dict[str, Any],
) -> list[dict[str, int | str]]:
    configured = get_cfg(cfg, "dynamics_pretraining.phases", None)
    if configured is None:
        configured = [
            {
                "name": "physics_only",
                "epochs": int(
                    get_cfg(cfg, "dynamics_pretraining.physics_only_epochs", 4)
                ),
            },
            {
                "name": "residual_warmup",
                "epochs": int(
                    get_cfg(cfg, "dynamics_pretraining.residual_warmup_epochs", 1)
                ),
            },
            {
                "name": "joint",
                "epochs": int(
                    get_cfg(cfg, "dynamics_pretraining.joint_epochs", 2)
                ),
            },
        ]
    schedule = [
        {"name": str(item["name"]), "epochs": int(item["epochs"])}
        for item in configured
    ]
    names = [str(item["name"]) for item in schedule]
    if names != ["physics_only", "residual_warmup", "joint"]:
        raise ValueError(
            "dynamics pretraining phases must be ordered as "
            "physics_only, residual_warmup, joint"
        )
    if any(int(item["epochs"]) < 0 for item in schedule):
        raise ValueError("dynamics pretraining phase epochs cannot be negative")
    if int(schedule[0]["epochs"]) < 1:
        raise ValueError("physics_only requires at least one epoch before its gate")
    return schedule


def _set_dynamics_pretraining_phase(
    model: CHPGNS,
    optimizer: torch.optim.Optimizer,
    phase: str,
    cfg: dict[str, Any],
) -> dict[str, float]:
    active = {
        "physics_only": {
            "inertia",
            "force_scales",
            "pair_heads",
            "physical_graph",
            "constitutive",
        },
        "residual_warmup": {"residual"},
        "joint": {
            "inertia",
            "force_scales",
            "pair_heads",
            "physical_graph",
            "constitutive",
            "residual",
        },
    }
    if phase not in active:
        raise ValueError(f"unknown dynamics pretraining phase: {phase!r}")
    base_lr = float(get_cfg(cfg, "dynamics_pretraining.lr", 1.0e-4))
    configured_lrs = {
        "inertia": float(
            get_cfg(cfg, "dynamics_pretraining.inertia_lr", 1.0e-3)
        ),
        "force_scales": float(
            get_cfg(cfg, "dynamics_pretraining.force_scale_lr", 5.0e-4)
        ),
        "pair_heads": float(
            get_cfg(cfg, "dynamics_pretraining.pair_head_lr", 5.0e-4)
        ),
        "physical_graph": float(
            get_cfg(cfg, "dynamics_pretraining.physical_graph_lr", base_lr)
        ),
        "constitutive": float(
            get_cfg(cfg, "dynamics_pretraining.constitutive_lr", 1.0e-5)
        ),
        "residual": float(
            get_cfg(
                cfg,
                "dynamics_pretraining.residual_warmup_lr"
                if phase == "residual_warmup"
                else "dynamics_pretraining.residual_joint_lr",
                1.0e-3 if phase == "residual_warmup" else 5.0e-4,
            )
        ),
    }
    applied: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group["name"])
        enabled = name in active[phase]
        lr = configured_lrs[name] if enabled else 0.0
        group["lr"] = lr
        applied[name] = lr
        for parameter in group["params"]:
            parameter.requires_grad_(enabled)
    model.dynamics_pretraining_phase = phase
    return applied


def _evaluate_dynamics_physics_gate(
    dynamics_metrics: dict[str, Any],
    stress_metrics: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    thresholds = {
        "active_acceleration_relative_rmse": float(
            get_cfg(
                cfg,
                "dynamics_pretraining.phase_gate.active_acceleration_relative_rmse",
                0.95,
            )
        ),
        "active_acceleration_cosine": float(
            get_cfg(
                cfg,
                "dynamics_pretraining.phase_gate.active_acceleration_cosine",
                0.05,
            )
        ),
        "teacher_stress_relative_rmse": float(
            get_cfg(
                cfg,
                "dynamics_pretraining.phase_gate.teacher_stress_relative_rmse",
                0.50,
            )
        ),
        "teacher_stress_admissible_coverage": float(
            get_cfg(
                cfg,
                "validation.teacher_stress_minimum_admissible_coverage",
                0.0,
            )
        ),
    }
    metrics = {
        "active_acceleration_relative_rmse": float(
            dynamics_metrics.get("active_acceleration_relative_rmse", float("inf"))
        ),
        "active_acceleration_cosine": float(
            dynamics_metrics.get("active_acceleration_cosine", float("-inf"))
        ),
        "teacher_stress_relative_rmse": float(
            stress_metrics.get("teacher_stress_relative_rmse", float("inf"))
        ),
        "teacher_stress_admissible_coverage": float(
            stress_metrics.get("teacher_stress_admissible_coverage", 0.0)
        ),
        "teacher_stress_source": str(
            stress_metrics.get("teacher_stress_source", "unavailable")
        ),
    }
    failures: list[str] = []
    if not (
        math.isfinite(metrics["active_acceleration_relative_rmse"])
        and metrics["active_acceleration_relative_rmse"]
        < thresholds["active_acceleration_relative_rmse"]
    ):
        failures.append("active_acceleration_relative_rmse")
    if not (
        math.isfinite(metrics["active_acceleration_cosine"])
        and metrics["active_acceleration_cosine"]
        > thresholds["active_acceleration_cosine"]
    ):
        failures.append("active_acceleration_cosine")
    if not (
        math.isfinite(metrics["teacher_stress_relative_rmse"])
        and metrics["teacher_stress_relative_rmse"]
        < thresholds["teacher_stress_relative_rmse"]
    ):
        failures.append("teacher_stress_relative_rmse")
    if not (
        math.isfinite(metrics["teacher_stress_admissible_coverage"])
        and metrics["teacher_stress_admissible_coverage"]
        >= thresholds["teacher_stress_admissible_coverage"]
    ):
        failures.append("teacher_stress_admissible_coverage")
    return {
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
    }


_TRANSITION_DYNAMICS_GATE_ROLE = "transition_pre_residual"
_FINAL_DYNAMICS_GATE_ROLE = "final_post_joint_residual_disabled"


def _dynamics_gate_with_role(
    dynamics_metrics: dict[str, Any],
    stress_metrics: dict[str, Any],
    cfg: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Evaluate the shared physics thresholds and identify when they ran."""

    if role not in {
        _TRANSITION_DYNAMICS_GATE_ROLE,
        _FINAL_DYNAMICS_GATE_ROLE,
    }:
        raise ValueError(f"unsupported dynamics gate role: {role!r}")
    return {
        **_evaluate_dynamics_physics_gate(dynamics_metrics, stress_metrics, cfg),
        "gate_role": role,
        "residual_enabled": False,
    }


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
    """Fit identifiable physical forces before unlocking the residual channel."""

    if not bool(get_cfg(cfg, "dynamics_pretraining.enabled", True)):
        return {"enabled": False}
    if val_cases is None or val_cache is None:
        raise ValueError("dynamics pretraining requires a validation split")
    if int(model.dynamics_semantics_version) != CHPGNS.dynamics_schema_version:
        raise ValueError(
            "formal dynamics pretraining requires the schema-9 integration contract"
        )
    schedule = _dynamics_pretraining_schedule(cfg)
    flattened_schedule = [
        (str(item["name"]), phase_epoch)
        for item in schedule
        for phase_epoch in range(1, int(item["epochs"]) + 1)
    ]
    parameter_groups = _dynamics_pretraining_parameters(model)
    original_requires_grad = {
        id(parameter): parameter.requires_grad for parameter in model.parameters()
    }
    weight_decay = float(
        get_cfg(cfg, "dynamics_pretraining.weight_decay", 1.0e-6)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "name": name,
                "params": parameters,
                "lr": 0.0,
                "weight_decay": (
                    weight_decay
                    if name in {"physical_graph", "pair_heads", "residual"}
                    else 0.0
                ),
            }
            for name, parameters in parameter_groups.items()
        ],
        lr=1.0,
        weight_decay=0.0,
    )
    state_path = output_dir / "dynamics_pretraining_latest.pt"
    complete_path = output_dir / "dynamics_pretraining_complete.pt"
    history: dict[str, Any] = {
        "enabled": True,
        "protocol": "physics_transition_final_gate_residual_v2",
        "schedule": schedule,
        "epochs": [],
    }
    completed_epochs = 0
    transition_gate: dict[str, Any] = {
        "status": "pending",
        "passed": False,
        "gate_role": _TRANSITION_DYNAMICS_GATE_ROLE,
        "residual_enabled": False,
    }
    final_gate: dict[str, Any] = {
        "status": "pending",
        "passed": False,
        "gate_role": _FINAL_DYNAMICS_GATE_ROLE,
        "residual_enabled": False,
    }

    def save_state(*, status: str, phase: str, phase_epoch: int) -> None:
        phase_state = {
            "phase": phase,
            "status": status,
            "global_epoch_completed": completed_epochs,
            "phase_epoch_completed": phase_epoch,
            "schedule": schedule,
            # ``physics_gate`` remains as a migration alias for the
            # uncommitted schema-9 pretraining artifact.  It means the final
            # gate only once the artifact is complete; both named gates are
            # always persisted by protocol v2.
            "physics_gate": (
                final_gate if status == "complete" else transition_gate
            ),
            "transition_physics_gate": transition_gate,
            "final_physics_gate": final_gate,
            "residual_zero_initialized": True,
            "residual_enabled": phase in {
                "residual_warmup",
                "joint",
                "complete",
            },
        }
        model.dynamics_pretraining_phase = phase
        model.dynamics_pretraining_phase_gate = (
            final_gate if status == "complete" else transition_gate
        )
        model.dynamics_pretraining_transition_gate = transition_gate
        model.dynamics_pretraining_final_gate = final_gate
        payload = {
            "artifact_type": "chp_dynamics_pretraining_checkpoint",
            "artifact_schema_version": 1,
            "checkpoint_schema_version": CHPGNS.checkpoint_schema_version,
            "dynamics_schema_version": int(model.dynamics_semantics_version),
            "dynamics_pretraining_protocol": (
                "physics_transition_final_gate_residual_v2"
            ),
            "phase_state": phase_state,
            "parameter_group_names": list(parameter_groups),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizers": normalizers.state_dict(),
            "history": history,
            "rng_state": _capture_rng_state(),
            "config": cfg,
        }
        _atomic_torch_save(payload, state_path)

    try:
        if state_path.is_file():
            saved = _torch_load(state_path, train_cache.device)
            if saved.get("artifact_type") != "chp_dynamics_pretraining_checkpoint":
                raise ValueError(f"invalid dynamics pretraining artifact: {state_path}")
            if int(saved.get("dynamics_schema_version", 0)) != int(
                model.dynamics_semantics_version
            ):
                raise ValueError("dynamics pretraining checkpoint semantics changed")
            saved_phase_state = saved.get("phase_state", {})
            if saved_phase_state.get("schedule") != schedule:
                raise ValueError("dynamics pretraining schedule changed across resume")
            if saved.get("parameter_group_names") != list(parameter_groups):
                raise ValueError("dynamics pretraining parameter groups changed")
            if not _checkpoint_values_equal(saved.get("config"), cfg):
                raise ValueError("dynamics pretraining config changed across resume")
            if not _checkpoint_values_equal(
                saved.get("normalizers"), normalizers.state_dict()
            ):
                raise ValueError(
                    "dynamics pretraining normalizers changed across resume"
                )
            if saved_phase_state.get("status") == "failed":
                raise RuntimeError(
                    "dynamics physics gate previously failed; refusing resume"
                )
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            history = saved["history"]
            completed_epochs = int(
                saved_phase_state.get("global_epoch_completed", 0)
            )
            transition_gate = dict(
                saved_phase_state.get(
                    "transition_physics_gate",
                    saved_phase_state.get("physics_gate", transition_gate),
                )
            )
            final_gate = dict(
                saved_phase_state.get("final_physics_gate", final_gate)
            )
            if saved.get("rng_state") is not None:
                _restore_rng_state(saved["rng_state"])
            if (
                saved_phase_state.get("status") == "complete"
                and final_gate.get("status") == "passed"
                and final_gate.get("gate_role") == _FINAL_DYNAMICS_GATE_ROLE
            ):
                model.dynamics_pretraining_phase = "complete"
                model.dynamics_pretraining_phase_gate = final_gate
                model.dynamics_pretraining_transition_gate = transition_gate
                model.dynamics_pretraining_final_gate = final_gate
                return history
        else:
            with torch.no_grad():
                torch.nn.init.zeros_(model.residual_channel.weight)
            initial = evaluate_teacher_forced_dynamics(
                model,
                val_cases,
                val_cache,
                cfg,
                normalizers,
                amp_dtype=amp_dtype,
                residual_enabled=False,
            )
            history["initial_validation"] = initial

        physics_epochs = int(schedule[0]["epochs"])
        for global_index, (phase, phase_epoch) in enumerate(
            flattened_schedule, start=1
        ):
            if global_index <= completed_epochs:
                continue
            residual_enabled = phase != "physics_only"
            group_lrs = _set_dynamics_pretraining_phase(
                model, optimizer, phase, cfg
            )
            train_metrics = train_exact_dynamics_epoch(
                model,
                train_cases,
                train_cache,
                sampling_scores,
                normalizers,
                optimizer,
                cfg,
                epoch=global_index,
                amp_dtype=amp_dtype,
                phase=phase,
                residual_enabled=residual_enabled,
            )
            validation = evaluate_teacher_forced_dynamics(
                model,
                val_cases,
                val_cache,
                cfg,
                normalizers,
                amp_dtype=amp_dtype,
                residual_enabled=residual_enabled,
            )
            score = max(
                validation["active_acceleration_relative_rmse"],
                validation["one_step_stress_relative_rmse"],
            )
            record = {
                "epoch": global_index,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "residual_enabled": residual_enabled,
                "train": train_metrics,
                "validation": validation,
                "score": score,
                "group_lrs": group_lrs,
            }
            history["epochs"] = [
                item
                for item in history["epochs"]
                if int(item["epoch"]) != global_index
            ]
            history["epochs"].append(record)
            completed_epochs = global_index

            if phase == "physics_only" and phase_epoch == physics_epochs:
                stress_validation = evaluate_teacher_forced_stress(
                    model,
                    val_cases,
                    val_cache,
                    cfg,
                    amp_dtype=amp_dtype,
                )
                transition_gate = _dynamics_gate_with_role(
                    validation,
                    stress_validation,
                    cfg,
                    role=_TRANSITION_DYNAMICS_GATE_ROLE,
                )
                history["transition_physics_gate"] = transition_gate
                # Keep this alias in JSON for readers of the initial schema-9
                # implementation; it is never accepted as the final gate.
                history["physics_gate"] = transition_gate
                model.dynamics_pretraining_phase_gate = transition_gate
                model.dynamics_pretraining_transition_gate = transition_gate
                if not bool(transition_gate["passed"]):
                    save_state(
                        status="failed",
                        phase="physics_gate",
                        phase_epoch=phase_epoch,
                    )
                    failure = {
                        "stage": "dynamics_pretraining.physics_gate",
                        "dynamics_schema_version": int(
                            model.dynamics_semantics_version
                        ),
                        "protocol": "physics_transition_final_gate_residual_v2",
                        "completed_physics_epochs": physics_epochs,
                        **transition_gate,
                        "test_content_accessed": False,
                        "action": (
                            "stop before residual learning and rollout; revise "
                            "physical forces, inertia, or constitutive fit"
                        ),
                    }
                    _save_json(
                        output_dir / "dynamics_physics_gate_failure.json",
                        failure,
                    )
                    _save_json(output_dir / "dynamics_pretraining.json", history)
                    raise RuntimeError(
                        "dynamics physics-only gate failed: "
                        + ", ".join(transition_gate["failures"])
                    )
            save_state(
                status="in_progress",
                phase=phase,
                phase_epoch=phase_epoch,
            )
            _save_json(output_dir / "dynamics_pretraining.json", history)

        # The residual channel is deliberately disabled after joint training.
        # This second, identically-thresholded gate proves that the physical
        # force path still carries the dynamics instead of being bypassed by
        # the bounded residual decoder.
        final_dynamics_validation = evaluate_teacher_forced_dynamics(
            model,
            val_cases,
            val_cache,
            cfg,
            normalizers,
            amp_dtype=amp_dtype,
            residual_enabled=False,
        )
        final_stress_validation = evaluate_teacher_forced_stress(
            model,
            val_cases,
            val_cache,
            cfg,
            amp_dtype=amp_dtype,
        )
        final_gate = _dynamics_gate_with_role(
            final_dynamics_validation,
            final_stress_validation,
            cfg,
            role=_FINAL_DYNAMICS_GATE_ROLE,
        )
        history["protocol"] = "physics_transition_final_gate_residual_v2"
        history["final_physics_validation"] = {
            "dynamics": final_dynamics_validation,
            "stress": final_stress_validation,
        }
        history["final_physics_gate"] = final_gate
        if not bool(final_gate["passed"]):
            save_state(
                status="failed",
                phase="final_physics_gate",
                phase_epoch=int(schedule[-1]["epochs"]),
            )
            failure = {
                "stage": "dynamics_pretraining.final_physics_gate",
                "dynamics_schema_version": int(model.dynamics_semantics_version),
                "protocol": "physics_transition_final_gate_residual_v2",
                **final_gate,
                "test_content_accessed": False,
                "action": (
                    "stop before rollout; revise joint training because the "
                    "residual-disabled physical force path regressed"
                ),
            }
            _save_json(
                output_dir / "dynamics_final_physics_gate_failure.json",
                failure,
            )
            _save_json(output_dir / "dynamics_pretraining.json", history)
            raise RuntimeError(
                "final residual-disabled dynamics physics gate failed: "
                + ", ".join(final_gate["failures"])
            )
        history["selected_score"] = max(
            float(
                final_dynamics_validation.get(
                    "active_acceleration_relative_rmse", float("inf")
                )
            ),
            float(
                final_dynamics_validation.get(
                    "one_step_stress_relative_rmse", float("inf")
                )
            ),
        )
        history["selected_phase"] = "complete"
        model.dynamics_pretraining_phase = "complete"
        model.dynamics_pretraining_phase_gate = final_gate
        model.dynamics_pretraining_transition_gate = transition_gate
        model.dynamics_pretraining_final_gate = final_gate
        final_phase_epoch = int(schedule[-1]["epochs"])
        save_state(
            status="complete", phase="complete", phase_epoch=final_phase_epoch
        )
        _atomic_torch_save(_torch_load(state_path, "cpu"), complete_path)
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
    _save_json(output_dir / "dynamics_pretraining.json", history)
    return history


def run_chp_training(cfg: dict[str, Any]) -> Path:
    """Train CHP-GNS using BF16 neural blocks and FP32 mechanics on CUDA."""

    validate_chp_problem_semantics(cfg)
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
    rng_provenance: dict[str, Any] = {
        "lineage_exact": True,
        "resume_rng_state": "fresh_seeded_run",
        "resume_source": None,
    }
    history = _load_json(output_dir / "history.json", default=[])
    resume = _resolve_resume(cfg, output_dir)
    if resume is None:
        dynamics_state_exists = (
            output_dir / "dynamics_pretraining_latest.pt"
        ).is_file()
        if dynamics_state_exists:
            # The dynamics artifact embeds the post-constitutive model and is
            # validated by run_dynamics_pretraining.  Do not repeat or mutate
            # constitutive pretraining before restoring it.
            pretraining = {"enabled": False, "resumed_from_dynamics": True}
        else:
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
            selected_admissible_coverage = float(
                pretraining.get(
                    "selected_teacher_stress_admissible_coverage", 0.0
                )
            )
            print(
                "constitutive pretraining selected teacher "
                f"rRMSE={selected:.4g} source={selected_source} "
                f"label_coverage={selected_coverage:.3f} "
                f"admissible_coverage={selected_admissible_coverage:.3f}"
            )
            teacher_threshold = float(
                get_cfg(cfg, "validation.teacher_stress_threshold", 0.50)
            )
            enforce_teacher_gate = bool(
                get_cfg(
                    cfg,
                    "validation.enforce_teacher_stress_gate",
                    True,
                )
            )
            minimum_admissible_coverage = float(
                get_cfg(
                    cfg,
                    "validation.teacher_stress_minimum_admissible_coverage",
                    0.0,
                )
            )
            _save_constitutive_gate_artifact(
                output_dir / "constitutive_gate.pt",
                model,
                normalizers,
                cfg,
                material_dim=material_dim,
                pretraining=pretraining,
                threshold=teacher_threshold,
                enforced=enforce_teacher_gate,
            )
            if enforce_teacher_gate and (
                selected >= teacher_threshold
                or selected_admissible_coverage < minimum_admissible_coverage
            ):
                failure = {
                    "stage": "constitutive_pretraining",
                    "teacher_stress_relative_rmse": selected,
                    "teacher_stress_source": selected_source,
                    "teacher_stress_label_coverage": selected_coverage,
                    "teacher_stress_admissible_coverage": (
                        selected_admissible_coverage
                    ),
                    "minimum_admissible_coverage": minimum_admissible_coverage,
                    "threshold": teacher_threshold,
                    "action": (
                        "stop before dynamics and rollout training; revise "
                        "constitutive admissibility or stress fit"
                    ),
                }
                _save_json(
                    output_dir / "teacher_stress_gate_failure.json", failure
                )
                raise RuntimeError(
                    "pretrained teacher-forced stress gate failed: "
                    f"rRMSE={selected:.4g} (threshold {teacher_threshold:.4g}), "
                    "admissible_coverage="
                    f"{selected_admissible_coverage:.4g} "
                    f"(minimum {minimum_admissible_coverage:.4g})"
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
        model.dynamics_pretraining_phase = str(
            checkpoint["dynamics_pretraining_phase"]
        )
        model.dynamics_pretraining_phase_gate = dict(
            checkpoint["dynamics_pretraining_phase_gate"]
        )
        model.dynamics_pretraining_final_gate = dict(
            checkpoint.get(
                "dynamics_pretraining_final_gate",
                checkpoint["dynamics_pretraining_phase_gate"],
            )
        )
        model.dynamics_pretraining_transition_gate = dict(
            checkpoint.get(
                "dynamics_pretraining_transition_gate",
                {"status": "unavailable"},
            )
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        saved_rng_state = checkpoint.get("rng_state")
        if saved_rng_state is None:
            rng_provenance = {
                "lineage_exact": False,
                "resume_rng_state": "legacy_checkpoint_missing",
                "resume_source": str(resume),
            }
            print(
                "warning: resume checkpoint has no RNG state; continuation "
                "is valid but not bitwise reproducible"
            )
        else:
            _restore_rng_state(saved_rng_state)
            prior_provenance = checkpoint.get("rng_provenance", {})
            rng_provenance = {
                "lineage_exact": bool(
                    prior_provenance.get("lineage_exact", True)
                ),
                "resume_rng_state": "restored",
                "resume_source": str(resume),
            }
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
            rng_provenance=rng_provenance,
        )
        _atomic_torch_save(payload, output_dir / "latest.pt")
        if (
            gate_status in {"passed", "not_required"}
            and score is not None
            and score < best_score
        ):
            best_score = score
            payload["best_score"] = best_score
            _atomic_torch_save(payload, output_dir / "best.pt")
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
            maximum_i1_bar=float(
                get_cfg(
                    cfg,
                    "training.noise_maximum_i1_bar",
                    get_cfg(cfg, "training.maximum_start_i1_bar", float("inf")),
                )
            ),
            maximum_i2_bar=float(
                get_cfg(
                    cfg,
                    "training.noise_maximum_i2_bar",
                    get_cfg(cfg, "training.maximum_start_i2_bar", 1.0e5),
                )
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
                    exact_fields = model.constitutive_fields(
                        trajectory.static, exact_position
                    )
                    exact_cell_stress = exact_fields.cauchy_stress
                    exact_nodal_stress = _nodal_stress_from_cell_tensor(
                        exact_cell_stress, trajectory.static
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
                        gate_mse_weight=float(
                            get_cfg(cfg, "loss.stress_gate_mse_weight", 0.0)
                        ),
                    )
                    step_loss = step_loss + float(
                        get_cfg(cfg, "loss.exact_constitutive", 0.5)
                    ) * exact_constitutive
                    reaction_weight = float(
                        get_cfg(cfg, "loss.reaction_equilibrium", 0.0)
                    )
                    if reaction_weight > 0.0 and not trajectory.has_solver_nodal_force:
                        raise ValueError(
                            "reaction-equilibrium supervision requires "
                            "solver_nodal_force.npy"
                        )
                    if trajectory.has_solver_nodal_force:
                        reaction_loss, reaction_relative = reaction_equilibrium_loss(
                            exact_fields.internal_force,
                            trajectory.solver_nodal_force[step + 1],
                            trajectory.static.fixed_mask,
                            huber_delta=float(
                                get_cfg(cfg, "loss.huber_delta", 1.0)
                            ),
                        )
                    else:
                        reaction_loss = step_loss.new_zeros(())
                        reaction_relative = step_loss.new_zeros(())
                    step_loss = step_loss + reaction_weight * reaction_loss
                    metrics["exact_constitutive"] = exact_metrics["loss"]
                    metrics["exact_stress_physical"] = exact_metrics[
                        "stress_physical"
                    ]
                    metrics["exact_stress_gate_mse"] = exact_metrics[
                        "stress_gate_mse"
                    ]
                    metrics["exact_stress_tensor_supervision"] = exact_metrics[
                        "stress_tensor_supervision"
                    ]
                    metrics["reaction_equilibrium"] = reaction_loss.detach()
                    metrics["reaction_equilibrium_relative_rmse"] = (
                        reaction_relative.detach()
                    )
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
            try:
                stable_clip_grad_norm_(
                    model.parameters(), float(clip)
                )
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f"non-finite CHP gradient values for {case.case_id} "
                    f"at frame {start}: {exc}"
                ) from exc
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


def _conservative_force_resultants(
    output: PhysicalStep,
) -> dict[str, torch.Tensor]:
    """Return FP64 net resultants of the conservative force channels.

    These are conservation diagnostics, not the discrete equation-solver
    residual ``M a - F``.  Internal and paired contact forces should each
    have a near-zero global resultant before boundary projection; their sum
    reports the complete conservative channel independently of damping,
    external loading, and the learned residual force.
    """

    internal = output.effective_internal_force.double()
    contact = output.effective_contact_force.double()
    if internal.ndim != 2 or internal.shape[-1] != 3:
        raise ValueError("internal force must have shape [N, 3]")
    if contact.shape != internal.shape:
        raise ValueError("contact and internal force must have matching shapes")
    return {
        "internal_force_resultant": torch.linalg.vector_norm(
            internal.sum(dim=0)
        ),
        "contact_force_resultant": torch.linalg.vector_norm(
            contact.sum(dim=0)
        ),
        "total_conservative_force_resultant": torch.linalg.vector_norm(
            (internal + contact).sum(dim=0)
        ),
    }


@torch.no_grad()
def evaluate_chp_rollouts(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    cfg: dict[str, Any],
    *,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
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
            "solver_force_balance_rms",
            "internal_force_resultant",
            "contact_force_resultant",
            "total_conservative_force_resultant",
            "energy",
            "backtrack",
            "integration_failure",
            "proposal_domain",
        )
    }
    evaluated_steps = 0
    attempted_steps = 0
    tensor_evaluated_steps = 0
    tensor_label_cases = 0
    nodal_label_cases = 0
    diverged = 0
    divergence_limit = float(get_cfg(cfg, "validation.divergence_position", 10.0))
    for case_index_value in tqdm(case_indices, desc="rollout-val", leave=False):
        case_index = int(case_index_value)
        case = cases[case_index]
        trajectory = cache.get(case_index)
        if _trajectory_has_cell_stress_tensor(trajectory):
            tensor_label_cases += 1
        else:
            nodal_label_cases += 1
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
        stress_mask = ~trajectory.static.prescribed_mask
        if not bool(stress_mask.any().item()):
            stress_mask = torch.ones_like(trajectory.static.prescribed_mask)
        rollout_stress_target = trajectory.stress[1 : steps + 1, stress_mask, :1]
        nodal_p95_threshold = torch.quantile(
            rollout_stress_target.abs().reshape(-1), 0.95
        )
        cell_vm_p95_threshold = None
        if _trajectory_has_cell_stress_tensor(trajectory):
            rollout_cell_vm_target = _tensor6_von_mises(
                trajectory.cell_stress[1 : steps + 1]
            )
            cell_vm_p95_threshold = torch.quantile(
                rollout_cell_vm_target.abs().reshape(-1), 0.95
            )
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
            predicted_gradient = deformation_gradient(
                state.position,
                trajectory.static.cells,
                trajectory.static.dm_inv,
            )
            predicted_deformation = invariants(predicted_gradient)
            predicted_domain_valid = (
                torch.isfinite(predicted_gradient).all()
                & torch.isfinite(predicted_deformation.c).all()
                & torch.isfinite(predicted_deformation.i1).all()
                & torch.isfinite(predicted_deformation.i2).all()
                & torch.isfinite(predicted_deformation.j).all()
                & torch.isfinite(predicted_deformation.i1_bar).all()
                & torch.isfinite(predicted_deformation.i2_bar).all()
                & (
                    predicted_deformation.j.min()
                    >= float(get_cfg(cfg, "validation.minimum_predicted_j", 1.0e-4))
                )
                & (
                    predicted_deformation.i1_bar.max()
                    <= float(
                        get_cfg(
                            cfg,
                            "validation.maximum_predicted_i1_bar",
                            1.0e4,
                        )
                    )
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
            target_stress = trajectory.stress[step + 1, stress_mask, :1]
            stress_error = output.nodal_stress[stress_mask, :1] - target_stress
            accum["stress_error"] += stress_error.double().square().sum()
            accum["stress_reference"] += target_stress.double().square().sum()
            peak = target_stress.abs() >= nodal_p95_threshold
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
                assert cell_vm_p95_threshold is not None
                cell_peak = target_vm.abs() >= cell_vm_p95_threshold
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
            accum["solver_force_balance_rms"] += output.energy_diagnostics[
                "discrete_force_balance_rms"
            ].double()
            force_resultants = _conservative_force_resultants(output)
            for key, value in force_resultants.items():
                accum[key] += value
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
    nodal_stress_relative = float(
        torch.sqrt(
            accum["stress_error"] / accum["stress_reference"].clamp_min(eps)
        ).item()
    )
    nodal_stress_p95_relative = float(
        torch.sqrt(
            accum["p95_error"] / accum["p95_reference"].clamp_min(eps)
        ).item()
    )
    cell_tensor_relative: float | None = (
        float(
            torch.sqrt(
                accum["cell_tensor_error"]
                / accum["cell_tensor_reference"].clamp_min(eps)
            ).item()
        )
        if tensor_evaluated_steps
        else None
    )
    cell_tensor_p95_relative: float | None = (
        float(
            torch.sqrt(
                accum["cell_tensor_p95_error"]
                / accum["cell_tensor_p95_reference"].clamp_min(eps)
            ).item()
        )
        if tensor_evaluated_steps
        else None
    )
    cell_vm_relative: float | None = (
        float(
            torch.sqrt(
                accum["cell_vm_error"]
                / accum["cell_vm_reference"].clamp_min(eps)
            ).item()
        )
        if tensor_evaluated_steps
        else None
    )
    cell_vm_p95_relative: float | None = (
        float(
            torch.sqrt(
                accum["cell_vm_p95_error"]
                / accum["cell_vm_p95_reference"].clamp_min(eps)
            ).item()
        )
        if tensor_evaluated_steps
        else None
    )
    if tensor_evaluated_steps:
        cell_tensor_relative = (
            cell_tensor_relative
            if cell_tensor_relative is not None and math.isfinite(cell_tensor_relative)
            else FAILED_RELATIVE_METRIC
        )
        cell_tensor_p95_relative = (
            cell_tensor_p95_relative
            if cell_tensor_p95_relative is not None
            and math.isfinite(cell_tensor_p95_relative)
            else FAILED_RELATIVE_METRIC
        )
        cell_vm_relative = (
            cell_vm_relative
            if cell_vm_relative is not None and math.isfinite(cell_vm_relative)
            else FAILED_RELATIVE_METRIC
        )
        cell_vm_p95_relative = (
            cell_vm_p95_relative
            if cell_vm_p95_relative is not None
            and math.isfinite(cell_vm_p95_relative)
            else FAILED_RELATIVE_METRIC
        )
    if tensor_label_cases and nodal_label_cases:
        raise ValueError(
            "validation subset mixes full cell-tensor and nodal-only stress labels"
        )
    if tensor_label_cases:
        stress_source = CELL_TENSOR_STRESS_SOURCE
        primary_stress_relative = (
            cell_tensor_relative
            if cell_tensor_relative is not None
            and math.isfinite(cell_tensor_relative)
            else FAILED_RELATIVE_METRIC
        )
        primary_stress_p95_relative = (
            cell_vm_p95_relative
            if cell_vm_p95_relative is not None
            and math.isfinite(cell_vm_p95_relative)
            else FAILED_RELATIVE_METRIC
        )
    else:
        stress_source = NODAL_STRESS_FALLBACK_SOURCE
        primary_stress_relative = nodal_stress_relative
        primary_stress_p95_relative = nodal_stress_p95_relative
    result = {
        "moving_displacement_relative_rmse": float(
            torch.sqrt(accum["u_error"] / accum["u_reference"].clamp_min(eps)).item()
        ),
        "final_displacement_relative_rmse": float(
            torch.sqrt(
                accum["final_error"] / accum["final_reference"].clamp_min(eps)
            ).item()
        ),
        "stress_relative_rmse": primary_stress_relative,
        "stress_p95_relative_rmse": primary_stress_p95_relative,
        "stress_metric_source": stress_source,
        "nodal_stress_relative_rmse": nodal_stress_relative,
        "nodal_stress_p95_relative_rmse": nodal_stress_p95_relative,
        "cell_stress_tensor_relative_rmse": cell_tensor_relative,
        "cell_stress_tensor_p95_relative_rmse": cell_tensor_p95_relative,
        "cell_stress_vm_relative_rmse": cell_vm_relative,
        "cell_stress_vm_p95_relative_rmse": cell_vm_p95_relative,
        "cell_stress_tensor_coverage": float(
            tensor_evaluated_steps / max(evaluated_steps, 1)
        ),
        "cell_stress_tensor_case_coverage": float(
            tensor_label_cases / max(tensor_label_cases + nodal_label_cases, 1)
        ),
        "diverged_cases": float(diverged),
        "mean_penetration": float((accum["penetration"] / max(evaluated_steps, 1)).item()),
        "mean_solver_force_balance_rms": float(
            (
                accum["solver_force_balance_rms"]
                / max(evaluated_steps, 1)
            ).item()
        ),
        "mean_discrete_force_balance_rms": float(
            (
                accum["solver_force_balance_rms"]
                / max(evaluated_steps, 1)
            ).item()
        ),
        "mean_internal_force_resultant": float(
            (
                accum["internal_force_resultant"]
                / max(evaluated_steps, 1)
            ).item()
        ),
        "mean_contact_force_resultant": float(
            (
                accum["contact_force_resultant"]
                / max(evaluated_steps, 1)
            ).item()
        ),
        "mean_total_conservative_force_resultant": float(
            (
                accum["total_conservative_force_resultant"]
                / max(evaluated_steps, 1)
            ).item()
        ),
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
            "reaction_error",
            "reaction_reference",
        )
    }
    requested_frames = 0
    admissible_frames = 0
    inverted_frames = 0
    near_singular_frames = 0
    extreme_i2_frames = 0
    tensor_label_frames = 0
    nodal_label_frames = 0
    reaction_label_frames = 0
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
                if hasattr(model, "constitutive_fields"):
                    exact_fields = model.constitutive_fields(
                        trajectory.static, exact_position
                    )
                    cell_prediction = exact_fields.cauchy_stress
                    nodal_prediction = _nodal_stress_from_cell_tensor(
                        cell_prediction, trajectory.static
                    )
                else:
                    # Lightweight protocol fixtures may expose only the
                    # historical observation helper.  Production CHPGNS
                    # always takes the constitutive_fields branch above.
                    exact_fields = None
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
            if getattr(trajectory, "has_solver_nodal_force", False):
                if exact_fields is None:
                    raise ValueError(
                        "solver reaction evaluation requires constitutive_fields"
                    )
                reaction_mask = trajectory.static.fixed_mask
                reaction_target = trajectory.solver_nodal_force[frame, reaction_mask]
                reaction_reference = reaction_target.square().sum()
                if bool((reaction_reference > 1.0e-20).item()):
                    reaction_residual = (
                        exact_fields.internal_force[reaction_mask]
                        + reaction_target
                    )
                    accum["reaction_error"] += reaction_residual.square().sum()
                    accum["reaction_reference"] += reaction_reference
                    reaction_label_frames += 1
    eps = torch.tensor(1.0e-12, device=cache.device)

    def relative(
        error_key: str, reference_key: str, frames: int
    ) -> float | None:
        reference = accum[reference_key]
        if frames and bool((reference > eps).item()):
            value = float(
                torch.sqrt(accum[error_key] / reference.clamp_min(eps)).item()
            )
            return value if math.isfinite(value) else FAILED_RELATIVE_METRIC
        return None

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
    reaction_relative = relative(
        "reaction_error", "reaction_reference", reaction_label_frames
    )
    if tensor_label_frames:
        source = CELL_TENSOR_STRESS_SOURCE
        primary_relative = (
            tensor_relative
            if tensor_relative is not None
            else FAILED_RELATIVE_METRIC
        )
        primary_peak_relative = (
            cell_vm_peak_relative
            if cell_vm_peak_relative is not None
            else FAILED_RELATIVE_METRIC
        )
        label_frames = tensor_label_frames
    else:
        source = NODAL_STRESS_FALLBACK_SOURCE
        primary_relative = (
            nodal_relative if nodal_relative is not None else FAILED_RELATIVE_METRIC
        )
        primary_peak_relative = (
            nodal_peak_relative
            if nodal_peak_relative is not None
            else FAILED_RELATIVE_METRIC
        )
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
        "teacher_reaction_equilibrium_relative_rmse": reaction_relative,
        "teacher_reaction_equilibrium_frames": float(reaction_label_frames),
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
    model_cfg = checkpoint.get("config", {}).get("model", {})
    integration_contract = {
        "contact_iterations": 2,
        "integration_substeps": 1,
        "contact_predictor_stop_gradient": True,
        "contact_force_average": "trapezoidal",
    }
    contract_mismatch = {
        key: model_cfg.get(key)
        for key, value in integration_contract.items()
        if model_cfg.get(key) != value
    }
    if contract_mismatch:
        raise ValueError(
            "checkpoint integration contract is missing or incompatible: "
            f"{source}; found={contract_mismatch}, expected={integration_contract}"
        )
    dynamics_enabled = bool(
        checkpoint.get("config", {})
        .get("dynamics_pretraining", {})
        .get("enabled", True)
    )
    if dynamics_enabled and checkpoint.get("dynamics_pretraining_phase") != "complete":
        raise ValueError(
            "checkpoint dynamics pretraining phase is incomplete: "
            f"{source}; phase={checkpoint.get('dynamics_pretraining_phase')!r}"
        )
    gate = checkpoint.get("dynamics_pretraining_phase_gate", {})
    if dynamics_enabled and (
        gate.get("status") != "passed"
        or gate.get("gate_role") != _FINAL_DYNAMICS_GATE_ROLE
        or bool(gate.get("residual_enabled", True))
    ):
        raise ValueError(
            "checkpoint final residual-disabled dynamics gate did not pass: "
            f"{source}; gate_status={gate.get('status')!r}, "
            f"gate_role={gate.get('gate_role')!r}, "
            f"residual_enabled={gate.get('residual_enabled')!r}"
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


def validate_chp_problem_semantics(cfg: dict[str, Any]) -> str:
    """Separate transient dynamics from quasi-static continuation claims."""

    problem_type = str(get_cfg(cfg, "training.problem_type", "dynamic")).lower()
    if problem_type not in {"dynamic", "quasi_static_continuation"}:
        raise ValueError(
            "training.problem_type must be 'dynamic' or "
            "'quasi_static_continuation'"
        )
    if problem_type == "quasi_static_continuation":
        time_semantics = str(get_cfg(cfg, "data.time_semantics", "")).lower()
        if "quasi_static" not in time_semantics:
            raise ValueError(
                "quasi-static continuation requires explicit data.time_semantics"
            )
        if bool(get_cfg(cfg, "dynamics_pretraining.enabled", False)):
            raise ValueError(
                "quasi-static load paths cannot use transient dynamics pretraining"
            )
        if float(get_cfg(cfg, "loss.velocity", 0.0)) != 0.0:
            raise ValueError("quasi-static load paths must set loss.velocity=0")
        if float(get_cfg(cfg, "loss.work_energy", 0.0)) != 0.0:
            raise ValueError(
                "quasi-static load paths must set loss.work_energy=0; "
                "pseudo-time kinetic energy is not physical evidence"
            )
        if float(get_cfg(cfg, "loss.reaction_equilibrium", 0.0)) <= 0.0:
            raise ValueError(
                "quasi-static continuation requires positive fixed-reaction "
                "equilibrium supervision"
            )
    return problem_type


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
    rng_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(model.dynamics_semantics_version) != CHPGNS.dynamics_schema_version:
        raise ValueError(
            "schema-8 model instances cannot be serialized as schema-9 checkpoints"
        )
    dynamics_enabled = bool(get_cfg(cfg, "dynamics_pretraining.enabled", True))
    phase = str(
        getattr(
            model,
            "dynamics_pretraining_phase",
            "missing" if dynamics_enabled else "not_required",
        )
    )
    phase_gate = dict(
        getattr(
            model,
            "dynamics_pretraining_phase_gate",
            {"status": "missing" if dynamics_enabled else "not_required"},
        )
    )
    if dynamics_enabled and (
        phase != "complete"
        or phase_gate.get("status") != "passed"
        or phase_gate.get("gate_role") != _FINAL_DYNAMICS_GATE_ROLE
        or bool(phase_gate.get("residual_enabled", True))
    ):
        raise ValueError(
            "rollout checkpoint requires completed dynamics pretraining and a "
            "passed final residual-disabled physics gate"
        )
    transition_gate = dict(
        getattr(
            model,
            "dynamics_pretraining_transition_gate",
            {"status": "missing" if dynamics_enabled else "not_required"},
        )
    )
    return {
        "artifact_type": "chp_rollout_checkpoint",
        "schema_version": CHPGNS.checkpoint_schema_version,
        "dynamics_schema_version": int(model.dynamics_semantics_version),
        "residual_parameterization": CHPGNS.residual_parameterization,
        "residual_gate": CHPGNS.residual_gate,
        "dynamics_pretraining_phase": phase,
        "dynamics_pretraining_phase_gate": phase_gate,
        "dynamics_pretraining_transition_gate": transition_gate,
        "dynamics_pretraining_final_gate": phase_gate,
        "architecture": "CHP-GNS",
        "problem_type": str(get_cfg(cfg, "training.problem_type", "dynamic")),
        "time_semantics": str(get_cfg(cfg, "data.time_semantics", "dynamic")),
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
        "rng_state": _capture_rng_state(),
        "rng_provenance": dict(
            rng_provenance
            or {
                "lineage_exact": True,
                "resume_rng_state": "fresh_seeded_run",
                "resume_source": None,
            }
        ),
    }


def _save_constitutive_gate_artifact(
    path: Path,
    model: CHPGNS,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
    *,
    material_dim: int,
    pretraining: dict[str, Any],
    threshold: float,
    enforced: bool,
) -> None:
    """Preserve an auditable teacher-gate state without making it rollout-valid."""

    selected = float(
        pretraining.get("selected_teacher_stress_relative_rmse", float("inf"))
    )
    admissible_coverage = float(
        pretraining.get("selected_teacher_stress_admissible_coverage", 0.0)
    )
    minimum_admissible_coverage = float(
        get_cfg(
            cfg,
            "validation.teacher_stress_minimum_admissible_coverage",
            0.0,
        )
    )
    payload = {
        # Deliberately distinct from the schema-v2 rollout checkpoint contract.
        "schema_version": 1,
        "artifact_type": "constitutive_teacher_gate",
        "architecture": "CHP-GNS",
        "dynamics_schema_version": int(model.dynamics_semantics_version),
        "teacher_stress_relative_rmse": selected,
        "teacher_stress_source": str(
            pretraining.get("selected_teacher_stress_source", "unavailable")
        ),
        "teacher_stress_label_coverage": float(
            pretraining.get("selected_teacher_stress_label_coverage", 0.0)
        ),
        "teacher_stress_admissible_coverage": admissible_coverage,
        "teacher_stress_minimum_admissible_coverage": (
            minimum_admissible_coverage
        ),
        "teacher_stress_threshold": float(threshold),
        "teacher_stress_gate_enforced": bool(enforced),
        "teacher_stress_gate_passed": bool(
            selected < float(threshold)
            and admissible_coverage >= minimum_admissible_coverage
        ),
        "scientific_scope": "teacher-forced constitutive audit only; no rollout claim",
        "material_dim": int(material_dim),
        "model": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "normalizers": normalizers.state_dict(),
        "config": cfg,
        "pretraining": pretraining,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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
        root / "dynamics_physics_gate_failure.json",
        root / "dynamics_final_physics_gate_failure.json",
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
    teacher_admissible_coverage = float(
        teacher_metrics.get("teacher_stress_admissible_coverage", 0.0)
    )
    minimum_admissible_coverage = float(
        get_cfg(
            cfg,
            "validation.teacher_stress_minimum_admissible_coverage",
            0.0,
        )
    )
    if (
        bool(get_cfg(cfg, "validation.enforce_teacher_stress_gate", True))
        and (
            teacher_relative >= teacher_threshold
            or teacher_admissible_coverage < minimum_admissible_coverage
        )
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
            "teacher_stress_admissible_coverage": teacher_admissible_coverage,
            "minimum_admissible_coverage": minimum_admissible_coverage,
            "threshold": teacher_threshold,
            "action": (
                "stop rollout curriculum and revise constitutive model/data"
                if teacher_source == CELL_TENSOR_STRESS_SOURCE
                else "stop rollout curriculum and add full tensor labels"
            ),
        }
        _save_json(output_dir / "teacher_stress_gate_failure.json", failure)
        raise RuntimeError(
            "teacher-forced stress gate failed: "
            f"rRMSE={teacher_relative:.4g} (threshold {teacher_threshold:.4g}), "
            f"admissible_coverage={teacher_admissible_coverage:.4g} "
            f"(minimum {minimum_admissible_coverage:.4g})"
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


def _capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream used by sampling, noise, and neural modules."""

    cuda_state = None
    # Avoid initializing CUDA from CPU-only tests and tooling.  Production CHP
    # training has already initialized CUDA before its first checkpoint.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_state = [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda_state,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a checkpoint RNG snapshot immediately before the next epoch."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"checkpoint RNG state is incomplete: {missing}")
    random.setstate(state["python"])
    np.random.set_state(tuple(state["numpy"]))
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"]).detach().cpu())
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value).detach().cpu() for value in cuda_state]
        )


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


def _checkpoint_values_equal(left: Any, right: Any) -> bool:
    """Compare nested checkpoint metadata without device-dependent tensor rules."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return set(left) == set(right) and all(
            _checkpoint_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right, (list, tuple)
        ):
            return False
        return len(left) == len(right) and all(
            _checkpoint_values_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


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


def _atomic_torch_save(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)
