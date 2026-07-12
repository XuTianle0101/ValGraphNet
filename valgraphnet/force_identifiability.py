"""Leakage-safe frozen-potential force identifiability diagnostic.

The diagnostic asks one deliberately narrow question: after excluding
constrained and current contact-neighbour nodes, can the internal force from a
frozen CHP constitutive gate explain the dataset's discrete acceleration with
one positive global inverse-inertia scale?

Only ``alpha = max(0, <q,a>/<q,q>)`` is fitted, where
``q = internal_force / unit_lumped_mass`` and ``a = V[t+1] - V[t]`` for the
exported deforming_plate ``dt=1`` convention.  No graph, contact, damping, or
residual parameter is trained.  Consequently this is a diagnostic of the
frozen potential, not a complete dynamics result and not proof that a force
map is impossible when it fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from valgraphnet.chp_model import CHPGNS, CHPStatic
from valgraphnet.config import get_cfg, load_config
from valgraphnet.hierarchy import TopologyHierarchy
from valgraphnet.mechanics import deformation_gradient, invariants


FORCE_DIAGNOSTIC_SCHEMA_VERSION = 1
FORCE_DIAGNOSTIC_TYPE = "frozen_potential_positive_inverse_inertia"
TARGET_SEMANTICS = "single_global_semi_implicit_dt1"
SCIENTIFIC_DISCLAIMER = (
    "Frozen-potential force identifiability diagnostic only. It excludes current "
    "contact-neighbour nodes and fits one non-negative global inverse-inertia "
    "scale. Failure can expose an unusable frozen internal-force direction, but "
    "does not prove that no jointly trained constitutive/contact/damping model "
    "exists. Success is not full-rollout or full-tensor validation."
)
FORCE_INPUT_ARRAYS = (
    "nodes.npy",
    "cells.npy",
    "Dm_inv.npy",
    "reference_volume.npy",
    "shape_gradients.npy",
    "lumped_mass.npy",
    "fixed_mask.npy",
    "prescribed_mask.npy",
    "times.npy",
    "pressure.npy",
    "U.npy",
    "V.npy",
)


@dataclass(frozen=True)
class ForceCase:
    """Minimal memory-mapped case view used by this diagnostic."""

    case_id: str
    root: Path
    nodes: np.ndarray
    cells: np.ndarray
    dm_inv: np.ndarray
    volume: np.ndarray
    shape_gradients: np.ndarray
    lumped_mass: np.ndarray
    fixed_mask: np.ndarray
    prescribed_mask: np.ndarray
    times: np.ndarray
    pressure: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.displacement.shape[0])


@dataclass
class ForceMetricSums:
    """Pooled float64 moments required by the closed-form positive scale."""

    q_square: float = 0.0
    target_square: float = 0.0
    q_target: float = 0.0
    vector_components: int = 0
    node_samples: int = 0

    def update(self, q: torch.Tensor, target: torch.Tensor) -> None:
        q64 = torch.as_tensor(q).detach().double()
        target64 = torch.as_tensor(target, device=q64.device).detach().double()
        if q64.shape != target64.shape or q64.ndim != 2 or q64.shape[1] != 3:
            raise ValueError("q and target must have the same [N,3] shape")
        if q64.shape[0] < 1:
            raise ValueError("force moments require at least one node sample")
        if not bool(torch.isfinite(q64).all() and torch.isfinite(target64).all()):
            raise ValueError("force moments require finite q and target")
        self.q_square += float(q64.square().sum().cpu())
        self.target_square += float(target64.square().sum().cpu())
        self.q_target += float((q64 * target64).sum().cpu())
        self.vector_components += int(q64.numel())
        self.node_samples += int(q64.shape[0])

    def __add__(self, other: "ForceMetricSums") -> "ForceMetricSums":
        return ForceMetricSums(
            q_square=self.q_square + other.q_square,
            target_square=self.target_square + other.target_square,
            q_target=self.q_target + other.q_target,
            vector_components=self.vector_components + other.vector_components,
            node_samples=self.node_samples + other.node_samples,
        )

    @property
    def unconstrained_alpha(self) -> float:
        if self.q_square <= 0.0:
            return 0.0
        return self.q_target / self.q_square

    @property
    def positive_alpha(self) -> float:
        return max(0.0, self.unconstrained_alpha)

    def metrics(self, alpha: float) -> dict[str, Any]:
        if self.vector_components <= 0 or self.target_square <= 0.0:
            raise ValueError("force metrics require non-zero target samples")
        if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
            raise ValueError("inverse-inertia alpha must be finite and non-negative")
        alpha = float(alpha)
        scaled_error = (
            alpha * alpha * self.q_square
            - 2.0 * alpha * self.q_target
            + self.target_square
        )
        scaled_error = max(scaled_error, 0.0)
        denominator = math.sqrt(max(self.q_square * self.target_square, 0.0))
        cosine = self.q_target / denominator if denominator > 0.0 else 0.0
        return {
            "inverse_inertia_alpha": alpha,
            "alpha_is_strictly_positive": bool(alpha > 0.0),
            "unconstrained_inverse_inertia_alpha": self.unconstrained_alpha,
            "zero_baseline_relative_rmse": 1.0,
            "force_cosine": cosine,
            "positive_scale_relative_rmse": math.sqrt(
                scaled_error / self.target_square
            ),
            "positive_scale_prediction_to_target_rms_ratio": alpha
            * math.sqrt(self.q_square / self.target_square),
            "unit_inertia_q_rms": math.sqrt(
                self.q_square / self.vector_components
            ),
            "target_acceleration_rms": math.sqrt(
                self.target_square / self.vector_components
            ),
            "scaled_prediction_rms": alpha
            * math.sqrt(self.q_square / self.vector_components),
            "node_samples": int(self.node_samples),
            "vector_components": int(self.vector_components),
            "q_square": self.q_square,
            "target_square": self.target_square,
            "q_target": self.q_target,
        }


@dataclass
class ForceCoverage:
    """Counts every fixed protocol exclusion rather than silently dropping data."""

    requested_frames: int = 0
    evaluated_frames: int = 0
    inadmissible_frames: int = 0
    total_node_slots: int = 0
    fixed_excluded: int = 0
    prescribed_excluded: int = 0
    moving_before_contact: int = 0
    contact_radius_excluded: int = 0
    inadmissible_moving_excluded: int = 0
    eligible_nodes: int = 0
    zero_eligible_frames: int = 0

    def __add__(self, other: "ForceCoverage") -> "ForceCoverage":
        values = {
            name: int(getattr(self, name)) + int(getattr(other, name))
            for name in self.__dataclass_fields__
        }
        return ForceCoverage(**values)

    def json_dict(self) -> dict[str, Any]:
        moving = max(self.moving_before_contact, 1)
        total = max(self.total_node_slots, 1)
        return {
            "requested_frames": int(self.requested_frames),
            "evaluated_frames": int(self.evaluated_frames),
            "inadmissible_frames": int(self.inadmissible_frames),
            "total_node_slots": int(self.total_node_slots),
            "fixed_excluded": int(self.fixed_excluded),
            "prescribed_excluded": int(self.prescribed_excluded),
            "moving_before_contact": int(self.moving_before_contact),
            "contact_radius_excluded": int(self.contact_radius_excluded),
            "inadmissible_moving_excluded": int(
                self.inadmissible_moving_excluded
            ),
            "eligible_nodes": int(self.eligible_nodes),
            "zero_eligible_frames": int(self.zero_eligible_frames),
            "eligible_fraction_of_all_node_slots": self.eligible_nodes / total,
            "eligible_fraction_of_moving_node_slots": self.eligible_nodes / moving,
            "contact_exclusion_fraction_of_moving_node_slots": (
                self.contact_radius_excluded / moving
            ),
            "frame_coverage": self.evaluated_frames
            / max(self.requested_frames, 1),
        }


def positive_inverse_inertia(q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return ``max(0, <q,a>/<q,q>)`` without differentiating or normalizing."""

    q64 = torch.as_tensor(q).detach().double()
    target64 = torch.as_tensor(target, device=q64.device).detach().double()
    if q64.shape != target64.shape or q64.numel() == 0:
        raise ValueError("q and target must have the same non-empty shape")
    if not bool(torch.isfinite(q64).all() and torch.isfinite(target64).all()):
        raise ValueError("positive scale fitting requires finite tensors")
    denominator = q64.square().sum()
    if bool((denominator <= 0.0).item()):
        return denominator.new_zeros(())
    return ((q64 * target64).sum() / denominator).clamp_min(0.0)


def single_step_acceleration_target(
    current_velocity: torch.Tensor,
    next_velocity: torch.Tensor,
    *,
    dt: float = 1.0,
) -> torch.Tensor:
    """Return the exported DP target ``V[t+1]-V[t]`` for fixed ``dt=1``.

    This deliberately differs from the current two-substep target.  It is the
    acceleration that makes one global symplectic-Euler step reproduce the
    backward-difference velocity stored by the case converter.
    """

    if not math.isfinite(float(dt)) or float(dt) != 1.0:
        raise ValueError("force diagnostic semantics require exactly dt=1")
    current = torch.as_tensor(current_velocity)
    following = torch.as_tensor(next_velocity, device=current.device)
    if current.shape != following.shape:
        raise ValueError("current and next velocity shapes must match")
    return following - current


def even_indices(size: int, count: int) -> list[int]:
    """Return deterministic endpoint-inclusive even indices."""

    if int(size) < 1 or int(count) < 1:
        raise ValueError("size and count must be positive")
    count = min(int(size), int(count))
    return np.unique(
        np.rint(np.linspace(0, int(size) - 1, count)).astype(int)
    ).tolist()


def transition_frame_indices(num_steps: int, count: int) -> list[int]:
    """Select fixed transition starts from ``0..T-2``."""

    if int(num_steps) < 2:
        raise ValueError("a force case requires at least two frames")
    return even_indices(int(num_steps) - 1, int(count))


def eligible_force_nodes(
    position: torch.Tensor,
    fixed_mask: torch.Tensor,
    prescribed_mask: torch.Tensor,
    contact_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclude constrained and moving nodes within radius of prescribed nodes."""

    position = torch.as_tensor(position).float()
    fixed = torch.as_tensor(fixed_mask, device=position.device, dtype=torch.bool)
    prescribed = torch.as_tensor(
        prescribed_mask, device=position.device, dtype=torch.bool
    )
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape [N,3]")
    if fixed.shape != (position.shape[0],) or prescribed.shape != fixed.shape:
        raise ValueError("constraint masks must have shape [N]")
    if bool((fixed & prescribed).any().item()):
        raise ValueError("fixed and prescribed masks must be disjoint")
    if not math.isfinite(float(contact_radius)) or float(contact_radius) <= 0.0:
        raise ValueError("contact radius must be finite and positive")
    moving = ~(fixed | prescribed)
    near_contact = torch.zeros_like(moving)
    if bool(moving.any().item()) and bool(prescribed.any().item()):
        moving_ids = torch.nonzero(moving, as_tuple=False).flatten()
        distance = torch.cdist(position[moving_ids], position[prescribed])
        within = distance.min(dim=1).values <= float(contact_radius)
        near_contact[moving_ids[within]] = True
    return moving & ~near_contact, near_contact


def validate_force_config(cfg: Mapping[str, Any]) -> None:
    """Fail closed on CPU formal runs, split drift, or target-semantic drift."""

    config = dict(cfg)
    if int(config.get("schema_version", 0)) != FORCE_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {FORCE_DIAGNOSTIC_SCHEMA_VERSION}"
        )
    if str(get_cfg(config, "diagnostic.device", "")).lower() != "cuda":
        raise ValueError("formal force-identifiability diagnostic is CUDA-only")
    if str(get_cfg(config, "diagnostic.mechanics_precision", "")).lower() != "float32":
        raise ValueError("diagnostic.mechanics_precision must be float32")
    if str(get_cfg(config, "diagnostic.target_semantics", "")) != TARGET_SEMANTICS:
        raise ValueError(f"diagnostic.target_semantics must be {TARGET_SEMANTICS}")
    if float(get_cfg(config, "diagnostic.time_step", 0.0)) != 1.0:
        raise ValueError("force diagnostic requires time_step=1")
    split_names = tuple(
        str(get_cfg(config, f"data.{name}_split", ""))
        for name in ("train", "val", "test")
    )
    if split_names != ("train", "val", "test"):
        raise ValueError("literal distinct train/val/test split names are required")
    expected_steps = int(get_cfg(config, "data.expected_steps", 0))
    if expected_steps != 400:
        raise ValueError("deforming_plate force diagnostic requires 400 frames")
    train_cases = int(get_cfg(config, "fit.cases", 0))
    val_cases = int(get_cfg(config, "validation.cases", 0))
    if train_cases < 1:
        raise ValueError("fit.cases must be positive")
    if val_cases != 20:
        raise ValueError("formal force diagnostic requires even val20")
    if str(get_cfg(config, "fit.case_selection", "")) != "even":
        raise ValueError("fit.case_selection must be even")
    if str(get_cfg(config, "validation.case_selection", "")) != "even":
        raise ValueError("validation.case_selection must be even")
    expected_train_frames = transition_frame_indices(expected_steps, 8)
    expected_val_frames = transition_frame_indices(expected_steps, 16)
    train_frames = [int(value) for value in get_cfg(config, "fit.frames", [])]
    val_frames = [
        int(value) for value in get_cfg(config, "validation.frames", [])
    ]
    if train_frames != expected_train_frames:
        raise ValueError(f"fit.frames must equal fixed {expected_train_frames}")
    if val_frames != expected_val_frames:
        raise ValueError(
            f"validation.frames must equal fixed {expected_val_frames}"
        )
    if not get_cfg(config, "checkpoint.path", None):
        raise ValueError("checkpoint.path is required")
    if not get_cfg(config, "diagnostic.output_dir", None):
        raise ValueError("diagnostic.output_dir is required")
    positive = {
        "contact.radius": get_cfg(config, "contact.radius", None),
        "admissibility.minimum_j": get_cfg(
            config, "admissibility.minimum_j", None
        ),
        "admissibility.maximum_i2_bar": get_cfg(
            config, "admissibility.maximum_i2_bar", None
        ),
        "admissibility.maximum_i1_bar": get_cfg(
            config, "admissibility.maximum_i1_bar", None
        ),
    }
    invalid = [
        name
        for name, value in positive.items()
        if value is None
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ]
    if invalid:
        raise ValueError(f"finite positive protocol values required: {invalid}")


def development_case_dirs(
    cfg: Mapping[str, Any],
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    """Resolve selected train/val directories without opening any test case."""

    validate_force_config(cfg)
    config = dict(cfg)
    root = Path(get_cfg(config, "data.root")).resolve()
    split_file = Path(get_cfg(config, "data.split_file")).resolve()
    with split_file.open("r", encoding="utf-8") as stream:
        splits = json.load(stream)
    if not isinstance(splits, Mapping):
        raise ValueError("split file must contain a mapping")
    names = {
        name: str(get_cfg(config, f"data.{name}_split"))
        for name in ("train", "val", "test")
    }
    ids = {
        name: [str(value) for value in splits.get(split_name, [])]
        for name, split_name in names.items()
    }
    if any(not values for values in ids.values()):
        raise ValueError("train, validation, and test manifests must be non-empty")
    for name, values in ids.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{name} split contains duplicate case ids")
        unsafe = [
            value
            for value in values
            if not value
            or Path(value).is_absolute()
            or len(Path(value).parts) != 1
            or Path(value).name != value
            or value in {".", ".."}
        ]
        if unsafe:
            raise ValueError(f"{name} split contains unsafe case ids: {unsafe[:3]}")
    sets = {name: set(values) for name, values in ids.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("train, validation, and test case ids must be disjoint")
    train_count = int(get_cfg(config, "fit.cases"))
    val_count = int(get_cfg(config, "validation.cases"))
    if len(ids["train"]) < train_count or len(ids["val"]) < val_count:
        raise ValueError("development split is smaller than the requested protocol")
    selected_train = [
        ids["train"][index]
        for index in even_indices(len(ids["train"]), train_count)
    ]
    selected_val = [
        ids["val"][index]
        for index in even_indices(len(ids["val"]), val_count)
    ]
    train_dirs = [(root / case_id).resolve() for case_id in selected_train]
    val_dirs = [(root / case_id).resolve() for case_id in selected_val]
    for path in (*train_dirs, *val_dirs):
        if path.parent != root:
            raise ValueError("selected development case resolves outside data root")
        if not path.is_dir():
            raise FileNotFoundError(f"selected development case is missing: {path}")
    return train_dirs, val_dirs, {
        "train_split": names["train"],
        "validation_split": names["val"],
        "test_split_name_only": names["test"],
        "selected_train_case_ids": selected_train,
        "selected_validation_case_ids": selected_val,
        "test_identifiers_read_for_overlap_audit": True,
        "test_case_arrays_loaded": 0,
        "test_content_accessed": False,
    }


def require_cuda(cfg: Mapping[str, Any]) -> torch.device:
    """Return the sole formal execution device; CPU fallback is forbidden."""

    validate_force_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal force identifiability")
    requested = torch.device(str(get_cfg(dict(cfg), "diagnostic.device", "cuda")))
    device = torch.device("cuda", requested.index or 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return device


def load_force_case(case_dir: str | Path) -> ForceCase:
    """Memory-map only arrays named in :data:`FORCE_INPUT_ARRAYS`."""

    root = Path(case_dir).resolve()
    arrays = {
        name: np.load(root / name, mmap_mode="r", allow_pickle=False)
        for name in FORCE_INPUT_ARRAYS
    }
    case = ForceCase(
        case_id=root.name,
        root=root,
        nodes=arrays["nodes.npy"],
        cells=arrays["cells.npy"],
        dm_inv=arrays["Dm_inv.npy"],
        volume=arrays["reference_volume.npy"],
        shape_gradients=arrays["shape_gradients.npy"],
        lumped_mass=arrays["lumped_mass.npy"],
        fixed_mask=arrays["fixed_mask.npy"],
        prescribed_mask=arrays["prescribed_mask.npy"],
        times=arrays["times.npy"],
        pressure=arrays["pressure.npy"],
        displacement=arrays["U.npy"],
        velocity=arrays["V.npy"],
    )
    _validate_force_case(case, expected_steps=400)
    return case


def _validate_force_case(
    case: ForceCase, *, expected_steps: int | None = None
) -> None:
    n, m, t = case.num_nodes, case.num_cells, case.num_steps
    expected = {
        "nodes": (n, 3),
        "cells": (m, 4),
        "dm_inv": (m, 3, 3),
        "shape_gradients": (m, 4, 3),
        "fixed_mask": (n,),
        "prescribed_mask": (n,),
        "times": (t,),
        "pressure": (t,),
        "displacement": (t, n, 3),
        "velocity": (t, n, 3),
    }
    for name, shape in expected.items():
        if tuple(getattr(case, name).shape) != shape:
            raise ValueError(
                f"{case.case_id}: {name} shape {getattr(case, name).shape} != {shape}"
            )
    if tuple(case.volume.shape) not in {(m,), (m, 1)}:
        raise ValueError(f"{case.case_id}: reference_volume must have M values")
    if tuple(case.lumped_mass.shape) not in {(n,), (n, 1)}:
        raise ValueError(f"{case.case_id}: lumped_mass must have N values")
    if t < 2 or n < 1 or m < 1:
        raise ValueError(f"{case.case_id}: force case is empty")
    if expected_steps is not None and t != int(expected_steps):
        raise ValueError(
            f"{case.case_id}: expected {int(expected_steps)} frames, got {t}"
        )
    times = np.asarray(case.times, dtype=np.float64)
    if not np.isfinite(times).all() or not np.all(np.diff(times) == 1.0):
        raise ValueError(
            f"{case.case_id}: times must be finite with strictly dt=1"
        )
    pressure = np.asarray(case.pressure, dtype=np.float64)
    if not np.isfinite(pressure).all() or np.any(pressure != 0.0):
        raise ValueError(
            f"{case.case_id}: internal-only diagnostic requires finite zero pressure"
        )
    cells = np.asarray(case.cells)
    if cells.min() < 0 or cells.max() >= n:
        raise ValueError(f"{case.case_id}: cell index is outside node range")
    if np.any(np.asarray(case.lumped_mass) <= 0.0):
        raise ValueError(f"{case.case_id}: lumped masses must be positive")
    if np.any(np.asarray(case.fixed_mask, bool) & np.asarray(case.prescribed_mask, bool)):
        raise ValueError(f"{case.case_id}: fixed and prescribed masks overlap")


def build_force_static(case: ForceCase, device: torch.device | str) -> CHPStatic:
    """Build only the static fields consumed by ``constitutive_fields``."""

    def tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.array(value, copy=True)).to(
            device=device, dtype=dtype
        )

    n, m = case.num_nodes, case.num_cells
    return CHPStatic(
        reference_position=tensor(case.nodes, torch.float32),
        cells=tensor(case.cells, torch.long),
        mesh_edge_index=torch.zeros((2, 0), device=device, dtype=torch.long),
        dm_inv=tensor(case.dm_inv, torch.float32),
        volume=tensor(case.volume, torch.float32).reshape(-1),
        shape_gradients=tensor(case.shape_gradients, torch.float32),
        lumped_mass=tensor(case.lumped_mass, torch.float32).reshape(-1),
        fixed_mask=tensor(case.fixed_mask, torch.bool),
        prescribed_mask=tensor(case.prescribed_mask, torch.bool),
        contact_surface_mask=torch.zeros(n, device=device, dtype=torch.bool),
        material_features=torch.zeros((m, 0), device=device, dtype=torch.float32),
        fiber_direction=torch.zeros((m, 3), device=device, dtype=torch.float32),
        hierarchy=TopologyHierarchy([], [], [n]),
    )


@torch.inference_mode()
def evaluate_force_case(
    model: Any,
    case: ForceCase,
    frame_indices: Sequence[int],
    *,
    device: torch.device | str,
    contact_radius: float,
    minimum_j: float,
    maximum_i1_bar: float,
    maximum_i2_bar: float,
) -> tuple[ForceMetricSums, ForceCoverage]:
    """Accumulate exact-geometry force moments for one case."""

    _validate_force_case(case)
    static = build_force_static(case, device)
    sums = ForceMetricSums()
    coverage = ForceCoverage(requested_frames=len(frame_indices))
    for frame in frame_indices:
        frame = int(frame)
        if frame < 0 or frame >= case.num_steps - 1:
            raise ValueError(f"{case.case_id}: transition frame {frame} is invalid")
        displacement = torch.from_numpy(
            np.array(case.displacement[frame], copy=True)
        ).to(device=device, dtype=torch.float32)
        position = static.reference_position + displacement
        current_velocity = torch.from_numpy(
            np.array(case.velocity[frame], copy=True)
        ).to(device=device, dtype=torch.float32)
        next_velocity = torch.from_numpy(
            np.array(case.velocity[frame + 1], copy=True)
        ).to(device=device, dtype=torch.float32)
        target = single_step_acceleration_target(
            current_velocity, next_velocity, dt=1.0
        )
        coverage.total_node_slots += case.num_nodes
        coverage.fixed_excluded += int(static.fixed_mask.sum().item())
        coverage.prescribed_excluded += int(static.prescribed_mask.sum().item())
        moving_count = int(
            (~(static.fixed_mask | static.prescribed_mask)).sum().item()
        )
        coverage.moving_before_contact += moving_count
        deformation = deformation_gradient(position, static.cells, static.dm_inv)
        state = invariants(deformation)
        finite_mechanics = (
            torch.isfinite(deformation).all()
            & torch.isfinite(state.c).all()
            & torch.isfinite(state.i1).all()
            & torch.isfinite(state.i2).all()
            & torch.isfinite(state.j).all()
            & torch.isfinite(state.i1_bar).all()
            & torch.isfinite(state.i2_bar).all()
        )
        admissible = bool(
            (
                finite_mechanics
                & (state.j.min() >= float(minimum_j))
                & (state.i1_bar.max() <= float(maximum_i1_bar))
                & (state.i2_bar.max() <= float(maximum_i2_bar))
            ).item()
        )
        if not admissible:
            coverage.inadmissible_frames += 1
            coverage.inadmissible_moving_excluded += moving_count
            continue
        eligible, near_contact = eligible_force_nodes(
            position,
            static.fixed_mask,
            static.prescribed_mask,
            contact_radius,
        )
        contact_count = int(near_contact.sum().item())
        eligible_count = int(eligible.sum().item())
        coverage.contact_radius_excluded += contact_count
        coverage.eligible_nodes += eligible_count
        if eligible_count == 0:
            coverage.zero_eligible_frames += 1
            continue
        fields = model.constitutive_fields(
            static, position, deformation=deformation
        )
        q = fields.internal_force / static.lumped_mass[:, None]
        sums.update(q[eligible], target[eligible])
        coverage.evaluated_frames += 1
    return sums, coverage


@torch.inference_mode()
def evaluate_force_cases(
    model: Any,
    case_dirs: Sequence[Path],
    frame_indices: Sequence[int],
    *,
    device: torch.device | str,
    contact_radius: float,
    minimum_j: float,
    maximum_i1_bar: float,
    maximum_i2_bar: float,
    fixed_alpha: float | None,
) -> tuple[ForceMetricSums, ForceCoverage, list[dict[str, Any]]]:
    """Evaluate a fixed case/frame set, optionally with a frozen train alpha."""

    total = ForceMetricSums()
    coverage = ForceCoverage()
    records: list[tuple[str, ForceMetricSums, ForceCoverage]] = []
    for case_dir in case_dirs:
        case = load_force_case(case_dir)
        case_sums, case_coverage = evaluate_force_case(
            model,
            case,
            frame_indices,
            device=device,
            contact_radius=contact_radius,
            minimum_j=minimum_j,
            maximum_i1_bar=maximum_i1_bar,
            maximum_i2_bar=maximum_i2_bar,
        )
        total = total + case_sums
        coverage = coverage + case_coverage
        records.append((case.case_id, case_sums, case_coverage))
    alpha = total.positive_alpha if fixed_alpha is None else float(fixed_alpha)
    per_case: list[dict[str, Any]] = []
    for case_id, case_sums, case_coverage in records:
        row: dict[str, Any] = {
            "case_id": case_id,
            "coverage": case_coverage.json_dict(),
        }
        if case_sums.target_square > 0.0 and case_sums.vector_components > 0:
            row["metrics"] = case_sums.metrics(alpha)
        else:
            row["metrics"] = None
        per_case.append(row)
    return total, coverage, per_case


def load_frozen_potential(
    checkpoint_path: Path,
    device: torch.device,
    cfg: Mapping[str, Any],
) -> tuple[CHPGNS, Mapping[str, Any]]:
    """Validate and load a passed deforming_plate constitutive gate."""

    checkpoint = _torch_load(checkpoint_path, device)
    if checkpoint.get("artifact_type") != "constitutive_teacher_gate":
        raise ValueError("checkpoint must be a constitutive_teacher_gate artifact")
    if checkpoint.get("architecture") != "CHP-GNS":
        raise ValueError("checkpoint architecture must be CHP-GNS")
    if not bool(checkpoint.get("teacher_stress_gate_passed", False)):
        raise ValueError("checkpoint did not pass its teacher stress gate")
    if int(checkpoint.get("material_dim", -1)) != 0:
        raise ValueError("deforming_plate force diagnostic requires material_dim=0")
    embedded = checkpoint.get("config")
    state = checkpoint.get("model")
    if not isinstance(embedded, Mapping) or not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing embedded config or model state")
    if int(get_cfg(dict(embedded), "model.fiber_order", 0)) != 0:
        raise ValueError("deforming_plate force diagnostic requires fiber_order=0")
    for key in ("root", "split_file"):
        embedded_path = Path(get_cfg(dict(embedded), f"data.{key}")).resolve()
        diagnostic_path = Path(get_cfg(dict(cfg), f"data.{key}")).resolve()
        if embedded_path != diagnostic_path:
            raise ValueError(
                f"diagnostic data.{key} differs from frozen checkpoint"
            )
    embedded_splits = tuple(
        str(get_cfg(dict(embedded), f"data.{name}_split", ""))
        for name in ("train", "val", "test")
    )
    if embedded_splits != ("train", "val", "test"):
        raise ValueError("frozen checkpoint does not use literal train/val/test")
    checkpoint_radius = float(get_cfg(dict(embedded), "contact.radius", -1.0))
    diagnostic_radius = float(get_cfg(dict(cfg), "contact.radius"))
    if checkpoint_radius != diagnostic_radius:
        raise ValueError("diagnostic contact radius differs from frozen checkpoint")
    model = CHPGNS(dict(embedded), material_dim=0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def run_force_identifiability(
    cfg: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Path]:
    """Fit on train only, freeze alpha, then evaluate even-val20 exactly once."""

    validate_force_config(cfg)
    config_file = Path(config_path).resolve()
    disk_cfg = load_config(config_file)
    runtime_config_hash = _canonical_sha256(cfg)
    if _canonical_sha256(disk_cfg) != runtime_config_hash:
        raise RuntimeError("runtime config differs from the frozen YAML file")
    device = require_cuda(cfg)
    torch.cuda.reset_peak_memory_stats(device)
    train_dirs, val_dirs, split_audit = development_case_dirs(cfg)
    checkpoint_path = Path(get_cfg(cfg, "checkpoint.path")).resolve()
    split_file = Path(get_cfg(cfg, "data.split_file")).resolve()
    output_dir = Path(get_cfg(cfg, "diagnostic.output_dir")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_path = output_dir / "frozen_train_fit.json"
    metrics_path = output_dir / "metrics.val20.json"
    provenance_path = output_dir / "provenance.json"
    train_manifest_path = output_dir / "train_inputs.before_fit.sha256.json"
    val_manifest_path = output_dir / "validation_inputs.after_fit.sha256.json"
    source_manifest_path = output_dir / "source_files.sha256.json"
    artifacts = (
        fit_path,
        metrics_path,
        provenance_path,
        train_manifest_path,
        val_manifest_path,
        source_manifest_path,
    )
    if any(path.exists() for path in artifacts):
        raise RuntimeError("formal force diagnostic artifacts exist; refusing overwrite")
    checkpoint_record = _file_fingerprint(checkpoint_path)
    train_manifest = _input_manifest("train", train_dirs)
    source_manifest = _source_manifest((config_file, split_file))
    _write_json(train_manifest_path, train_manifest)
    _write_json(source_manifest_path, source_manifest)
    model, checkpoint = load_frozen_potential(checkpoint_path, device, cfg)

    contact_radius = float(get_cfg(cfg, "contact.radius"))
    minimum_j = float(get_cfg(cfg, "admissibility.minimum_j"))
    maximum_i1 = float(get_cfg(cfg, "admissibility.maximum_i1_bar"))
    maximum_i2 = float(get_cfg(cfg, "admissibility.maximum_i2_bar"))
    train_frames = [int(value) for value in get_cfg(cfg, "fit.frames")]
    train_sums, train_coverage, train_cases = evaluate_force_cases(
        model,
        train_dirs,
        train_frames,
        device=device,
        contact_radius=contact_radius,
        minimum_j=minimum_j,
        maximum_i1_bar=maximum_i1,
        maximum_i2_bar=maximum_i2,
        fixed_alpha=None,
    )
    alpha = train_sums.positive_alpha
    fit_payload = {
        "schema_version": FORCE_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_type": FORCE_DIAGNOSTIC_TYPE,
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        "fit_split": "train",
        "validation_content_accessed_before_fit_freeze": False,
        "test_content_accessed": False,
        "checkpoint_sha256": checkpoint_record["sha256"],
        "checkpoint_embedded_config_sha256": _canonical_sha256(
            checkpoint["config"]
        ),
        "runtime_config_sha256": runtime_config_hash,
        "target_semantics": _target_semantics_payload(),
        "admissibility": {
            "minimum_j": minimum_j,
            "maximum_i1_bar": maximum_i1,
            "maximum_i2_bar": maximum_i2,
            "finite_required": ["F", "C", "I1", "I2", "J", "I1_bar", "I2_bar"],
        },
        "selected_case_ids": [path.name for path in train_dirs],
        "selected_transition_frames": train_frames,
        "metrics": train_sums.metrics(alpha),
        "coverage": train_coverage.json_dict(),
        "per_case": train_cases,
        "train_input_manifest_sha256": train_manifest["aggregate_sha256"],
    }
    _write_json(fit_path, fit_payload)
    fit_sha = _sha256(fit_path)
    frozen_fit = json.loads(fit_path.read_text(encoding="utf-8"))
    frozen_alpha = float(frozen_fit["metrics"]["inverse_inertia_alpha"])
    if frozen_alpha != alpha:
        raise RuntimeError("serialized train-only alpha differs from the fitted value")
    alpha = frozen_alpha

    # Validation arrays are first opened/fingerprinted after alpha is frozen.
    val_manifest = _input_manifest("validation", val_dirs)
    _write_json(val_manifest_path, val_manifest)
    val_frames = [int(value) for value in get_cfg(cfg, "validation.frames")]
    val_sums, val_coverage, val_cases = evaluate_force_cases(
        model,
        val_dirs,
        val_frames,
        device=device,
        contact_radius=contact_radius,
        minimum_j=minimum_j,
        maximum_i1_bar=maximum_i1,
        maximum_i2_bar=maximum_i2,
        fixed_alpha=alpha,
    )
    data_hash = _canonical_sha256(
        {
            "train": train_manifest["aggregate_sha256"],
            "validation": val_manifest["aggregate_sha256"],
        }
    )
    metrics = {
        "schema_version": FORCE_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_type": FORCE_DIAGNOSTIC_TYPE,
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        "evaluation_split": "val",
        "case_selection": "even",
        "requested_cases": 20,
        "selected_case_ids": [path.name for path in val_dirs],
        "selected_transition_frames": val_frames,
        "validation_evaluation_count": 1,
        "alpha_source": "frozen_train_only_closed_form_fit",
        "frozen_train_fit_sha256": fit_sha,
        "checkpoint_sha256": checkpoint_record["sha256"],
        "checkpoint_embedded_config_sha256": _canonical_sha256(
            checkpoint["config"]
        ),
        "config_sha256": _sha256(config_file),
        "runtime_config_sha256": runtime_config_hash,
        "development_data_sha256": data_hash,
        "split_manifest_sha256": _sha256(split_file),
        "target_semantics": _target_semantics_payload(),
        "admissibility": {
            "minimum_j": minimum_j,
            "maximum_i1_bar": maximum_i1,
            "maximum_i2_bar": maximum_i2,
            "finite_required": ["F", "C", "I1", "I2", "J", "I1_bar", "I2_bar"],
        },
        "metrics": val_sums.metrics(alpha),
        "coverage": val_coverage.json_dict(),
        "per_case": val_cases,
        "teacher_stress_gate_context": {
            "relative_rmse": checkpoint.get("teacher_stress_relative_rmse"),
            "source": checkpoint.get("teacher_stress_source"),
            "label_coverage": checkpoint.get("teacher_stress_label_coverage"),
        },
        "test_case_arrays_loaded": 0,
        "test_content_accessed": False,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    _write_json(metrics_path, metrics)
    _verify_manifest_unchanged(train_manifest)
    _verify_manifest_unchanged(val_manifest)
    _verify_manifest_unchanged(source_manifest)
    _verify_manifest_unchanged({"files": [checkpoint_record]})
    if _sha256(fit_path) != fit_sha:
        raise RuntimeError("frozen train-only fit changed during validation")

    repository_root = Path(__file__).resolve().parents[1]
    provenance = {
        "schema_version": FORCE_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_type": FORCE_DIAGNOSTIC_TYPE,
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        "created_at_utc": _utc_now(),
        "configuration": _file_fingerprint(config_file),
        "checkpoint": checkpoint_record,
        "checkpoint_embedded_config_sha256": _canonical_sha256(
            checkpoint["config"]
        ),
        "split_manifest": _file_fingerprint(split_file),
        "development_data_sha256": data_hash,
        "development_split_audit": split_audit,
        "frozen_train_fit": {
            **_file_fingerprint(fit_path),
            "frozen_before_validation_arrays_opened": True,
        },
        "validation_metrics": {
            **_file_fingerprint(metrics_path),
            "evaluation_count": 1,
        },
        "train_input_manifest_before_fit": {
            **_file_fingerprint(train_manifest_path),
            "aggregate_sha256": train_manifest["aggregate_sha256"],
            "test_case_arrays_hashed": 0,
        },
        "validation_input_manifest_after_fit_freeze": {
            **_file_fingerprint(val_manifest_path),
            "aggregate_sha256": val_manifest["aggregate_sha256"],
            "frozen_train_fit_sha256_before_manifest": fit_sha,
            "test_case_arrays_hashed": 0,
        },
        "source_file_manifest": {
            **_file_fingerprint(source_manifest_path),
            "aggregate_sha256": source_manifest["aggregate_sha256"],
        },
        "target_semantics": _target_semantics_payload(),
        "precision": {
            "formal_device": "CUDA only",
            "deformation_stress_force_assembly": "float32",
            "pooled_fit_and_metrics": "float64",
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "source_revision": _git_revision(repository_root),
        "test_case_arrays_loaded": 0,
        "test_content_accessed": False,
    }
    _write_json(provenance_path, provenance)
    return {
        "fit": fit_path,
        "metrics": metrics_path,
        "provenance": provenance_path,
    }


def _target_semantics_payload() -> dict[str, Any]:
    return {
        "definition": "A[t] = V[t+1] - V[t]",
        "time_step": 1.0,
        "integrator_interpretation": "one global semi-implicit Euler step",
        "fitted_quantity": "alpha=max(0,<q,A>/<q,q>)",
        "q_definition": "potential_internal_force / unit_lumped_mass",
        "excluded_physics": ["contact", "mesh_damping", "residual_force"],
    }


def _input_manifest(split: str, case_dirs: Sequence[Path]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        arrays = [
            {"name": name, **_file_fingerprint(Path(case_dir) / name)}
            for name in FORCE_INPUT_ARRAYS
        ]
        cases.append(
            {"split": split, "case_id": Path(case_dir).name, "arrays": arrays}
        )
    return {
        "schema_version": 1,
        "split": split,
        "hash_scope": "complete selected force-diagnostic input arrays",
        "test_case_arrays_hashed": 0,
        "cases": cases,
        "aggregate_sha256": _canonical_sha256(cases),
    }


def _source_manifest(protocol_files: Sequence[str | Path]) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    candidates = {
        Path(__file__).resolve(),
        Path(deformation_gradient.__code__.co_filename).resolve(),
        (repository_root / "valgraphnet" / "chp_model.py").resolve(),
        (repository_root / "scripts" / "run_force_identifiability.py").resolve(),
        *(Path(path).resolve() for path in protocol_files),
    }
    files = [_file_fingerprint(path) for path in sorted(candidates) if path.is_file()]
    if len(files) != len(candidates):
        missing = [str(path) for path in candidates if not path.is_file()]
        raise FileNotFoundError(f"diagnostic source file is missing: {missing[0]}")
    return {
        "schema_version": 1,
        "files": files,
        "aggregate_sha256": _canonical_sha256(files),
    }


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"cannot fingerprint missing file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(resolved),
    }


def _verify_manifest_unchanged(manifest: Mapping[str, Any]) -> None:
    records: list[Mapping[str, Any]] = []
    if isinstance(manifest.get("files"), list):
        records.extend(manifest["files"])
    for case in manifest.get("cases", []):
        if isinstance(case, Mapping) and isinstance(case.get("arrays"), list):
            records.extend(case["arrays"])
    for record in records:
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise RuntimeError(f"frozen diagnostic input disappeared: {path}")
        stat = path.stat()
        if (
            int(record.get("size_bytes", -1)) != stat.st_size
            or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
            or str(record.get("sha256", "")) != _sha256(path)
        ):
            raise RuntimeError(f"frozen diagnostic input changed during run: {path}")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid checkpoint mapping: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_revision(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "FORCE_DIAGNOSTIC_SCHEMA_VERSION",
    "FORCE_DIAGNOSTIC_TYPE",
    "FORCE_INPUT_ARRAYS",
    "SCIENTIFIC_DISCLAIMER",
    "TARGET_SEMANTICS",
    "ForceCase",
    "ForceCoverage",
    "ForceMetricSums",
    "development_case_dirs",
    "eligible_force_nodes",
    "evaluate_force_case",
    "even_indices",
    "positive_inverse_inertia",
    "run_force_identifiability",
    "single_step_acceleration_target",
    "transition_frame_indices",
    "validate_force_config",
]
