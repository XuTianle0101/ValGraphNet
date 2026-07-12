"""CUDA-only scalar-stress identifiability control for deforming_plate.

This module is intentionally separate from CHP-GNS.  It learns a direct,
non-negative scalar value per tetrahedron from objective deformation
invariants and applies the same reference-volume weighted nodal projection as
CHP.  It is a diagnostic decoder, not an energy potential, and cannot provide
stress-tensor, internal-force, or constitutive-consistency evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from valgraphnet.config import get_cfg, load_config
from valgraphnet.mechanics import (
    DeformationInvariants,
    deformation_gradient,
    invariants,
    project_cell_to_nodes,
)


CONTROL_SCHEMA_VERSION = 1
DIAGNOSTIC_TYPE = "direct_nonnegative_cell_scalar_stress_decoder"
SCIENTIFIC_DISCLAIMER = (
    "Diagnostic direct scalar decoder only. It is not a scalar energy potential, "
    "does not produce a full stress tensor or internal force, and is not evidence "
    "of constitutive consistency. A high diagnostic error alone also cannot prove "
    "that an objective constitutive map does not exist."
)
CONTROL_INPUT_ARRAYS = (
    "nodes.npy",
    "cells.npy",
    "Dm_inv.npy",
    "reference_volume.npy",
    "U.npy",
    "S.npy",
    "prescribed_mask.npy",
)


@dataclass(frozen=True)
class TrainOnlyStatistics:
    """Feature and target scales fitted exclusively on selected train frames."""

    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    stress_rms: float
    stress_mean: float
    cell_samples: int
    nodal_samples: int
    requested_frames: int
    admissible_frames: int

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.detach().cpu(),
            "feature_std": self.feature_std.detach().cpu(),
            "stress_rms": float(self.stress_rms),
            "stress_mean": float(self.stress_mean),
            "cell_samples": int(self.cell_samples),
            "nodal_samples": int(self.nodal_samples),
            "requested_frames": int(self.requested_frames),
            "admissible_frames": int(self.admissible_frames),
        }

    def json_dict(self) -> dict[str, Any]:
        state = self.state_dict()
        state["feature_mean"] = state["feature_mean"].tolist()
        state["feature_std"] = state["feature_std"].tolist()
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TrainOnlyStatistics":
        return cls(
            feature_mean=torch.as_tensor(state["feature_mean"]).detach().float().cpu(),
            feature_std=torch.as_tensor(state["feature_std"]).detach().float().cpu(),
            stress_rms=float(state["stress_rms"]),
            stress_mean=float(state["stress_mean"]),
            cell_samples=int(state["cell_samples"]),
            nodal_samples=int(state["nodal_samples"]),
            requested_frames=int(state["requested_frames"]),
            admissible_frames=int(state["admissible_frames"]),
        )


@dataclass(frozen=True)
class ControlCase:
    """Minimal memory-mapped view required by the scalar control."""

    case_id: str
    root: Path
    nodes: np.ndarray
    cells: np.ndarray
    dm_inv: np.ndarray
    reference_volume: np.ndarray
    displacement: np.ndarray
    stress: np.ndarray
    prescribed_mask: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.displacement.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

@dataclass
class StressErrorSums:
    """Pooled physical-unit error sums with per-frame peak regions."""

    squared_error: float = 0.0
    squared_reference: float = 0.0
    count: int = 0
    peak_squared_error: float = 0.0
    peak_squared_reference: float = 0.0
    peak_count: int = 0
    nonzero_count: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().float().reshape(-1)
        target = target.detach().float().reshape(-1)
        if prediction.shape != target.shape or target.numel() == 0:
            raise ValueError("prediction and target must have the same non-empty shape")
        if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
            raise ValueError("stress metrics require finite predictions and targets")
        residual = prediction.double() - target.double()
        target64 = target.double()
        threshold = torch.quantile(target.abs(), 0.95)
        peak = target.abs() >= threshold
        self.squared_error += float(residual.square().sum().cpu())
        self.squared_reference += float(target64.square().sum().cpu())
        self.count += int(target.numel())
        self.peak_squared_error += float(residual[peak].square().sum().cpu())
        self.peak_squared_reference += float(target64[peak].square().sum().cpu())
        self.peak_count += int(peak.sum().cpu())
        self.nonzero_count += int((target != 0.0).sum().cpu())

    def __add__(self, other: "StressErrorSums") -> "StressErrorSums":
        return StressErrorSums(
            squared_error=self.squared_error + other.squared_error,
            squared_reference=self.squared_reference + other.squared_reference,
            count=self.count + other.count,
            peak_squared_error=self.peak_squared_error + other.peak_squared_error,
            peak_squared_reference=(
                self.peak_squared_reference + other.peak_squared_reference
            ),
            peak_count=self.peak_count + other.peak_count,
            nonzero_count=self.nonzero_count + other.nonzero_count,
        )

    def metrics(self) -> dict[str, float]:
        if self.count <= 0 or self.squared_reference <= 0.0:
            raise ValueError("pooled stress metrics require a positive reference")
        if self.peak_count <= 0 or self.peak_squared_reference <= 0.0:
            raise ValueError("P95 stress metrics require a positive peak reference")
        relative = math.sqrt(self.squared_error / self.squared_reference)
        peak_relative = math.sqrt(
            self.peak_squared_error / self.peak_squared_reference
        )
        return {
            "teacher_stress_relative_rmse": relative,
            "teacher_stress_p95_relative_rmse": peak_relative,
            "teacher_stress_pooled_relative_rmse": relative,
            "teacher_stress_per_frame_p95_relative_rmse": peak_relative,
            "teacher_stress_rmse": math.sqrt(self.squared_error / self.count),
            "evaluated_nodes": float(self.count),
            "p95_nodes": float(self.peak_count),
            "target_nonzero_fraction": self.nonzero_count / self.count,
        }


class CellScalarStressMLP(nn.Module):
    """Small direct cell-scalar decoder with a non-negative softplus output."""

    def __init__(
        self,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        *,
        initial_dimensionless_mean: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("hidden_dim and hidden_layers must be positive")
        layers: list[nn.Module] = []
        input_dim = 3
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU()))
            input_dim = hidden_dim
        output = nn.Linear(input_dim, 1)
        nn.init.zeros_(output.weight)
        initial = max(float(initial_dimensionless_mean), 1.0e-6)
        nn.init.constant_(output.bias, math.log(math.expm1(initial)))
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, normalized_invariants: torch.Tensor) -> torch.Tensor:
        if normalized_invariants.shape[-1] != 3:
            raise ValueError("cell invariant input must end in three features")
        return functional.softplus(self.network(normalized_invariants))


def objective_invariant_features(
    position: torch.Tensor,
    cells: torch.Tensor,
    dm_inv: torch.Tensor,
) -> tuple[torch.Tensor, DeformationInvariants]:
    """Return ``I1_bar-3, I2_bar-3, J-1`` in mechanics FP32."""

    position32 = torch.as_tensor(position).float()
    cells = torch.as_tensor(cells, device=position32.device, dtype=torch.long)
    dm_inv32 = torch.as_tensor(
        dm_inv, device=position32.device, dtype=torch.float32
    )
    with torch.autocast(device_type=position32.device.type, enabled=False):
        state = invariants(deformation_gradient(position32, cells, dm_inv32))
        features = torch.stack(
            (state.i1_bar - 3.0, state.i2_bar - 3.0, state.j - 1.0), dim=-1
        ).float()
    return features, state


def volume_weighted_nodal_projection(
    cell_scalar: torch.Tensor,
    cells: torch.Tensor,
    volume: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Apply CHP's reference-volume weighted projection in FP32.

    ``cell_scalar`` may be ``[M,1]`` or ``[B,M,1]``.  The latter is projected
    one frame at a time because CHP's primitive takes the cell axis first.
    """

    values = torch.as_tensor(cell_scalar).float()
    cells = torch.as_tensor(cells, device=values.device, dtype=torch.long)
    volume = torch.as_tensor(volume, device=values.device, dtype=torch.float32).reshape(-1)
    with torch.autocast(device_type=values.device.type, enabled=False):
        if values.ndim == 2:
            return project_cell_to_nodes(
                values, cells, int(num_nodes), weights=volume
            ).float()
        if values.ndim != 3:
            raise ValueError("cell scalar must have shape [M,1] or [B,M,1]")
        return torch.stack(
            [
                project_cell_to_nodes(frame, cells, int(num_nodes), weights=volume)
                for frame in values
            ],
            dim=0,
        ).float()


def predict_nodal_scalar_stress(
    model: CellScalarStressMLP,
    invariant_features: torch.Tensor,
    statistics: TrainOnlyStatistics,
    cells: torch.Tensor,
    volume: torch.Tensor,
    num_nodes: int,
    *,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only the MLP under BF16 autocast, then project in FP32."""

    features = invariant_features.float()
    mean = statistics.feature_mean.to(features.device, dtype=torch.float32)
    std = statistics.feature_std.to(features.device, dtype=torch.float32)
    normalized = (features - mean) / std.clamp_min(1.0e-8)
    enabled = features.device.type == "cuda"
    with torch.autocast(
        device_type=features.device.type,
        dtype=amp_dtype if enabled else torch.bfloat16,
        enabled=enabled,
    ):
        dimensionless = model(normalized)
    cell_stress = dimensionless.float() * float(statistics.stress_rms)
    nodal_stress = volume_weighted_nodal_projection(
        cell_stress, cells, volume, num_nodes
    )
    return cell_stress, nodal_stress


def even_indices(size: int, count: int) -> list[int]:
    """Deterministic endpoint-inclusive even selection."""

    if size < 1 or count < 1:
        raise ValueError("size and count must be positive")
    count = min(size, count)
    return np.unique(np.rint(np.linspace(0, size - 1, count)).astype(int)).tolist()


def teacher_frame_indices(num_steps: int, count: int) -> list[int]:
    """Match CHP teacher-stress sampling: evenly spaced frames 1..T-1."""

    if num_steps < 2:
        raise ValueError("a trajectory must contain at least two frames")
    return [index + 1 for index in even_indices(num_steps - 1, count)]


def validate_control_config(cfg: Mapping[str, Any]) -> None:
    """Fail closed on CPU execution, test leakage, or protocol drift."""

    if int(cfg.get("schema_version", 0)) != CONTROL_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONTROL_SCHEMA_VERSION}")
    requested = str(get_cfg(dict(cfg), "training.device", "")).lower()
    if not requested.startswith("cuda"):
        raise ValueError("constitutive identifiability control is CUDA-only")
    if not bool(get_cfg(dict(cfg), "training.amp", False)):
        raise ValueError("training.amp must be enabled")
    if str(get_cfg(dict(cfg), "training.amp_dtype", "")).lower() not in {
        "bf16",
        "bfloat16",
    }:
        raise ValueError("training.amp_dtype must be bfloat16")
    train_split = str(get_cfg(dict(cfg), "data.train_split", ""))
    val_split = str(get_cfg(dict(cfg), "data.val_split", ""))
    test_split = str(get_cfg(dict(cfg), "data.test_split", ""))
    if not train_split or not val_split or not test_split:
        raise ValueError("explicit train, val, and test split names are required")
    if len({train_split, val_split, test_split}) != 3:
        raise ValueError("train, validation, and test split names must be distinct")
    if (train_split, val_split, test_split) != ("train", "val", "test"):
        raise ValueError(
            "deforming_plate control requires literal train/val/test split names"
        )
    if int(get_cfg(dict(cfg), "validation.cases", -1)) != 20:
        raise ValueError("the diagnostic requires fixed even-val20")
    if int(get_cfg(dict(cfg), "validation.frames", -1)) != 16:
        raise ValueError("the diagnostic requires exactly 16 teacher frames")
    if str(get_cfg(dict(cfg), "validation.case_selection", "")) != "even":
        raise ValueError("validation.case_selection must be even")
    if str(get_cfg(dict(cfg), "training.checkpoint_selection", "")) != "fixed_final_epoch":
        raise ValueError("validation-selected checkpoints are forbidden")
    if int(get_cfg(dict(cfg), "model.input_dim", -1)) != 3:
        raise ValueError("the diagnostic input is exactly three objective invariants")
    if str(get_cfg(dict(cfg), "model.output", "")) != "nonnegative_cell_scalar":
        raise ValueError("the diagnostic output must be a nonnegative cell scalar")
    positive_integers = {
        "training.cases": get_cfg(dict(cfg), "training.cases", 0),
        "training.frames": get_cfg(dict(cfg), "training.frames", 0),
        "training.epochs": get_cfg(dict(cfg), "training.epochs", 0),
        "model.hidden_dim": get_cfg(dict(cfg), "model.hidden_dim", 0),
        "model.hidden_layers": get_cfg(dict(cfg), "model.hidden_layers", 0),
    }
    invalid = [key for key, value in positive_integers.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"positive protocol values required: {invalid}")
    if not get_cfg(dict(cfg), "training.output_dir", None):
        raise ValueError("training.output_dir is required")
    finite_positive = {
        "training.lr": get_cfg(dict(cfg), "training.lr", None),
        "training.grad_clip_norm": get_cfg(
            dict(cfg), "training.grad_clip_norm", None
        ),
        "training.minimum_j": get_cfg(dict(cfg), "training.minimum_j", None),
        "training.maximum_i2_bar": get_cfg(
            dict(cfg), "training.maximum_i2_bar", None
        ),
        "loss.huber_delta": get_cfg(dict(cfg), "loss.huber_delta", None),
    }
    invalid_positive = [
        key
        for key, value in finite_positive.items()
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0
    ]
    if invalid_positive:
        raise ValueError(f"finite positive protocol values required: {invalid_positive}")
    nonnegative = {
        "training.min_lr": get_cfg(dict(cfg), "training.min_lr", None),
        "training.weight_decay": get_cfg(dict(cfg), "training.weight_decay", None),
        "loss.physical_mse_weight": get_cfg(
            dict(cfg), "loss.physical_mse_weight", None
        ),
        "loss.peak_mse_weight": get_cfg(dict(cfg), "loss.peak_mse_weight", None),
    }
    invalid_nonnegative = [
        key
        for key, value in nonnegative.items()
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0
    ]
    if invalid_nonnegative:
        raise ValueError(
            f"finite non-negative protocol values required: {invalid_nonnegative}"
        )
    if float(nonnegative["training.min_lr"]) > float(
        finite_positive["training.lr"]
    ):
        raise ValueError("training.min_lr cannot exceed training.lr")


def development_case_dirs(
    cfg: Mapping[str, Any],
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    """Resolve train and validation directories without opening a test case."""

    validate_control_config(cfg)
    root = Path(get_cfg(dict(cfg), "data.root")).resolve()
    split_path = Path(get_cfg(dict(cfg), "data.split_file")).resolve()
    with split_path.open("r", encoding="utf-8") as stream:
        splits = json.load(stream)
    if not isinstance(splits, Mapping):
        raise ValueError("split file must contain a mapping")
    train_name = str(get_cfg(dict(cfg), "data.train_split"))
    val_name = str(get_cfg(dict(cfg), "data.val_split"))
    test_name = str(get_cfg(dict(cfg), "data.test_split"))
    train_ids = [str(value) for value in splits.get(train_name, [])]
    val_ids = [str(value) for value in splits.get(val_name, [])]
    test_ids = [str(value) for value in splits.get(test_name, [])]
    if not train_ids or not val_ids or not test_ids:
        raise ValueError("train, validation, and test splits must be non-empty")
    for split_name, identifiers in (
        (train_name, train_ids),
        (val_name, val_ids),
        (test_name, test_ids),
    ):
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{split_name} split contains duplicate case ids")
        unsafe = []
        for case_id in identifiers:
            candidate = Path(case_id)
            if (
                not case_id
                or candidate.is_absolute()
                or len(candidate.parts) != 1
                or candidate.name != case_id
                or case_id in {".", ".."}
            ):
                unsafe.append(case_id)
        if unsafe:
            raise ValueError(
                f"{split_name} split contains unsafe case ids: {unsafe[:3]}"
            )
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train, validation, and test case ids must be disjoint")
    train_count = int(get_cfg(dict(cfg), "training.cases", 256))
    val_count = int(get_cfg(dict(cfg), "validation.cases", 20))
    if len(train_ids) < train_count:
        raise ValueError(
            f"training split has {len(train_ids)} cases, fewer than required {train_count}"
        )
    if len(val_ids) < val_count:
        raise ValueError(
            f"validation split has {len(val_ids)} cases, fewer than required {val_count}"
        )
    physical_paths = {
        name: [(root / case_id).resolve() for case_id in identifiers]
        for name, identifiers in (
            (train_name, train_ids),
            (val_name, val_ids),
            (test_name, test_ids),
        )
    }
    for name, paths in physical_paths.items():
        escaped = [path for path in paths if path.parent != root]
        if escaped:
            raise ValueError(f"{name} split resolves outside the data root")
        if len(paths) != len(set(paths)):
            raise ValueError(f"{name} split resolves duplicate physical case paths")
    if (
        set(physical_paths[train_name]) & set(physical_paths[val_name])
        or set(physical_paths[train_name]) & set(physical_paths[test_name])
        or set(physical_paths[val_name]) & set(physical_paths[test_name])
    ):
        raise ValueError("train, validation, and test physical paths must be disjoint")
    selected_train = [
        train_ids[index] for index in even_indices(len(train_ids), train_count)
    ]
    selected_val = [
        val_ids[index] for index in even_indices(len(val_ids), val_count)
    ]
    train_dirs = [(root / case_id).resolve() for case_id in selected_train]
    val_dirs = [(root / case_id).resolve() for case_id in selected_val]
    missing = [path for path in (*train_dirs, *val_dirs) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"selected development case is missing: {missing[0]}")
    audit = {
        "train_split": train_name,
        "validation_split": val_name,
        "test_split_name_only": test_name,
        "selected_train_case_ids": selected_train,
        "selected_validation_case_ids": selected_val,
        "test_cases_loaded": 0,
    }
    return train_dirs, val_dirs, audit


def require_cuda_bf16(cfg: Mapping[str, Any]) -> torch.device:
    """Return the configured CUDA device or reject fallback execution."""

    validate_control_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the constitutive control")
    requested = torch.device(str(get_cfg(dict(cfg), "training.device", "cuda")))
    device = torch.device("cuda", requested.index or 0)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device must support BF16")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return device


def run_control_experiment(
    cfg: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Path]:
    """Train on train only, freeze the final epoch, then evaluate val20 once."""

    validate_control_config(cfg)
    config_file = Path(config_path).resolve()
    on_disk_cfg = load_config(config_file)
    runtime_config_sha256 = _canonical_sha256(cfg)
    if _canonical_sha256(on_disk_cfg) != runtime_config_sha256:
        raise RuntimeError(
            "runtime diagnostic config differs from the frozen config file"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    device = require_cuda_bf16(cfg)
    torch.cuda.reset_peak_memory_stats(device)
    train_dirs, val_dirs, split_audit = development_case_dirs(cfg)
    split_audit["selected_train_frame_indices"] = teacher_frame_indices(
        400, int(get_cfg(cfg, "training.frames", 8))
    )
    split_audit["selected_validation_frame_indices"] = teacher_frame_indices(
        400, int(get_cfg(cfg, "validation.frames", 16))
    )
    output_dir = Path(get_cfg(cfg, "training.output_dir")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output_dir / "fixed_final_epoch.pt"
    provenance_path = output_dir / "provenance.json"
    metrics_path = output_dir / "metrics.val20.json"
    statistics_path = output_dir / "train_only_statistics.json"
    history_path = output_dir / "history.train_only.json"
    train_manifest_path = output_dir / "train_inputs.before_training.sha256.json"
    val_manifest_path = output_dir / "validation_inputs.after_freeze.sha256.json"
    source_manifest_path = output_dir / "source_files.sha256.json"
    formal_artifacts = (
        final_checkpoint,
        provenance_path,
        metrics_path,
        statistics_path,
        history_path,
        train_manifest_path,
        val_manifest_path,
        source_manifest_path,
    )
    if any(path.exists() for path in formal_artifacts):
        raise RuntimeError("formal diagnostic artifacts already exist; refusing overwrite")

    split_file = Path(get_cfg(cfg, "data.split_file")).resolve()
    frozen_split_sha256 = _sha256(split_file)
    # Only train arrays are opened before the fixed final-epoch checkpoint is
    # frozen. Validation arrays are first fingerprinted after that freeze.
    train_manifest = _development_input_manifest(train_dirs, ())
    source_manifest = _source_file_manifest((config_file, split_file))
    _write_json(train_manifest_path, train_manifest)
    _write_json(source_manifest_path, source_manifest)

    statistics = fit_train_only_statistics(train_dirs, cfg, device)
    _write_json(statistics_path, statistics.json_dict())
    initial_mean = statistics.stress_mean / max(statistics.stress_rms, 1.0e-12)
    model = CellScalarStressMLP(
        hidden_dim=int(get_cfg(cfg, "model.hidden_dim", 64)),
        hidden_layers=int(get_cfg(cfg, "model.hidden_layers", 2)),
        initial_dimensionless_mean=initial_mean,
    ).to(device)
    history = train_fixed_epochs(model, train_dirs, statistics, cfg, device)
    _write_json(history_path, history)

    checkpoint = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        "checkpoint_selection": "fixed_final_epoch_before_validation",
        "training_device": "cuda",
        "network_precision": "bfloat16_autocast",
        "mechanics_precision": "float32",
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "epoch": int(get_cfg(cfg, "training.epochs")),
        "model": model.state_dict(),
        "statistics": statistics.state_dict(),
        "config": cfg,
        "runtime_config_sha256": runtime_config_sha256,
        "split_file_sha256": frozen_split_sha256,
        "source_manifest_sha256": source_manifest["aggregate_sha256"],
        "train_input_manifest_sha256": train_manifest["aggregate_sha256"],
    }
    torch.save(checkpoint, final_checkpoint)
    checkpoint_sha = _sha256(final_checkpoint)
    frozen_checkpoint = _torch_load(final_checkpoint, device)
    model.load_state_dict(frozen_checkpoint["model"])
    frozen_statistics = TrainOnlyStatistics.from_state_dict(
        frozen_checkpoint["statistics"]
    )
    if frozen_statistics.json_dict() != statistics.json_dict():
        raise RuntimeError("frozen checkpoint statistics differ from train fit")

    # This is the first operation that opens validation arrays. The checkpoint
    # hash above is already immutable and cannot be selected on these labels.
    val_manifest = _development_input_manifest((), val_dirs)
    _write_json(val_manifest_path, val_manifest)
    posthoc_train_dirs = [
        train_dirs[index] for index in even_indices(len(train_dirs), 20)
    ]
    posthoc_train = evaluate_val20_once(
        model, posthoc_train_dirs, frozen_statistics, cfg, device
    )
    posthoc_train["evaluation_split"] = "train_posthoc"
    validation = evaluate_val20_once(
        model, val_dirs, frozen_statistics, cfg, device
    )
    validation["posthoc_train20"] = posthoc_train
    validation.update(
        {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "diagnostic_type": DIAGNOSTIC_TYPE,
            "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_selected_by": "fixed_final_epoch_before_validation",
            "evaluation_split": split_audit["validation_split"],
            "case_selection": "even",
            "requested_cases": 20,
            "requested_frames_per_case": 16,
            "test_cases_loaded": 0,
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
    )
    _write_json(metrics_path, validation)
    _verify_manifest_files_unchanged(train_manifest)
    _verify_manifest_files_unchanged(val_manifest)
    _verify_manifest_files_unchanged(source_manifest)

    repository_root = Path(__file__).resolve().parents[1]
    provenance = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        "created_at_utc": _utc_now(),
        "seed": seed,
        "configuration": {
            "path": str(config_file),
            "sha256": _sha256(config_file),
        },
        "split_manifest": {
            "path": str(split_file),
            "sha256": _sha256(split_file),
        },
        "development_split_audit": split_audit,
        "checkpoint": {
            "path": str(final_checkpoint),
            "sha256": checkpoint_sha,
            "selection": "fixed_final_epoch_before_validation",
        },
        "validation_metrics": {
            "path": str(metrics_path),
            "sha256": _sha256(metrics_path),
        },
        "train_input_manifest_before_training": {
            "path": str(train_manifest_path),
            "sha256": _sha256(train_manifest_path),
            "verified_unchanged_after_run": True,
            "test_case_arrays_hashed": 0,
        },
        "validation_input_manifest_after_checkpoint_freeze": {
            "path": str(val_manifest_path),
            "sha256": _sha256(val_manifest_path),
            "verified_unchanged_after_run": True,
            "checkpoint_sha256_before_manifest": checkpoint_sha,
            "test_case_arrays_hashed": 0,
        },
        "source_file_manifest": {
            "path": str(source_manifest_path),
            "sha256": _sha256(source_manifest_path),
            "verified_unchanged_after_run": True,
        },
        "train_only_artifacts": {
            "statistics": {
                "path": str(statistics_path),
                "sha256": _sha256(statistics_path),
            },
            "history": {
                "path": str(history_path),
                "sha256": _sha256(history_path),
            },
        },
        "precision": {
            "network": "CUDA BF16 autocast with FP32 master parameters",
            "deformation_invariants": "float32",
            "volume_weighted_projection": "float32",
        },
        "model_contract": {
            "inputs": ["I1_bar_minus_3", "I2_bar_minus_3", "J_minus_1"],
            "output": "nonnegative_cell_scalar_stress",
            "nodal_projection": "CHP reference-volume weighted average",
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        },
        "reproducibility": {
            "fixed_python_numpy_torch_seed": True,
            "fixed_even_case_and_frame_selection": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32_enabled": False,
            "bitwise_caveat": (
                "CHP-equivalent CUDA index_add projection can vary at roundoff "
                "level across GPU, driver, or PyTorch versions"
            ),
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "source_revision": _git_revision(repository_root),
    }
    _write_json(provenance_path, provenance)
    return {
        "checkpoint": final_checkpoint,
        "metrics": metrics_path,
        "provenance": provenance_path,
    }


def fit_train_only_statistics(
    train_dirs: Sequence[Path],
    cfg: Mapping[str, Any],
    device: torch.device,
) -> TrainOnlyStatistics:
    """Fit invariant moments and target RMS without loading validation."""

    feature_sum = torch.zeros(3, dtype=torch.float64)
    feature_square = torch.zeros(3, dtype=torch.float64)
    stress_sum = 0.0
    stress_square = 0.0
    cell_count = 0
    nodal_count = 0
    requested = 0
    admissible = 0
    frame_count = int(get_cfg(dict(cfg), "training.frames", 8))
    for case_dir in train_dirs:
        case = _load_control_case(case_dir)
        frames = teacher_frame_indices(case.num_steps, frame_count)
        features, state, targets, mask = _case_frame_tensors(case, frames, device)
        valid = _admissible_frames(state, cfg)
        requested += len(frames)
        admissible += int(valid.sum().item())
        if not bool(valid.any().item()):
            continue
        selected = features[valid]
        feature_sum += selected.sum(dim=(0, 1)).double().cpu()
        feature_square += selected.square().sum(dim=(0, 1)).double().cpu()
        cell_count += int(selected.shape[0] * selected.shape[1])
        selected_target = targets[valid][:, mask, 0].float()
        stress_sum += float(selected_target.double().sum().cpu())
        stress_square += float(selected_target.double().square().sum().cpu())
        nodal_count += int(selected_target.numel())
    if cell_count <= 0 or nodal_count <= 0 or stress_square <= 0.0:
        raise RuntimeError("train-only statistics contain no admissible stress data")
    mean = feature_sum / cell_count
    variance = feature_square / cell_count - mean.square()
    std = variance.clamp_min(1.0e-12).sqrt()
    return TrainOnlyStatistics(
        feature_mean=mean.float(),
        feature_std=std.float(),
        stress_rms=math.sqrt(stress_square / nodal_count),
        stress_mean=stress_sum / nodal_count,
        cell_samples=cell_count,
        nodal_samples=nodal_count,
        requested_frames=requested,
        admissible_frames=admissible,
    )


def train_fixed_epochs(
    model: CellScalarStressMLP,
    train_dirs: Sequence[Path],
    statistics: TrainOnlyStatistics,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run a fixed train-only schedule; validation is never consulted."""

    epochs = int(get_cfg(dict(cfg), "training.epochs", 20))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get_cfg(dict(cfg), "training.lr", 1.0e-3)),
        weight_decay=float(get_cfg(dict(cfg), "training.weight_decay", 1.0e-6)),
        fused=bool(get_cfg(dict(cfg), "training.fused_optimizer", True)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(get_cfg(dict(cfg), "training.min_lr", 1.0e-5)),
    )
    frame_count = int(get_cfg(dict(cfg), "training.frames", 8))
    seed = int(dict(cfg).get("seed", 42))
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        generator = np.random.default_rng(seed + epoch * 1_000_003)
        order = generator.permutation(len(train_dirs))
        totals = {"loss": 0.0, "base": 0.0, "physical": 0.0, "peak": 0.0}
        used = 0
        model.train()
        for index in order:
            case = _load_control_case(train_dirs[int(index)])
            frames = teacher_frame_indices(case.num_steps, frame_count)
            features, state, targets, mask = _case_frame_tensors(case, frames, device)
            valid = _admissible_frames(state, cfg)
            if not bool(valid.any().item()):
                continue
            optimizer.zero_grad(set_to_none=True)
            _, prediction = predict_nodal_scalar_stress(
                model,
                features[valid],
                statistics,
                torch.as_tensor(
                    np.array(case.cells, copy=True), device=device, dtype=torch.long
                ),
                torch.as_tensor(
                    np.array(case.reference_volume, copy=True),
                    device=device,
                    dtype=torch.float32,
                ).reshape(-1),
                case.num_nodes,
            )
            loss, parts = _training_loss(
                prediction[:, mask, 0], targets[valid][:, mask, 0], statistics, cfg
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite train loss in {case.case_id}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(get_cfg(dict(cfg), "training.grad_clip_norm", 5.0)),
            )
            if not bool(torch.isfinite(gradient).item()):
                raise RuntimeError(f"non-finite train gradient in {case.case_id}")
            optimizer.step()
            for key, value in parts.items():
                totals[key] += float(value.detach().cpu())
            used += 1
        scheduler.step()
        if used == 0:
            raise RuntimeError("an epoch contained no admissible train cases")
        row = {
            "epoch": epoch,
            **{key: value / used for key, value in totals.items()},
            "optimizer_steps": used,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
            "validation_used": False,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['loss']:.6g} "
            f"physical={row['physical']:.6g} time={row['seconds']:.1f}s",
            flush=True,
        )
    return {
        "checkpoint_selection": "fixed_final_epoch",
        "validation_used_for_training_or_selection": False,
        "epochs": history,
    }


@torch.no_grad()
def evaluate_val20_once(
    model: CellScalarStressMLP,
    val_dirs: Sequence[Path],
    statistics: TrainOnlyStatistics,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate all non-prescribed nodes on fixed even-val20 x 16 once."""

    if len(val_dirs) != 20:
        raise ValueError("formal diagnostic evaluation requires exactly val20")
    model.eval()
    total = StressErrorSums()
    per_case: list[dict[str, Any]] = []
    requested = 0
    admissible = 0
    skipped: list[dict[str, Any]] = []
    zero_coverage_cases: list[str] = []
    frame_count = int(get_cfg(dict(cfg), "validation.frames", 16))
    for case_dir in val_dirs:
        case = _load_control_case(case_dir)
        frames = teacher_frame_indices(case.num_steps, frame_count)
        features, state, targets, mask = _case_frame_tensors(case, frames, device)
        valid = _admissible_frames(state, cfg)
        case_sums = StressErrorSums()
        requested += len(frames)
        admissible += int(valid.sum().item())
        valid_prediction: torch.Tensor | None = None
        valid_cursor = 0
        if bool(valid.any().item()):
            _, valid_prediction = predict_nodal_scalar_stress(
                model,
                features[valid],
                statistics,
                torch.as_tensor(
                    np.array(case.cells, copy=True), device=device, dtype=torch.long
                ),
                torch.as_tensor(
                    np.array(case.reference_volume, copy=True),
                    device=device,
                    dtype=torch.float32,
                ).reshape(-1),
                case.num_nodes,
            )
        for local_index, frame in enumerate(frames):
            if not bool(valid[local_index].item()):
                skipped.append(
                    {
                        "case_id": case.case_id,
                        "frame": int(frame),
                        "minimum_j": float(state.j[local_index].min().cpu()),
                        "maximum_i2_bar": float(
                            state.i2_bar[local_index].max().cpu()
                        ),
                    }
                )
                continue
            if valid_prediction is None:
                raise RuntimeError("admissible frame has no prediction batch")
            case_sums.update(
                valid_prediction[valid_cursor, mask, 0],
                targets[local_index, mask, 0],
            )
            valid_cursor += 1
        total = total + case_sums
        if case_sums.count:
            case_metrics: dict[str, Any] = case_sums.metrics()
        else:
            zero_coverage_cases.append(case.case_id)
            case_metrics = {
                "teacher_stress_relative_rmse": None,
                "teacher_stress_p95_relative_rmse": None,
                "teacher_stress_pooled_relative_rmse": None,
                "teacher_stress_per_frame_p95_relative_rmse": None,
                "teacher_stress_rmse": None,
                "evaluated_nodes": 0.0,
                "p95_nodes": 0.0,
                "target_nonzero_fraction": None,
            }
        per_case.append(
            {
                "case_id": case.case_id,
                "requested_frames": len(frames),
                "evaluated_frames": len(frames)
                - sum(item["case_id"] == case.case_id for item in skipped),
                **case_metrics,
            }
        )
    if total.count == 0:
        raise RuntimeError("val20 contains no admissible teacher-stress frames")
    return {
        "summary": {
            **total.metrics(),
            "requested_frames": float(requested),
            "admissible_frames": float(admissible),
            "admissible_coverage": admissible / max(requested, 1),
        },
        "per_case": per_case,
        "skipped_frames": skipped,
        "zero_admissible_case_ids": zero_coverage_cases,
        "metric_definition": {
            "mask": "~prescribed_mask",
            "aggregation": "pooled physical-unit squared error",
            "p95": "teacher exact-geometry: per-frame truth top 5%, then pooled",
            "not_comparable_to": (
                "trajectory-global P95 rollout metrics without explicit conversion"
            ),
        },
    }


def _case_frame_tensors(
    case: ControlCase,
    frames: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, DeformationInvariants, torch.Tensor, torch.Tensor]:
    reference = torch.as_tensor(
        np.array(case.nodes, copy=True), device=device, dtype=torch.float32
    )
    displacement = torch.as_tensor(
        np.array(case.displacement[list(frames)], copy=True),
        device=device,
        dtype=torch.float32,
    )
    cells = torch.as_tensor(
        np.array(case.cells, copy=True), device=device, dtype=torch.long
    )
    dm_inv = torch.as_tensor(
        np.array(case.dm_inv, copy=True), device=device, dtype=torch.float32
    )
    features, state = objective_invariant_features(
        reference[None] + displacement, cells, dm_inv
    )
    targets = torch.as_tensor(
        np.array(case.stress[list(frames), :, :1], copy=True),
        device=device,
        dtype=torch.float32,
    )
    mask = ~torch.as_tensor(
        np.array(case.prescribed_mask, copy=True),
        device=device,
        dtype=torch.bool,
    )
    if not bool(mask.any().item()):
        mask = torch.ones(case.num_nodes, device=device, dtype=torch.bool)
    if not bool(
        torch.isfinite(reference).all()
        and torch.isfinite(displacement).all()
        and torch.isfinite(targets).all()
        and torch.isfinite(features).all()
    ):
        raise ValueError(f"{case.case_id}: selected diagnostic tensors must be finite")
    return features, state, targets, mask


def _load_control_case(case_dir: str | Path) -> ControlCase:
    """Map only the seven arrays used by this diagnostic, never test outputs."""

    root = Path(case_dir)

    def required(name: str) -> np.ndarray:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{root}: missing required diagnostic array {name}")
        return np.load(path, allow_pickle=False, mmap_mode="r")

    nodes = required("nodes.npy")
    cells = required("cells.npy")
    dm_inv = required("Dm_inv.npy")
    volume = required("reference_volume.npy")
    displacement = required("U.npy")
    stress = required("S.npy")
    prescribed = required("prescribed_mask.npy")
    num_nodes = int(nodes.shape[0])
    num_cells = int(cells.shape[0])
    if nodes.shape != (num_nodes, 3):
        raise ValueError(f"{root}: nodes.npy must have shape [N,3]")
    if cells.shape != (num_cells, 4):
        raise ValueError(f"{root}: cells.npy must have shape [M,4]")
    if dm_inv.shape != (num_cells, 3, 3):
        raise ValueError(f"{root}: Dm_inv.npy must have shape [M,3,3]")
    if volume.shape not in {(num_cells,), (num_cells, 1)}:
        raise ValueError(f"{root}: reference_volume.npy must have M values")
    if displacement.shape != (400, num_nodes, 3):
        raise ValueError(f"{root}: control requires U.npy shape [400,N,3]")
    if stress.shape != (400, num_nodes, 1):
        raise ValueError(f"{root}: control requires S.npy shape [400,N,1]")
    if prescribed.shape not in {(num_nodes,), (num_nodes, 1)}:
        raise ValueError(f"{root}: prescribed_mask.npy must have N values")
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= num_nodes):
        raise ValueError(f"{root}: cells.npy contains an invalid node index")
    for name, values in (
        ("nodes.npy", nodes),
        ("Dm_inv.npy", dm_inv),
        ("reference_volume.npy", volume),
    ):
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"{root}: {name} must contain only finite values")
    if not bool((np.asarray(volume) > 0.0).all()):
        raise ValueError(f"{root}: reference_volume.npy must be strictly positive")
    return ControlCase(
        case_id=root.name,
        root=root,
        nodes=nodes,
        cells=cells,
        dm_inv=dm_inv,
        reference_volume=volume,
        displacement=displacement,
        stress=stress,
        prescribed_mask=np.asarray(prescribed).reshape(-1),
    )


def _admissible_frames(
    state: DeformationInvariants, cfg: Mapping[str, Any]
) -> torch.Tensor:
    minimum_j = float(get_cfg(dict(cfg), "training.minimum_j", 0.01))
    maximum_i2 = float(get_cfg(dict(cfg), "training.maximum_i2_bar", 1.0e5))
    finite = torch.isfinite(state.j).all(dim=-1) & torch.isfinite(
        state.i2_bar
    ).all(dim=-1)
    return finite & (state.j.amin(dim=-1) >= minimum_j) & (
        state.i2_bar.amax(dim=-1) <= maximum_i2
    )


def _training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    statistics: TrainOnlyStatistics,
    cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = prediction.new_tensor(statistics.stress_rms).clamp_min(1.0e-12)
    transformed_prediction = torch.asinh(prediction / scale)
    transformed_target = torch.asinh(target / scale)
    base = functional.smooth_l1_loss(
        transformed_prediction,
        transformed_target,
        beta=float(get_cfg(dict(cfg), "loss.huber_delta", 0.1)),
    )
    physical = ((prediction - target) / scale).square().mean()
    peak_losses: list[torch.Tensor] = []
    for frame_prediction, frame_target in zip(prediction, target):
        threshold = torch.quantile(frame_target.abs(), 0.90)
        peak = frame_target.abs() >= threshold
        peak_losses.append(
            ((frame_prediction[peak] - frame_target[peak]) / scale).square().mean()
        )
    peak_loss = torch.stack(peak_losses).mean()
    total = (
        base
        + float(get_cfg(dict(cfg), "loss.physical_mse_weight", 1.0)) * physical
        + float(get_cfg(dict(cfg), "loss.peak_mse_weight", 0.25)) * peak_loss
    )
    return total, {
        "loss": total,
        "base": base,
        "physical": physical,
        "peak": peak_loss,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _development_input_manifest(
    train_dirs: Sequence[Path], val_dirs: Sequence[Path]
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for split, case_dirs in (("train", train_dirs), ("val", val_dirs)):
        for case_dir in case_dirs:
            arrays = [
                {"name": name, **_file_fingerprint(Path(case_dir) / name)}
                for name in CONTROL_INPUT_ARRAYS
            ]
            cases.append(
                {"split": split, "case_id": Path(case_dir).name, "arrays": arrays}
            )
    return {
        "schema_version": 1,
        "hash_scope": "complete files for selected train and validation cases",
        "test_case_arrays_hashed": 0,
        "cases": cases,
        "aggregate_sha256": _canonical_sha256(cases),
    }


def _source_file_manifest(
    frozen_protocol_files: Sequence[str | Path] = (),
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    candidates = {
        Path(__file__).resolve(),
        Path(project_cell_to_nodes.__code__.co_filename).resolve(),
        (repository_root / "scripts" / "run_constitutive_identifiability.py").resolve(),
        *(Path(path).resolve() for path in frozen_protocol_files),
    }
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"diagnostic source file is missing: {missing[0]}")
    files = [_file_fingerprint(path) for path in sorted(candidates)]
    return {
        "schema_version": 1,
        "files": files,
        "aggregate_sha256": _canonical_sha256(files),
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"cannot fingerprint missing file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }


def _verify_manifest_files_unchanged(manifest: Mapping[str, Any]) -> None:
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
        raise RuntimeError(f"invalid frozen checkpoint: {path}")
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
    "CONTROL_SCHEMA_VERSION",
    "DIAGNOSTIC_TYPE",
    "SCIENTIFIC_DISCLAIMER",
    "CellScalarStressMLP",
    "StressErrorSums",
    "TrainOnlyStatistics",
    "development_case_dirs",
    "even_indices",
    "objective_invariant_features",
    "predict_nodal_scalar_stress",
    "run_control_experiment",
    "teacher_frame_indices",
    "validate_control_config",
    "volume_weighted_nodal_projection",
]
