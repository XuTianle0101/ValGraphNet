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

    def to(self, device: torch.device | str) -> "CHPNormalizers":
        return CHPNormalizers(
            self.displacement_scale.to(device),
            self.velocity_scale.to(device),
            self.stress.to(device),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "displacement_scale": self.displacement_scale.cpu(),
            "velocity_scale": self.velocity_scale.cpu(),
            "stress": self.stress.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CHPNormalizers":
        return cls(
            displacement_scale=state["displacement_scale"].float(),
            velocity_scale=state["velocity_scale"].float(),
            stress=AsinhStressTransform.from_state_dict(state["stress"]),
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
    stress_samples: list[torch.Tensor] = []
    for case_index in case_indices:
        case = cases[int(case_index)]
        frame_count = min(max(int(frames_per_case), 1), case.num_steps - 1)
        frames = np.linspace(0, case.num_steps - 2, frame_count).round().astype(int)
        node_count = min(max(int(nodes_per_frame), 1), case.num_nodes)
        nodes = np.linspace(0, case.num_nodes - 1, node_count).round().astype(int)
        for frame in frames:
            delta = case.displacement[frame + 1, nodes] - case.displacement[frame, nodes]
            delta_samples.append(torch.from_numpy(np.array(delta, copy=True)).float())
            velocity_samples.append(
                torch.from_numpy(np.array(case.velocity[frame + 1, nodes], copy=True)).float()
            )
            if case.stress_dim:
                stress_samples.append(
                    torch.from_numpy(
                        np.array(case.stress[frame + 1, nodes, :1], copy=True)
                    ).float()
                )
    if not stress_samples:
        raise ValueError("CHP-GNS requires at least one nodal stress label channel")
    delta_values = torch.cat(delta_samples, dim=0)
    velocity_values = torch.cat(velocity_samples, dim=0)
    displacement_scale = delta_values.square().mean(0).sqrt().clamp_min(1.0e-6)
    velocity_scale = velocity_values.square().mean(0).sqrt().clamp_min(1.0e-6)
    stress = AsinhStressTransform.fit(stress_samples)
    return CHPNormalizers(displacement_scale, velocity_scale, stress)


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


@torch.no_grad()
def geometry_safe_state_noise(
    static: CHPStatic,
    position: torch.Tensor,
    standard_deviation: float,
    *,
    smoothing_steps: int = 4,
    minimum_j: float = 0.2,
    max_backtracks: int = 8,
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
    scale = 1.0
    for _ in range(max(int(max_backtracks), 0) + 1):
        candidate = position + scale * noise
        j = invariants(
            deformation_gradient(candidate, static.cells, static.dm_inv)
        ).j
        if bool((j.min() >= float(minimum_j)).item()):
            return scale * noise, float(scale)
        scale *= 0.5
    return torch.zeros_like(position), 0.0


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
    )


def chp_step_loss(
    output: PhysicalStep,
    trajectory: CHPDeviceCase,
    next_step: int,
    normalizers: CHPNormalizers,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dimensionless state, stress, work, contact, residual, and J loss."""

    static = trajectory.static
    exact_position = static.reference_position + trajectory.displacement[next_step]
    exact_velocity = trajectory.velocity[next_step]
    moving = ~(static.fixed_mask | static.prescribed_mask)
    if not bool(moving.any().item()):
        moving = ~static.fixed_mask
    predicted_position = output.next_position[moving]
    predicted_velocity = output.next_velocity[moving]
    target_position = exact_position[moving]
    target_velocity = exact_velocity[moving]
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
    stress_target = trajectory.stress[next_step, stress_mask, :1]
    stress_prediction = output.nodal_stress[stress_mask, :1]
    transformed_prediction = normalizers.stress.transform(stress_prediction)
    transformed_target = normalizers.stress.transform(stress_target)
    stress_loss, stress_parts = robust_stress_loss(
        transformed_prediction,
        transformed_target,
        peak_fraction=float(get_cfg(cfg, "loss.peak_fraction", 0.1)),
        peak_weight=float(get_cfg(cfg, "loss.peak_weight", 0.5)),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )

    diagnostics = output.energy_diagnostics
    work_scale = (
        diagnostics["kinetic"].abs()
        + diagnostics["kinetic_after"].abs()
        + diagnostics["potential"].abs()
        + diagnostics["potential_after"].abs()
        + diagnostics["external_work"].abs()
        + diagnostics["boundary_work"].abs()
    ).detach().clamp_min(1.0e-8)
    work_loss = F.huber_loss(
        diagnostics["work_energy_balance"] / work_scale,
        torch.zeros_like(diagnostics["work_energy_balance"]),
        delta=1.0,
    )
    penetration_loss = diagnostics["max_penetration"].square()
    residual_loss = output.residual_force.square().mean()
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
        "work_energy": work_loss.detach(),
        "penetration": penetration_loss.detach(),
        "residual": residual_loss.detach(),
        "negative_j": negative_j_loss.detach(),
    }


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
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "rollout" in payload:
            payload = payload["rollout"]
        inline = payload
    if not inline:
        return None
    missing = [key for key in ROLLOUT_METRIC_KEYS if key not in inline]
    if missing:
        raise ValueError(f"native checkpoint reference is missing: {missing}")
    return {key: float(inline[key]) for key in ROLLOUT_METRIC_KEYS}


def run_chp_training(cfg: dict[str, Any]) -> Path:
    """Train CHP-GNS using BF16 neural blocks and FP32 mechanics on CUDA."""

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    device = _require_cuda(cfg)
    output_dir = Path(get_cfg(cfg, "training.output_dir", "outputs/chp_gns"))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset = _build_chp_datasets(cfg)
    material_dim = max(
        case.material_features.shape[1]
        for case in train_dataset.cases + (val_dataset.cases if val_dataset else [])
    )

    normalizer_path = output_dir / "normalizers.pt"
    if normalizer_path.exists():
        normalizers = CHPNormalizers.from_state_dict(
            _torch_load(normalizer_path, "cpu")
        )
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
    optimizer_kwargs: dict[str, Any] = {
        "lr": float(get_cfg(cfg, "training.lr", 1.0e-4)),
        "weight_decay": float(get_cfg(cfg, "training.weight_decay", 1.0e-6)),
    }
    if bool(get_cfg(cfg, "training.fused_optimizer", True)):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
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
    if resume is not None:
        checkpoint = _torch_load(resume, device)
        if int(checkpoint.get("schema_version", 0)) != CHPGNS.checkpoint_schema_version:
            raise ValueError(f"unsupported CHP checkpoint schema: {resume}")
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
        rollout_metrics = (
            evaluate_chp_rollouts(
                model, val_dataset.cases, val_cache, cfg, amp_dtype=amp_dtype
            )
            if val_dataset is not None and val_cache is not None
            else {key: float("inf") for key in ROLLOUT_METRIC_KEYS}
        )
        teacher_metrics = (
            evaluate_teacher_forced_stress(
                model, val_dataset.cases, val_cache, cfg, amp_dtype=amp_dtype
            )
            if val_dataset is not None and val_cache is not None
            else {"teacher_stress_relative_rmse": float("inf")}
        )
        rollout_metrics.update(teacher_metrics)
        score = minimax_checkpoint_score(rollout_metrics, native_reference)
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
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            normalizers,
            cfg,
            epoch,
            score,
            min(best_score, score),
            rollout_metrics,
            material_dim,
        )
        torch.save(payload, output_dir / "latest.pt")
        if score < best_score:
            best_score = score
            payload["best_score"] = best_score
            torch.save(payload, output_dir / "best.pt")
        print(
            f"epoch={epoch:02d} K={horizon:02d} loss={train_metrics['loss']:.5g} "
            f"score={score:.4g} peak={peak_gib:.2f}GiB "
            f"time={record['seconds']:.1f}s"
        )
        gate_epoch = _teacher_gate_epoch(stages)
        teacher_relative = float(teacher_metrics["teacher_stress_relative_rmse"])
        teacher_threshold = float(
            get_cfg(cfg, "validation.teacher_stress_threshold", 0.50)
        )
        if (
            bool(get_cfg(cfg, "validation.enforce_teacher_stress_gate", True))
            and epoch == gate_epoch
            and teacher_relative >= teacher_threshold
        ):
            failure = {
                "epoch": epoch,
                "teacher_stress_relative_rmse": teacher_relative,
                "threshold": teacher_threshold,
                "action": "stop rollout curriculum and add full tensor labels",
            }
            _save_json(output_dir / "teacher_stress_gate_failure.json", failure)
            raise RuntimeError(
                "teacher-forced stress gate failed: "
                f"{teacher_relative:.4g} >= {teacher_threshold:.4g}"
            )
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
        category_counts[category] = category_counts.get(category, 0) + 1
        trajectory = cache.get_slice(case_index, start, start + horizon + 1)
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
                        output, trajectory, step + 1, normalizers, cfg
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
        "u_error": torch.zeros((), device=device),
        "u_reference": torch.zeros((), device=device),
        "final_error": torch.zeros((), device=device),
        "final_reference": torch.zeros((), device=device),
        "stress_error": torch.zeros((), device=device),
        "stress_reference": torch.zeros((), device=device),
        "p95_error": torch.zeros((), device=device),
        "p95_reference": torch.zeros((), device=device),
        "penetration": torch.zeros((), device=device),
        "momentum": torch.zeros((), device=device),
        "energy": torch.zeros((), device=device),
    }
    evaluated_steps = 0
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
            if (
                not bool(torch.isfinite(state.position).all().item())
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
            accum["u_error"] += u_error.square().sum()
            accum["u_reference"] += u_reference.square().sum()
            stress_mask = ~trajectory.static.prescribed_mask
            if not bool(stress_mask.any().item()):
                stress_mask = torch.ones_like(trajectory.static.prescribed_mask)
            target_stress = trajectory.stress[step + 1, stress_mask, :1]
            stress_error = output.nodal_stress[stress_mask, :1] - target_stress
            accum["stress_error"] += stress_error.square().sum()
            accum["stress_reference"] += target_stress.square().sum()
            threshold = torch.quantile(target_stress.abs().reshape(-1), 0.95)
            peak = target_stress.abs() >= threshold
            accum["p95_error"] += stress_error[peak].square().sum()
            accum["p95_reference"] += target_stress[peak].square().sum()
            accum["penetration"] += output.energy_diagnostics["max_penetration"]
            resultant = output.internal_force + output.contact_force
            accum["momentum"] += torch.linalg.vector_norm(resultant.sum(0))
            accum["energy"] += output.energy_diagnostics["work_energy_balance"].abs()
            evaluated_steps += 1
        exact_final = (
            trajectory.static.reference_position
            + trajectory.displacement[min(steps, case.num_steps - 1)]
        )
        final_error = state.position[moving] - exact_final[moving]
        accum["final_error"] += final_error.square().sum()
        accum["final_reference"] += trajectory.displacement[
            min(steps, case.num_steps - 1), moving
        ].square().sum()
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
        "diverged_cases": float(diverged),
        "mean_penetration": float((accum["penetration"] / max(evaluated_steps, 1)).item()),
        "mean_momentum_residual": float((accum["momentum"] / max(evaluated_steps, 1)).item()),
        "mean_work_energy_error": float((accum["energy"] / max(evaluated_steps, 1)).item()),
        "evaluated_steps": float(evaluated_steps),
    }
    if diverged:
        divergence_floor = 1.0e6 * diverged / max(count, 1)
        for key in ROLLOUT_METRIC_KEYS:
            result[key] = max(float(result[key]), divergence_floor)
    return result


@torch.no_grad()
def evaluate_teacher_forced_stress(
    model: CHPGNS,
    cases: list[ValveCase],
    cache: CHPCaseCache,
    cfg: dict[str, Any],
    *,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Isolate constitutive error by evaluating the exact next geometry."""

    model.eval()
    case_count = min(
        int(get_cfg(cfg, "validation.teacher_stress_cases", 20)), len(cases)
    )
    frame_count = max(int(get_cfg(cfg, "validation.teacher_stress_frames", 16)), 1)
    case_indices = np.linspace(0, len(cases) - 1, case_count).round().astype(int)
    error = torch.zeros((), device=cache.device)
    reference = torch.zeros((), device=cache.device)
    peak_error = torch.zeros((), device=cache.device)
    peak_reference = torch.zeros((), device=cache.device)
    for case_index_value in tqdm(
        case_indices, desc="teacher-stress", leave=False
    ):
        case_index = int(case_index_value)
        case = cases[case_index]
        trajectory = cache.get(case_index)
        frames = np.linspace(1, case.num_steps - 1, min(frame_count, case.num_steps - 1))
        for frame_value in np.unique(frames.round().astype(int)):
            frame = int(frame_value)
            exact_position = (
                trajectory.static.reference_position + trajectory.displacement[frame]
            )
            with _autocast(cache.device, amp_dtype):
                prediction, _ = model.nodal_stress_at(
                    trajectory.static, exact_position
                )
            mask = ~trajectory.static.prescribed_mask
            if not bool(mask.any().item()):
                mask = torch.ones_like(mask)
            target = trajectory.stress[frame, mask, :1]
            residual = prediction[mask, :1] - target
            error += residual.square().sum()
            reference += target.square().sum()
            threshold = torch.quantile(target.abs().reshape(-1), 0.95)
            peak = target.abs() >= threshold
            peak_error += residual[peak].square().sum()
            peak_reference += target[peak].square().sum()
    eps = torch.tensor(1.0e-12, device=cache.device)
    return {
        "teacher_stress_relative_rmse": float(
            torch.sqrt(error / reference.clamp_min(eps)).item()
        ),
        "teacher_stress_p95_relative_rmse": float(
            torch.sqrt(peak_error / peak_reference.clamp_min(eps)).item()
        ),
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
    if int(checkpoint.get("schema_version", 0)) != CHPGNS.checkpoint_schema_version:
        raise ValueError("legacy checkpoints cannot populate the CHP physical decoder")
    model = CHPGNS(cfg, material_dim=int(checkpoint.get("material_dim", 0))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizers = CHPNormalizers.from_state_dict(checkpoint["normalizers"]).to(device)
    return model, normalizers, checkpoint


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
) -> dict[str, Any]:
    return {
        "schema_version": CHPGNS.checkpoint_schema_version,
        "architecture": "CHP-GNS",
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
