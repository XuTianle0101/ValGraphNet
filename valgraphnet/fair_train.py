"""CUDA-only training and rollout evaluation for the fair deforming-plate MGN.

The baseline intentionally predicts only a normalized position increment and an
``asinh``-transformed scalar stress.  Velocity and acceleration are always
derived from the position update, so the baseline cannot improve a metric by
emitting mutually inconsistent kinematic channels.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from valgraphnet.config import get_cfg
from valgraphnet.fair_baseline import (
    FairDeformingPlateBaseline,
    add_position_noise,
    integrate_position_delta,
)
from valgraphnet.normalization import Normalizers, fit_normalizers
from valgraphnet.physical_evaluation import validate_reference_protocol
from valgraphnet.stress_transform import AsinhStressTransform, robust_stress_loss
from valgraphnet.train import build_datasets


FAIR_CHECKPOINT_SCHEMA_VERSION = 2
FAIR_MODEL_FAMILY = "fair_deforming_plate_mgn"
ROLLOUT_METRIC_KEYS = (
    "moving_displacement_relative_rmse",
    "final_displacement_relative_rmse",
    "stress_relative_rmse",
    "stress_p95_relative_rmse",
)


def require_cuda_bf16(cfg: dict[str, Any]) -> torch.device:
    """Reject CPU/fallback execution so reported timing is GPU-only."""

    requested = str(get_cfg(cfg, "training.device", "cuda")).lower()
    if not requested.startswith("cuda"):
        raise ValueError("the fair deforming_plate baseline is CUDA-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fair baseline training and inference")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")
    if not bool(get_cfg(cfg, "training.amp", True)):
        raise ValueError("training.amp must remain enabled for the fair baseline")
    dtype = str(get_cfg(cfg, "training.amp_dtype", "bfloat16")).lower()
    if dtype not in {"bfloat16", "bf16"}:
        raise ValueError("training.amp_dtype must be bfloat16/bf16")
    requested_device = torch.device(requested)
    device = torch.device("cuda", requested_device.index or 0)
    torch.cuda.set_device(device.index)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device


def one_start_per_trajectory(
    cases: Iterable[Any], *, epoch: int, seed: int
) -> list[tuple[int, int]]:
    """Select exactly one uniformly distributed valid start for every case."""

    rng = np.random.default_rng(int(seed) + 1_000_003 * int(epoch))
    starts = [
        (case_index, int(rng.integers(0, case.num_steps - 1)))
        for case_index, case in enumerate(cases)
    ]
    rng.shuffle(starts)
    return starts


def fit_fair_statistics(
    dataset,
    cfg: dict[str, Any],
) -> tuple[Normalizers, AsinhStressTransform]:
    """Fit input/delta statistics and a bounded deterministic stress sample."""

    max_graphs = int(get_cfg(cfg, "training.normalizer_graphs", 128))
    normalizers = fit_normalizers(dataset, max_samples=max_graphs)
    stress_transform = AsinhStressTransform.fit(
        _sample_stress_tensors(dataset.cases, cfg),
        max_values=int(get_cfg(cfg, "training.stress_normalizer_values", 1_000_000)),
    )
    return normalizers, stress_transform


def _sample_stress_tensors(cases: list[Any], cfg: dict[str, Any]):
    case_count = min(len(cases), int(get_cfg(cfg, "training.normalizer_cases", 128)))
    frame_count = int(get_cfg(cfg, "training.normalizer_frames", 8))
    node_count = int(get_cfg(cfg, "training.normalizer_nodes", 256))
    case_indices = _even_indices(len(cases), case_count)
    for case_index in case_indices:
        case = cases[case_index]
        if case.stress_dim < 1:
            raise ValueError(f"{case.root}: fair baseline requires a scalar stress label")
        frames = _even_indices(case.num_steps, min(case.num_steps, frame_count))
        nodes = _even_indices(case.num_nodes, min(case.num_nodes, node_count))
        values = np.asarray(case.stress[np.ix_(frames, nodes, [0])], dtype=np.float32)
        yield torch.from_numpy(np.array(values, copy=True))


def _even_indices(size: int, count: int) -> list[int]:
    if size <= 0 or count <= 0:
        return []
    if count >= size:
        return list(range(size))
    return np.linspace(0, size - 1, num=count).round().astype(np.int64).tolist()


def fair_one_step_loss(
    prediction: dict[str, torch.Tensor],
    corrected_delta: torch.Tensor,
    target_stress: torch.Tensor,
    moving_mask: torch.Tensor,
    delta_scale: torch.Tensor,
    stress_transform: AsinhStressTransform,
    cfg: dict[str, Any],
    stress_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise only normalized ``delta_x`` and transformed stress."""

    scale = delta_scale.to(corrected_delta.device, corrected_delta.dtype).clamp_min(1.0e-8)
    target_delta_normalized = corrected_delta / scale
    delta_loss = F.huber_loss(
        prediction["delta_x"][moving_mask],
        target_delta_normalized[moving_mask],
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    encoded_stress = stress_transform.transform(target_stress)
    if stress_mask is None:
        stress_mask = moving_mask
    stress_loss, stress_parts = robust_stress_loss(
        prediction["stress_transformed"][stress_mask],
        encoded_stress[stress_mask],
        ranking_target=target_stress[stress_mask],
        peak_fraction=float(get_cfg(cfg, "loss.peak_fraction", 0.1)),
        peak_weight=float(get_cfg(cfg, "loss.peak_weight", 0.5)),
        delta=float(get_cfg(cfg, "loss.huber_delta", 1.0)),
    )
    total = (
        float(get_cfg(cfg, "loss.position", 1.0)) * delta_loss
        + float(get_cfg(cfg, "loss.stress", 1.0)) * stress_loss
    )
    with torch.no_grad():
        physical_delta = prediction["delta_x"].float() * scale.float()
        delta_rmse = torch.sqrt(
            ((physical_delta[moving_mask] - corrected_delta[moving_mask].float()) ** 2).mean()
        )
    return total, {
        "total": total.detach().float(),
        "delta": delta_loss.detach().float(),
        "stress": stress_loss.detach().float(),
        "stress_base": stress_parts["stress_base"].float(),
        "stress_peak": stress_parts["stress_peak"].float(),
        "delta_rmse": delta_rmse,
    }


def train_fair_epoch(
    model: FairDeformingPlateBaseline,
    dataset,
    optimizer: torch.optim.Optimizer,
    normalizers: Normalizers,
    stress_transform: AsinhStressTransform,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    seed = int(cfg.get("seed", 42))
    starts = one_start_per_trajectory(dataset.cases, epoch=epoch, seed=seed)
    noise_std = float(get_cfg(cfg, "data.noise_std", 0.003))
    if not math.isclose(noise_std, 0.003, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("fair native state noise must be exactly 0.003")
    delta_scale = normalizers.target_scale[:3].to(device)
    gpu_stress_transform = stress_transform.to(device)
    noise_generator = torch.Generator(device=device).manual_seed(seed + epoch * 97_409)
    clip_norm = float(get_cfg(cfg, "training.grad_clip_norm", 1.0))

    for case_index, step in tqdm(starts, desc=f"fair-train-{epoch:03d}", leave=False):
        case = dataset.cases[case_index]
        tensors = dataset.gpu_builder.case_tensors(case, device)
        exact_position = tensors["nodes"] + tensors["U"][step]
        next_position = tensors["nodes"] + tensors["U"][step + 1]
        moving = ~(tensors["fixed"] | tensors["prescribed"])
        if not np.any(~(case.fixed_mask | case.prescribed_mask)):
            moving = ~tensors["fixed"]
        noisy_position, corrected_delta, _ = add_position_noise(
            exact_position,
            next_position,
            moving,
            noise_std,
            generator=noise_generator,
        )
        state = {
            "U": noisy_position - tensors["nodes"],
            "V": tensors["V"][step],
            "A": tensors["A"][step],
        }
        graph = dataset.make_graph_gpu(case_index, step, device, state=state)
        target_stress = tensors["S"][step + 1, :, :1]
        stress_mask = ~tensors["prescribed"]
        if not np.any(~case.prescribed_mask):
            stress_mask = torch.ones_like(tensors["prescribed"])

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(graph)
            loss, metrics = fair_one_step_loss(
                prediction,
                corrected_delta,
                target_stress,
                moving,
                delta_scale,
                gpu_stress_transform,
                cfg,
                stress_mask=stress_mask,
            )
        loss.backward()
        if clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        _accumulate(totals, metrics)
        count += 1
    return _average(totals, count)


@torch.inference_mode()
def evaluate_fair_rollouts(
    model: FairDeformingPlateBaseline,
    dataset,
    normalizers: Normalizers,
    stress_transform: AsinhStressTransform,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Run closed-loop GPU trajectories and report all four selection metrics."""

    if device.type != "cuda":
        raise ValueError("fair rollout evaluation is CUDA-only")
    model.eval()
    case_count = min(len(dataset.cases), int(get_cfg(cfg, "validation.cases", 20)))
    case_indices = _even_indices(len(dataset.cases), case_count)
    requested_steps = get_cfg(cfg, "validation.steps", None)
    delta_scale = normalizers.target_scale[:3].to(device)
    gpu_stress_transform = stress_transform.to(device)
    divergence_position = float(get_cfg(cfg, "validation.divergence_position", 10.0))

    accum = {
        "u_error": torch.zeros((), device=device),
        "u_reference": torch.zeros((), device=device),
        "u_count": torch.zeros((), device=device),
        "final_error": torch.zeros((), device=device),
        "final_reference": torch.zeros((), device=device),
        "final_count": torch.zeros((), device=device),
        "stress_error": torch.zeros((), device=device),
        "stress_reference": torch.zeros((), device=device),
        "stress_count": torch.zeros((), device=device),
        "p95_error": torch.zeros((), device=device),
        "p95_reference": torch.zeros((), device=device),
        "p95_count": torch.zeros((), device=device),
    }
    divergent_cases = 0

    for case_index in tqdm(case_indices, desc="fair-rollout-val", leave=False):
        case = dataset.cases[case_index]
        tensors = dataset.gpu_builder.case_tensors(case, device)
        state = dataset.gpu_builder.state(case, 0, device)
        steps = case.num_steps - 1
        if requested_steps is not None:
            steps = min(steps, int(requested_steps))
        moving = ~(tensors["fixed"] | tensors["prescribed"])
        if not np.any(~(case.fixed_mask | case.prescribed_mask)):
            moving = ~tensors["fixed"]
        stress_mask = ~tensors["prescribed"]
        if not np.any(~case.prescribed_mask):
            stress_mask = torch.ones_like(tensors["prescribed"])
        stress_history = tensors["S"][1 : steps + 1, stress_mask, :1].abs().reshape(-1)
        p95_threshold = torch.quantile(stress_history, 0.95) if stress_history.numel() else None

        case_invalid = torch.zeros((), dtype=torch.bool, device=device)
        for step in range(steps):
            graph = dataset.make_graph_gpu(case_index, step, device, state=state)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(graph)
            delta_position = prediction["delta_x"].float() * delta_scale
            stress = gpu_stress_transform.inverse(
                prediction["stress_transformed"].float()
            )
            current_position = tensors["nodes"] + state["U"]
            integrated = integrate_position_delta(
                current_position,
                state["V"],
                delta_position,
                tensors["times"][step + 1] - tensors["times"][step],
                fixed_mask=tensors["fixed"],
                prescribed_mask=tensors["prescribed"],
                prescribed_position=tensors["nodes"] + tensors["U"][step + 1],
                prescribed_velocity=tensors["V"][step + 1],
            )
            candidate_state = {
                "U": integrated.next_position - tensors["nodes"],
                "V": integrated.next_velocity,
                "A": integrated.acceleration,
            }
            for value in candidate_state.values():
                case_invalid |= ~torch.isfinite(value).all()
            case_invalid |= ~torch.isfinite(stress).all()
            case_invalid |= candidate_state["U"].abs().max() > divergence_position
            # Once a trajectory is invalid, retain the failure flag but keep
            # subsequent CUDA graph construction numerically defined.  This
            # avoids a device synchronization on every one of the 399 steps.
            safe_limit = 2.0 * divergence_position
            state = {
                "U": torch.nan_to_num(
                    candidate_state["U"], nan=0.0, posinf=safe_limit, neginf=-safe_limit
                ).clamp(-safe_limit, safe_limit),
                "V": torch.nan_to_num(candidate_state["V"]),
                "A": torch.nan_to_num(candidate_state["A"]),
            }
            safe_stress = torch.nan_to_num(
                stress, nan=0.0, posinf=torch.finfo(torch.float32).max, neginf=-torch.finfo(torch.float32).max
            )

            truth_u = tensors["U"][step + 1, moving]
            residual_u = state["U"][moving] - truth_u
            accum["u_error"] += residual_u.square().sum()
            accum["u_reference"] += truth_u.square().sum()
            accum["u_count"] += residual_u.numel()

            truth_stress = tensors["S"][step + 1, stress_mask, :1]
            residual_stress = safe_stress[stress_mask] - truth_stress
            accum["stress_error"] += residual_stress.square().sum()
            accum["stress_reference"] += truth_stress.square().sum()
            accum["stress_count"] += residual_stress.numel()
            if p95_threshold is not None:
                peak = truth_stress.abs() >= p95_threshold
                accum["p95_error"] += residual_stress[peak].square().sum()
                accum["p95_reference"] += truth_stress[peak].square().sum()
                accum["p95_count"] += peak.sum()

        case_diverged = bool(case_invalid.item())
        if case_diverged:
            divergent_cases += 1
        else:
            truth_final = tensors["U"][steps, moving]
            final_residual = state["U"][moving] - truth_final
            accum["final_error"] += final_residual.square().sum()
            accum["final_reference"] += truth_final.square().sum()
            accum["final_count"] += final_residual.numel()

    if divergent_cases:
        return {
            **{key: float("inf") for key in ROLLOUT_METRIC_KEYS},
            "moving_displacement_rmse": float("inf"),
            "final_displacement_rmse": float("inf"),
            "stress_rmse": float("inf"),
            "stress_p95_rmse": float("inf"),
            "divergent_cases": float(divergent_cases),
            "cases": float(case_count),
        }

    eps = torch.tensor(1.0e-20, device=device)

    def rmse(error: str, count: str) -> torch.Tensor:
        return torch.sqrt(accum[error] / accum[count].clamp_min(1.0))

    def relative(error: str, reference: str) -> torch.Tensor:
        return torch.sqrt(accum[error] / accum[reference].clamp_min(eps))

    metrics = {
        "moving_displacement_rmse": rmse("u_error", "u_count"),
        "moving_displacement_relative_rmse": relative("u_error", "u_reference"),
        "final_displacement_rmse": rmse("final_error", "final_count"),
        "final_displacement_relative_rmse": relative("final_error", "final_reference"),
        "stress_rmse": rmse("stress_error", "stress_count"),
        "stress_relative_rmse": relative("stress_error", "stress_reference"),
        "stress_p95_rmse": rmse("p95_error", "p95_count"),
        "stress_p95_relative_rmse": relative("p95_error", "p95_reference"),
    }
    return {
        **{key: float(value.item()) for key, value in metrics.items()},
        "divergent_cases": 0.0,
        "cases": float(case_count),
    }


def minimax_checkpoint_score(
    metrics: dict[str, float], native_reference: dict[str, float]
) -> float:
    """Worst native-normalized rollout metric; no objective can be traded away."""

    ratios: list[float] = []
    for key in ROLLOUT_METRIC_KEYS:
        if key not in metrics or key not in native_reference:
            raise KeyError(f"missing minimax metric: {key}")
        value = float(metrics[key])
        reference = float(native_reference[key])
        if reference < 0.0 or not math.isfinite(reference):
            raise ValueError(f"native reference {key} must be finite and non-negative")
        if reference == 0.0:
            ratios.append(0.0 if value == 0.0 else float("inf"))
        else:
            ratios.append(value / reference)
    return max(ratios)


def load_native_reference(cfg: dict[str, Any]) -> dict[str, float]:
    """Load canonical native rollout metrics from inline data or JSON."""

    payload = get_cfg(cfg, "validation.native_reference", None)
    path = get_cfg(cfg, "validation.native_reference_file", None)
    if payload is None and path:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    if payload is None:
        if bool(get_cfg(cfg, "validation.require_native_reference", False)):
            raise ValueError("validation.native_reference(_file) is required")
        payload = {key: 1.0 for key in ROLLOUT_METRIC_KEYS}
    source_payload = payload
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
    for container in ("summary", "aggregate", "rollout"):
        if container in payload and isinstance(payload[container], dict):
            payload = payload[container]
            break
    aliases = {
        "moving_displacement_relative_rmse": ("displacement_relative_rmse",),
        "final_displacement_relative_rmse": ("final_relative_rmse",),
        "stress_p95_relative_rmse": ("p95_stress_relative_rmse",),
    }
    result: dict[str, float] = {}
    for key in ROLLOUT_METRIC_KEYS:
        value = payload.get(key)
        for alias in aliases.get(key, ()):
            if value is None:
                value = payload.get(alias)
        if value is None:
            raise KeyError(f"native reference is missing {key}")
        result[key] = float(value)
    return result


def run_fair_training(cfg: dict[str, Any]) -> Path:
    """Train the corrected fair MGN and return its minimax-best checkpoint."""

    device = require_cuda_bf16(cfg)
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    train_dataset, val_dataset = build_datasets(cfg)
    if val_dataset is None:
        raise ValueError("fair rollout checkpointing requires a validation split")
    normalizers, stress_transform = fit_fair_statistics(train_dataset, cfg)
    train_dataset.normalizers = normalizers
    val_dataset.normalizers = normalizers

    model = FairDeformingPlateBaseline(cfg).to(device)
    fused = bool(get_cfg(cfg, "training.fused_optimizer", True))
    optimizer_kwargs = {
        "lr": float(get_cfg(cfg, "training.lr", 1.0e-4)),
        "weight_decay": float(get_cfg(cfg, "training.weight_decay", 0.0)),
    }
    if fused:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    epochs = int(get_cfg(cfg, "training.epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(get_cfg(cfg, "training.min_lr", 1.0e-6)),
    )
    native_reference = load_native_reference(cfg)
    output_dir = Path(get_cfg(cfg, "training.output_dir", "outputs/deforming_plate_fair"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    history_path = output_dir / "history.json"
    start_epoch = 1
    best_score = float("inf")
    history: list[dict[str, Any]] = []

    resume = get_cfg(cfg, "training.resume_from", None)
    resume_path = latest_path if str(resume).lower() == "auto" else Path(resume) if resume else None
    if resume_path is not None and resume_path.exists():
        checkpoint = _torch_load(resume_path, device)
        _validate_fair_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        normalizers = Normalizers.from_state_dict(checkpoint["normalizers"])
        stress_transform = AsinhStressTransform.from_state_dict(checkpoint["stress_transform"])
        train_dataset.normalizers = normalizers
        val_dataset.normalizers = normalizers
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", checkpoint["score"]))
        history = _load_history(history_path)

    validation_every = int(get_cfg(cfg, "validation.every", 1))
    save_every = int(get_cfg(cfg, "training.save_every", 1))
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_fair_epoch(
            model,
            train_dataset,
            optimizer,
            normalizers,
            stress_transform,
            device,
            cfg,
            epoch,
        )
        scheduler.step()
        rollout_metrics: dict[str, float] | None = None
        score = float("inf")
        if _fair_validation_due(epoch, epochs, validation_every):
            rollout_metrics = evaluate_fair_rollouts(
                model,
                val_dataset,
                normalizers,
                stress_transform,
                device,
                cfg,
            )
            score = minimax_checkpoint_score(rollout_metrics, native_reference)
            rollout_metrics["minimax_score"] = score
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "rollout": rollout_metrics,
            "score": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - epoch_start,
        }
        history = [item for item in history if int(item["epoch"]) != epoch]
        history.append(row)
        _save_history(history_path, history)
        print(_format_epoch(row), flush=True)

        if rollout_metrics is not None and math.isfinite(score) and score < best_score:
            best_score = score
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    normalizers,
                    stress_transform,
                    native_reference,
                    rollout_metrics or {},
                    epoch,
                    score,
                    best_score,
                ),
                best_path,
            )
        torch.save(
            _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                cfg,
                normalizers,
                stress_transform,
                native_reference,
                rollout_metrics or {},
                epoch,
                score,
                best_score,
            ),
            latest_path,
        )
        if save_every > 0 and epoch % save_every == 0:
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    normalizers,
                    stress_transform,
                    native_reference,
                    rollout_metrics or {},
                    epoch,
                    score,
                    best_score,
                ),
                output_dir / f"epoch_{epoch:04d}.pt",
            )
    if not best_path.exists() or not math.isfinite(best_score):
        raise RuntimeError(
            "fair baseline produced no finite rollout-validated checkpoint"
        )
    return best_path


def _fair_validation_due(epoch: int, total_epochs: int, every: int) -> bool:
    """Validate on cadence and always audit the final trained epoch."""

    return int(every) > 0 and (
        int(epoch) == int(total_epochs) or int(epoch) % int(every) == 0
    )


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    cfg: dict[str, Any],
    normalizers: Normalizers,
    stress_transform: AsinhStressTransform,
    native_reference: dict[str, float],
    rollout_metrics: dict[str, float],
    epoch: int,
    score: float,
    best_score: float,
) -> dict[str, Any]:
    return {
        "schema_version": FAIR_CHECKPOINT_SCHEMA_VERSION,
        "model_family": FAIR_MODEL_FAMILY,
        "output_contract": "normalized_delta_x_plus_asinh_stress",
        "state_integration": "velocity_and_acceleration_derived_from_delta_x",
        "training_device": "cuda",
        "training_precision": "bfloat16",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": cfg,
        "normalizers": normalizers.state_dict(),
        "stress_transform": stress_transform.state_dict(),
        "native_reference": native_reference,
        "rollout_metrics": rollout_metrics,
        "checkpoint_metric": "four_metric_native_ratio_minimax",
        "epoch": int(epoch),
        "score": float(score),
        "best_score": float(best_score),
    }


def load_fair_checkpoint(
    cfg: dict[str, Any], path: str | Path
) -> tuple[FairDeformingPlateBaseline, Normalizers, AsinhStressTransform, dict[str, Any]]:
    """Load a fair baseline checkpoint directly onto CUDA for inference."""

    device = require_cuda_bf16(cfg)
    checkpoint = _torch_load(path, device)
    _validate_fair_checkpoint(checkpoint)
    model = FairDeformingPlateBaseline(checkpoint.get("cfg", cfg)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizers = Normalizers.from_state_dict(checkpoint["normalizers"])
    stress_transform = AsinhStressTransform.from_state_dict(checkpoint["stress_transform"])
    return model, normalizers, stress_transform, checkpoint


def _validate_fair_checkpoint(checkpoint: dict[str, Any]) -> None:
    if int(checkpoint.get("schema_version", 0)) != FAIR_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported fair baseline checkpoint schema")
    if checkpoint.get("model_family") != FAIR_MODEL_FAMILY:
        raise ValueError("checkpoint is not a corrected fair deforming_plate MGN")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _accumulate(total: dict[str, Any], values: dict[str, Any]) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + value


def _average(total: dict[str, Any], count: int) -> dict[str, float]:
    averaged = {}
    for key, value in total.items():
        value = value / max(count, 1)
        averaged[key] = float(value.item()) if torch.is_tensor(value) else float(value)
    return averaged


def _torch_load(path: str | Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return list(json.load(stream))


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(sorted(history, key=lambda item: int(item["epoch"])), stream, indent=2)


def _format_epoch(row: dict[str, Any]) -> str:
    train = ", ".join(f"{key}={value:.5g}" for key, value in sorted(row["train"].items()))
    rollout = row["rollout"]
    if rollout:
        selected = ", ".join(
            f"{key}={rollout[key]:.5g}"
            for key in (*ROLLOUT_METRIC_KEYS, "minimax_score")
        )
        return f"epoch {row['epoch']:04d} | train: {train} | rollout: {selected}"
    return f"epoch {row['epoch']:04d} | train: {train}"
