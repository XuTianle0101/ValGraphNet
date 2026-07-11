"""Training loop for ValGraphNet."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from valgraphnet.config import get_cfg
from valgraphnet.data import PerTrajectoryStepSampler, ValveGraphDataset, collate_valve_graphs
from valgraphnet.losses import valve_loss
from valgraphnet.gpu_graph import update_state
from valgraphnet.model import build_model
from valgraphnet.normalization import Normalizers, fit_normalizers, split_target


def run_training(cfg: dict[str, Any]) -> Path:
    """Run model training and return the best checkpoint path."""

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(get_cfg(cfg, "training.device", "auto")))
    output_dir = Path(get_cfg(cfg, "training.output_dir", "outputs/valve_hybrid"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "config_snapshot.json", cfg)

    train_dataset, val_dataset = build_datasets(cfg)
    resume_path = _resolve_resume_path(cfg, output_dir)
    normalizers = None
    if bool(get_cfg(cfg, "normalization.enabled", True)):
        stats_path = resume_path or get_cfg(cfg, "training.initial_checkpoint", None)
        stats_state = (
            _torch_load(Path(stats_path), map_location="cpu")
            if stats_path and Path(stats_path).exists()
            else None
        )
        if stats_state is not None and stats_state.get("normalizers") is not None:
            normalizers = Normalizers.from_state_dict(stats_state["normalizers"])
            print(f"reused normalization statistics from: {stats_path}")
        else:
            normalizers = fit_normalizers(
                train_dataset,
                max_samples=get_cfg(cfg, "data.max_train_samples_for_stats", 512),
                eps=float(get_cfg(cfg, "normalization.eps", 1.0e-8)),
            )
        train_dataset.normalizers = normalizers
        if val_dataset is not None:
            val_dataset.normalizers = normalizers

    model = build_model(cfg, output_dim=train_dataset.output_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get_cfg(cfg, "training.lr", 1.0e-4)),
        weight_decay=float(get_cfg(cfg, "training.weight_decay", 1.0e-6)),
        fused=device.type == "cuda",
    )
    amp_enabled, amp_dtype = amp_settings(cfg, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(
        f"device: {device} | amp: {amp_enabled}"
        f" ({str(amp_dtype).removeprefix('torch.') if amp_enabled else 'disabled'})"
    )

    gpu_graphs = bool(get_cfg(cfg, "training.gpu_graph_build", False))
    if gpu_graphs and device.type != "cuda":
        raise RuntimeError("training.gpu_graph_build requires training.device=cuda")

    train_sampler = _build_step_sampler(train_dataset, cfg, training=True)
    val_sampler = (
        _build_step_sampler(val_dataset, cfg, training=False)
        if val_dataset is not None
        else None
    )
    train_source = range(len(train_dataset)) if gpu_graphs else train_dataset
    train_loader = DataLoader(
        train_source,
        batch_size=int(get_cfg(cfg, "training.batch_size", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(get_cfg(cfg, "training.num_workers", 0)),
        collate_fn=_collate_index if gpu_graphs else collate_valve_graphs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            range(len(val_dataset)) if gpu_graphs else val_dataset,
            batch_size=int(get_cfg(cfg, "training.batch_size", 1)),
            shuffle=False,
            sampler=val_sampler,
            num_workers=int(get_cfg(cfg, "training.num_workers", 0)),
            collate_fn=_collate_index if gpu_graphs else collate_valve_graphs,
        )

    best_loss = float("inf")
    best_path = output_dir / "best.pt"
    epochs = int(get_cfg(cfg, "training.epochs", 100))
    save_every = int(get_cfg(cfg, "training.save_every", 10))
    save_latest = bool(get_cfg(cfg, "training.save_latest", False))
    start_epoch = 1
    history_path = output_dir / "history.json"
    history = _load_history(history_path)
    if resume_path is not None:
        checkpoint = _torch_load(resume_path, map_location=device)
        model.load_compatible_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("score", best_loss))
        if best_path.exists():
            best_loss = float(_torch_load(best_path, map_location="cpu").get("score", best_loss))
        print(f"resumed checkpoint: {resume_path} at epoch {start_epoch - 1}")
    else:
        initial_path = get_cfg(cfg, "training.initial_checkpoint", None)
        if initial_path:
            checkpoint = _torch_load(Path(initial_path), map_location=device)
            model.load_compatible_state_dict(checkpoint["model"])
            print(f"initialized compatible weights from: {initial_path}")
            stress_path = get_cfg(cfg, "training.stress_initial_checkpoint", None)
            if stress_path:
                stress_checkpoint = _torch_load(Path(stress_path), map_location=device)
                model.load_stress_decoder_state_dict(stress_checkpoint["model"])
                print(f"initialized stress decoder from: {stress_path}")

            if val_dataset is not None and str(
                get_cfg(cfg, "training.checkpoint_metric", "one_step")
            ).lower() == "rollout":
                initial_rollout = evaluate_rollout_metric(
                    model, val_dataset, device, cfg
                )
                best_loss = initial_rollout["score"]
                history = [item for item in history if int(item["epoch"]) != 0]
                history.append(
                    {
                        "epoch": 0,
                        "train": {},
                        "val": {},
                        "rollout_val": initial_rollout,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "seconds": 0.0,
                        "train_steps": 0,
                        "val_steps": 0,
                    }
                )
                _save_history(history_path, history)
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scaler,
                    cfg,
                    normalizers,
                    train_dataset.output_dim,
                    0,
                    best_loss,
                )
                print(
                    "initial rollout checkpoint: "
                    f"score={best_loss:.6g}, "
                    f"displacement_rmse={initial_rollout['displacement_rmse']:.6g}, "
                    f"stress_rmse={initial_rollout['stress_rmse']:.6g}"
                )

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg,
            dataset=train_dataset if gpu_graphs else None,
        )
        val_metrics = (
            evaluate(
                model, val_loader, device, cfg,
                dataset=val_dataset if gpu_graphs else None,
            )
            if val_loader is not None
            else None
        )
        checkpoint_metric = str(
            get_cfg(cfg, "training.checkpoint_metric", "one_step")
        ).lower()
        rollout_metrics = None
        if (
            val_dataset is not None
            and checkpoint_metric == "rollout"
            and _rollout_validation_due(epoch, epochs, cfg)
        ):
            rollout_metrics = evaluate_rollout_metric(
                model, val_dataset, device, cfg
            )
        if checkpoint_metric == "rollout":
            # Never compare an inexpensive one-step loss with rollout scores.
            # On skipped epochs only latest.pt advances; best.pt is unchanged.
            score = rollout_metrics["score"] if rollout_metrics is not None else None
        else:
            score = val_metrics["total"] if val_metrics else train_metrics["total"]

        print(_format_epoch(epoch, train_metrics, val_metrics, rollout_metrics))
        history = [item for item in history if int(item["epoch"]) != epoch]
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "rollout_val": rollout_metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.perf_counter() - epoch_start,
                "train_steps": len(train_loader),
                "val_steps": len(val_loader) if val_loader is not None else 0,
            }
        )
        _save_history(history_path, history)
        if score is not None and score < best_loss:
            best_loss = score
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                cfg,
                normalizers,
                train_dataset.output_dim,
                epoch,
                best_loss,
            )
        checkpoint_score = score if score is not None else best_loss
        if not np.isfinite(checkpoint_score):
            checkpoint_score = 1.0e30
        if save_latest:
            save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                scaler,
                cfg,
                normalizers,
                train_dataset.output_dim,
                epoch,
                checkpoint_score,
            )
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scaler,
                cfg,
                normalizers,
                train_dataset.output_dim,
                epoch,
                checkpoint_score,
            )

    return best_path


def _build_step_sampler(dataset, cfg: dict[str, Any], training: bool):
    key = (
        "training.steps_per_trajectory_per_epoch"
        if training
        else "training.validation_steps_per_trajectory"
    )
    steps = get_cfg(cfg, key, None)
    if steps is None:
        return None
    groups = dataset.trajectory_index_groups
    if training:
        rollout_steps = int(get_cfg(cfg, "training.rollout_steps", 1))
        if rollout_steps > 1:
            groups = [
                range(group.start, group.stop - rollout_steps + 1)
                for group in groups
                if len(group) >= rollout_steps
            ]
    return PerTrajectoryStepSampler(
        groups,
        steps_per_trajectory=int(steps),
        shuffle=training,
        seed=int(cfg.get("seed", 42)) + (0 if training else 1_000_000),
    )


def _collate_index(batch) -> int:
    if len(batch) != 1:
        raise ValueError("GPU graph construction currently requires batch_size=1")
    return int(batch[0])


def _resolve_resume_path(cfg: dict[str, Any], output_dir: Path) -> Path | None:
    resume_from = get_cfg(cfg, "training.resume_from", None)
    if not resume_from:
        return None
    path = output_dir / "latest.pt" if str(resume_from).lower() == "auto" else Path(resume_from)
    return path if path.exists() else None


def _torch_load(path: str | Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(json.load(f))


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted(history, key=lambda item: int(item["epoch"])), f, indent=2)


def build_datasets(cfg: dict[str, Any]) -> tuple[ValveGraphDataset, ValveGraphDataset | None]:
    data_root = get_cfg(cfg, "data.root", "data/processed")
    split_file = get_cfg(cfg, "data.split_file", None)
    split_path = Path(split_file) if split_file else None
    if split_path and split_path.exists():
        train_dataset = ValveGraphDataset(
            data_root=data_root,
            cfg=cfg,
            split=str(get_cfg(cfg, "data.train_split", "train")),
            split_file=split_path,
        )
        val_dataset = ValveGraphDataset(
            data_root=data_root,
            cfg=cfg,
            split=str(get_cfg(cfg, "data.val_split", "val")),
            split_file=split_path,
        )
        if val_dataset.output_dim != train_dataset.output_dim:
            raise ValueError("Train and validation output dimensions differ; check stress exports.")
        return train_dataset, val_dataset

    train_dataset = ValveGraphDataset(data_root=data_root, cfg=cfg)
    return train_dataset, None


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    cfg: dict[str, Any],
    dataset: ValveGraphDataset | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        if dataset is not None:
            loss, metrics = _multistep_gpu_loss(
                model, dataset, int(batch), device, cfg
            )
        else:
            batch = batch.to(device)
            with autocast_context(cfg, device):
                pred = model(batch)
                loss, metrics = valve_loss(pred, batch, cfg)
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        clip_norm = get_cfg(cfg, "training.grad_clip_norm", None)
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
        scaler.step(optimizer)
        scaler.update()
        _accumulate(totals, metrics)
        count += 1
    return _average(totals, count)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    cfg: dict[str, Any],
    dataset: ValveGraphDataset | None = None,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        if dataset is not None:
            case_index, step = dataset.samples[int(batch)]
            batch = dataset.make_graph_gpu(case_index, step, device)
        else:
            batch = batch.to(device)
        with autocast_context(cfg, device):
            pred = model(batch)
            _, metrics = valve_loss(pred, batch, cfg)
        _accumulate(totals, metrics)
        count += 1
    return _average(totals, count)


def _multistep_gpu_loss(
    model,
    dataset: ValveGraphDataset,
    sample_index: int,
    device: torch.device,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    case_index, start_step = dataset.samples[sample_index]
    case = dataset.cases[case_index]
    steps = min(
        int(get_cfg(cfg, "training.rollout_steps", 1)),
        case.num_steps - 1 - start_step,
    )
    discount = float(get_cfg(cfg, "training.rollout_loss_discount", 1.0))
    stress_steps = int(get_cfg(cfg, "training.rollout_stress_steps", steps))
    state = dataset.gpu_builder.state(case, start_step, device)
    case_tensors = dataset.gpu_builder.case_tensors(case, device)
    loss = torch.zeros((), device=device)
    totals: dict[str, float] = {}
    weight_sum = 0.0

    for offset in range(steps):
        step = start_step + offset
        graph = dataset.make_graph_gpu(case_index, step, device, state=state)
        weight = discount**offset
        with autocast_context(cfg, device):
            pred = model(graph)
            step_cfg = cfg
            if offset >= stress_steps:
                step_cfg = {
                    **cfg,
                    "loss": {**cfg.get("loss", {}), "stress": 0.0},
                }
            step_loss, metrics = valve_loss(pred, graph, step_cfg)
        loss = loss + weight * step_loss
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + weight * float(value)
        weight_sum += weight

        prediction = torch.cat(
            [pred["delta_u"], pred["delta_v"], pred["accel"], pred["stress"]],
            dim=1,
        )
        if dataset.normalizers is not None:
            prediction = prediction * graph.target_scale
        state = update_state(
            split_target(prediction), state, case_tensors, step + 1
        )

    loss = loss / max(weight_sum, 1.0e-12)
    metrics = {key: value / max(weight_sum, 1.0e-12) for key, value in totals.items()}
    metrics["rollout_steps"] = float(steps)
    return loss, metrics


@torch.no_grad()
def evaluate_rollout_metric(
    model,
    dataset: ValveGraphDataset,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Evaluate true autoregressive trajectories and return checkpoint score."""

    model.eval()
    case_count = min(
        int(get_cfg(cfg, "training.rollout_validation_cases", 5)),
        len(dataset.cases),
    )
    case_indices = (
        torch.linspace(0, len(dataset.cases) - 1, steps=case_count)
        .round()
        .long()
        .tolist()
    )
    requested_steps = get_cfg(cfg, "training.rollout_validation_steps", None)
    u_error = torch.zeros((), device=device)
    u_reference = torch.zeros((), device=device)
    u_count = torch.zeros((), device=device)
    stress_error = torch.zeros((), device=device)
    stress_reference = torch.zeros((), device=device)
    stress_count = torch.zeros((), device=device)
    final_error = torch.zeros((), device=device)
    final_count = torch.zeros((), device=device)
    max_evaluated_steps = 1

    for case_index in tqdm(case_indices, desc="rollout-val", leave=False):
        case = dataset.cases[case_index]
        tensors = dataset.gpu_builder.case_tensors(case, device)
        state = dataset.gpu_builder.state(case, 0, device)
        steps = case.num_steps - 1
        if requested_steps is not None:
            steps = min(steps, int(requested_steps))
        max_evaluated_steps = max(max_evaluated_steps, steps)
        free = ~(tensors["fixed"] | tensors["prescribed"])
        if not bool(free.any().item()):
            free = ~tensors["fixed"]

        for step in range(steps):
            graph = dataset.make_graph_gpu(case_index, step, device, state=state)
            with autocast_context(cfg, device):
                pred = model(graph)
            prediction = torch.cat(
                [pred["delta_u"], pred["delta_v"], pred["accel"], pred["stress"]],
                dim=1,
            )
            if dataset.normalizers is not None:
                prediction = prediction * graph.target_scale
            physical = split_target(prediction)
            state = update_state(physical, state, tensors, step + 1)

            u_residual = state["U"][free] - tensors["U"][step + 1, free]
            u_error += (u_residual * u_residual).sum()
            u_reference += (tensors["U"][step + 1, free] ** 2).sum()
            u_count += u_residual.numel()
            if physical["stress"].numel() > 0:
                truth_stress = tensors["S"][step + 1, free, : physical["stress"].shape[1]]
                stress_residual = physical["stress"][free] - truth_stress
                stress_error += (stress_residual * stress_residual).sum()
                stress_reference += (truth_stress * truth_stress).sum()
                stress_count += stress_residual.numel()
        final_residual = state["U"][free] - tensors["U"][steps, free]
        final_error += (final_residual * final_residual).sum()
        final_count += final_residual.numel()

    eps = torch.tensor(1.0e-12, device=device)
    displacement_rmse = torch.sqrt(u_error / u_count.clamp_min(1.0))
    displacement_relative = torch.sqrt(u_error / u_reference.clamp_min(eps))
    final_rmse = torch.sqrt(final_error / final_count.clamp_min(1.0))
    if bool((stress_count > 0).item()):
        stress_rmse = torch.sqrt(stress_error / stress_count)
        stress_relative = torch.sqrt(stress_error / stress_reference.clamp_min(eps))
    else:
        stress_rmse = torch.zeros((), device=device)
        stress_relative = torch.zeros((), device=device)
    stress_weight = float(
        get_cfg(cfg, "training.rollout_checkpoint_stress_weight", 0.1)
    )
    if dataset.normalizers is not None:
        target_scale = dataset.normalizers.target_scale.to(device)
        delta_scale = torch.sqrt((target_scale[:3] ** 2).mean())
        stress_scale = (
            torch.sqrt((target_scale[9:] ** 2).mean())
            if target_scale.numel() > 9
            else torch.ones((), device=device)
        )
    else:
        delta_scale = torch.ones((), device=device)
        stress_scale = torch.ones((), device=device)
    reference_u_rms = torch.sqrt(u_reference / u_count.clamp_min(1.0))
    rollout_scale = delta_scale * float(max_evaluated_steps) ** 0.5
    displacement_normalized = displacement_rmse / torch.maximum(
        reference_u_rms, rollout_scale.clamp_min(eps)
    )
    stress_normalized = stress_rmse / stress_scale.clamp_min(eps)
    score = displacement_normalized + stress_weight * stress_normalized
    return {
        "score": float(score.item()),
        "displacement_rmse": float(displacement_rmse.item()),
        "displacement_relative_rmse": float(displacement_relative.item()),
        "displacement_normalized_rmse": float(displacement_normalized.item()),
        "final_displacement_rmse": float(final_rmse.item()),
        "stress_rmse": float(stress_rmse.item()),
        "stress_relative_rmse": float(stress_relative.item()),
        "stress_normalized_rmse": float(stress_normalized.item()),
        "cases": float(case_count),
    }


def _rollout_validation_due(
    epoch: int,
    total_epochs: int,
    cfg: dict[str, Any],
) -> bool:
    """Return whether an expensive autoregressive checkpoint audit is due."""

    every = int(get_cfg(cfg, "training.rollout_validation_every", 1))
    if every <= 0:
        raise ValueError("training.rollout_validation_every must be positive")
    return int(epoch) == int(total_epochs) or int(epoch) % every == 0


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    cfg: dict[str, Any],
    normalizers,
    output_dim: int,
    epoch: int,
    score: float,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg,
        "normalizers": normalizers.state_dict() if normalizers is not None else None,
        "output_dim": int(output_dim),
        "epoch": int(epoch),
        "score": float(score),
    }
    torch.save(state, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def autocast_context(cfg: dict[str, Any], device: torch.device):
    """Return the configured CUDA autocast context."""

    enabled, dtype = amp_settings(cfg, device)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def amp_settings(
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[bool, torch.dtype]:
    enabled = bool(get_cfg(cfg, "training.amp", False)) and device.type == "cuda"
    dtype_name = str(get_cfg(cfg, "training.amp_dtype", "float16")).lower()
    dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtypes:
        raise ValueError("training.amp_dtype must be float16/fp16 or bfloat16/bf16")
    return enabled, dtypes[dtype_name]


def _accumulate(totals: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def _average(totals: dict[str, float], count: int) -> dict[str, float]:
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def _format_epoch(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    rollout_metrics: dict[str, float] | None = None,
) -> str:
    train_part = ", ".join(f"{k}={v:.5g}" for k, v in sorted(train_metrics.items()))
    rollout_part = ""
    if rollout_metrics:
        rollout_part = " | rollout: " + ", ".join(
            f"{k}={v:.5g}" for k, v in sorted(rollout_metrics.items())
        )
    if val_metrics:
        val_part = ", ".join(f"{k}={v:.5g}" for k, v in sorted(val_metrics.items()))
        return f"epoch {epoch:04d} | train: {train_part} | val: {val_part}{rollout_part}"
    return f"epoch {epoch:04d} | train: {train_part}{rollout_part}"


def _save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

