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

from valgraphnet.checkpoint_provenance import (
    atomic_torch_save,
    build_repo_data_contract,
    checkpoint_metadata,
    resume_config_sha256,
    strict_checkpoint_provenance,
    validate_repo_checkpoint,
)
from valgraphnet.config import get_cfg
from valgraphnet.data import PerTrajectoryStepSampler, ValveGraphDataset, collate_valve_graphs
from valgraphnet.losses import valve_loss
from valgraphnet.gpu_graph import update_state
from valgraphnet.model import build_model
from valgraphnet.normalization import Normalizers, fit_normalizers, split_target
from valgraphnet.physical_evaluation import validate_reference_protocol


ROLLOUT_METRIC_KEYS = (
    "moving_displacement_relative_rmse",
    "final_displacement_relative_rmse",
    "stress_relative_rmse",
    "stress_p95_relative_rmse",
)


def run_training(cfg: dict[str, Any]) -> Path:
    """Run model training and return the best checkpoint path."""

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(get_cfg(cfg, "training.device", "auto")))
    output_dir = Path(get_cfg(cfg, "training.output_dir", "outputs/valve_hybrid"))
    resume_path = _resolve_resume_path(cfg, output_dir)
    _validate_output_directory_for_resume(cfg, output_dir, resume_path)
    train_dataset, val_dataset = build_datasets(cfg)
    data_contract = (
        build_repo_data_contract(cfg)
        if strict_checkpoint_provenance(cfg)
        else None
    )
    resume_checkpoint_cpu = (
        _torch_load(resume_path, map_location="cpu")
        if resume_path is not None
        else None
    )
    if resume_checkpoint_cpu is not None:
        validate_repo_checkpoint(
            resume_checkpoint_cpu,
            cfg,
            data_contract,
            purpose="resume",
            source=resume_path,
        )
    initial_path = get_cfg(cfg, "training.initial_checkpoint", None)
    if resume_path is None and initial_path and not Path(initial_path).exists():
        raise FileNotFoundError(
            f"initial checkpoint does not exist: {Path(initial_path)}"
        )
    initial_checkpoint_cpu = (
        _torch_load(Path(initial_path), map_location="cpu")
        if resume_path is None and initial_path and Path(initial_path).exists()
        else None
    )
    if initial_checkpoint_cpu is not None:
        validate_repo_checkpoint(
            initial_checkpoint_cpu,
            cfg,
            data_contract,
            purpose="warm_start",
            source=initial_path,
        )
    stress_initial_path = get_cfg(
        cfg, "training.stress_initial_checkpoint", None
    )
    if resume_path is None and initial_path and stress_initial_path:
        if not Path(stress_initial_path).exists():
            raise FileNotFoundError(
                f"stress initial checkpoint does not exist: {stress_initial_path}"
            )
        stress_initial_checkpoint_cpu = _torch_load(
            Path(stress_initial_path), map_location="cpu"
        )
        validate_repo_checkpoint(
            stress_initial_checkpoint_cpu,
            cfg,
            data_contract,
            purpose="warm_start",
            source=stress_initial_path,
        )

    # In strict mode no run metadata is overwritten until resume provenance is
    # accepted.  A corrupt or 200-frame checkpoint therefore cannot taint the
    # formal full400 output directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "config_snapshot.json", cfg)
    if data_contract is not None:
        _save_json(output_dir / "data_contract.json", data_contract)

    normalizers = None
    if bool(get_cfg(cfg, "normalization.enabled", True)):
        # A strict warm-start migrates weights only.  It must not silently carry
        # normalizers from another data contract.
        stats_path = resume_path or initial_path
        stats_state = resume_checkpoint_cpu
        if stats_state is None and not strict_checkpoint_provenance(cfg):
            stats_state = initial_checkpoint_cpu
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
        validate_repo_checkpoint(
            checkpoint,
            cfg,
            data_contract,
            purpose="resume",
            source=resume_path,
        )
        if strict_checkpoint_provenance(cfg):
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_compatible_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("score", best_loss))
        if best_path.exists():
            best_checkpoint = _torch_load(best_path, map_location="cpu")
            validate_repo_checkpoint(
                best_checkpoint,
                cfg,
                data_contract,
                purpose="resume",
                source=best_path,
            )
            best_loss = float(best_checkpoint.get("score", best_loss))
        print(f"resumed checkpoint: {resume_path} at epoch {start_epoch - 1}")
    else:
        if initial_path:
            checkpoint = _torch_load(Path(initial_path), map_location=device)
            validate_repo_checkpoint(
                checkpoint,
                cfg,
                data_contract,
                purpose="warm_start",
                source=initial_path,
            )
            model.load_compatible_state_dict(checkpoint["model"])
            print(f"initialized compatible weights from: {initial_path}")
            stress_path = stress_initial_path
            if stress_path:
                stress_checkpoint = _torch_load(Path(stress_path), map_location=device)
                validate_repo_checkpoint(
                    stress_checkpoint,
                    cfg,
                    data_contract,
                    purpose="warm_start",
                    source=stress_path,
                )
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
                    data_contract,
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
                data_contract,
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
                data_contract,
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
                data_contract,
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
    if str(resume_from).lower() == "auto":
        path = output_dir / "latest.pt"
        return path if path.exists() else None
    path = Path(resume_from)
    if not path.exists():
        raise FileNotFoundError(f"explicit resume checkpoint does not exist: {path}")
    return path


def _validate_output_directory_for_resume(
    cfg: dict[str, Any],
    output_dir: Path,
    resume_path: Path | None,
) -> None:
    """Reject accidental fresh training in a strict, populated run directory."""

    if not strict_checkpoint_provenance(cfg) or not output_dir.exists():
        return
    snapshot_path = output_dir / "config_snapshot.json"
    if snapshot_path.exists():
        with snapshot_path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        if resume_config_sha256(snapshot) != resume_config_sha256(cfg):
            raise ValueError(
                f"{snapshot_path}: existing run config does not match strict resume"
            )
    artifacts = [
        path
        for path in output_dir.iterdir()
        if path.name in {"history.json", "best.pt", "latest.pt"}
        or (path.name.startswith("epoch_") and path.suffix == ".pt")
    ]
    if artifacts and resume_path is None:
        names = ", ".join(sorted(path.name for path in artifacts))
        raise RuntimeError(
            "strict_v2 refuses to start fresh in a populated output directory: "
            f"{names}"
        )


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
    final_reference = torch.zeros((), device=device)
    final_count = torch.zeros((), device=device)
    stress_p95_error = torch.zeros((), device=device)
    stress_p95_reference = torch.zeros((), device=device)
    max_evaluated_steps = 1

    for case_index in tqdm(case_indices, desc="rollout-val", leave=False):
        case = dataset.cases[case_index]
        tensors = dataset.gpu_builder.case_tensors(case, device)
        state = dataset.gpu_builder.state(case, 0, device)
        steps = case.num_steps - 1
        if requested_steps is not None:
            steps = min(steps, int(requested_steps))
        max_evaluated_steps = max(max_evaluated_steps, steps)
        free, stress_mask = _rollout_metric_masks(tensors)
        case_stress_truth: list[torch.Tensor] = []
        case_stress_residual: list[torch.Tensor] = []

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
            stress_dim = min(
                int(physical["stress"].shape[1]),
                int(tensors["S"].shape[-1]),
                1,
            )
            if stress_dim > 0:
                truth_stress = tensors["S"][
                    step + 1, stress_mask, :stress_dim
                ]
                stress_residual = (
                    physical["stress"][stress_mask, :stress_dim] - truth_stress
                )
                stress_error += (stress_residual * stress_residual).sum()
                stress_reference += (truth_stress * truth_stress).sum()
                stress_count += stress_residual.numel()
                case_stress_truth.append(truth_stress.reshape(-1))
                case_stress_residual.append(stress_residual.reshape(-1))
        if case_stress_truth:
            case_p95_error, case_p95_reference = _trajectory_stress_p95_sums(
                case_stress_truth,
                case_stress_residual,
            )
            stress_p95_error += case_p95_error
            stress_p95_reference += case_p95_reference
        final_residual = state["U"][free] - tensors["U"][steps, free]
        final_error += (final_residual * final_residual).sum()
        final_reference += (tensors["U"][steps, free] ** 2).sum()
        final_count += final_residual.numel()

    eps = torch.tensor(1.0e-12, device=device)
    displacement_rmse = torch.sqrt(u_error / u_count.clamp_min(1.0))
    displacement_relative = torch.sqrt(u_error / u_reference.clamp_min(eps))
    final_rmse = torch.sqrt(final_error / final_count.clamp_min(1.0))
    final_relative = torch.sqrt(final_error / final_reference.clamp_min(eps))
    if bool((stress_count > 0).item()):
        stress_rmse = torch.sqrt(stress_error / stress_count)
        stress_relative = torch.sqrt(stress_error / stress_reference.clamp_min(eps))
        stress_p95_relative = torch.sqrt(
            stress_p95_error / stress_p95_reference.clamp_min(eps)
        )
    else:
        stress_rmse = torch.zeros((), device=device)
        stress_relative = torch.zeros((), device=device)
        stress_p95_relative = torch.zeros((), device=device)
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
    weighted_score = displacement_normalized + stress_weight * stress_normalized
    result = {
        "displacement_rmse": float(displacement_rmse.item()),
        "displacement_relative_rmse": float(displacement_relative.item()),
        "moving_displacement_relative_rmse": float(displacement_relative.item()),
        "displacement_normalized_rmse": float(displacement_normalized.item()),
        "final_displacement_rmse": float(final_rmse.item()),
        "final_displacement_relative_rmse": float(final_relative.item()),
        "stress_rmse": float(stress_rmse.item()),
        "stress_relative_rmse": float(stress_relative.item()),
        "stress_p95_relative_rmse": float(stress_p95_relative.item()),
        "stress_normalized_rmse": float(stress_normalized.item()),
        "cases": float(case_count),
    }
    score_mode = str(
        get_cfg(cfg, "training.rollout_checkpoint_score_mode", "weighted_sum")
    ).lower()
    if score_mode == "four_metric_native_ratio_minimax":
        reference = _load_rollout_native_reference(cfg)
        ratios = {
            key: result[key] / reference[key] for key in ROLLOUT_METRIC_KEYS
        }
        result.update({f"native_ratio_{key}": value for key, value in ratios.items()})
        result["score"] = max(ratios.values())
    elif score_mode == "weighted_sum":
        result["score"] = float(weighted_score.item())
    else:
        raise ValueError(
            "training.rollout_checkpoint_score_mode must be weighted_sum or "
            "four_metric_native_ratio_minimax"
        )
    return result


def _rollout_metric_masks(
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the same moving/stress masks as the shared physical evaluator."""

    fixed = tensors["fixed"].bool()
    prescribed = tensors["prescribed"].bool()
    moving = ~(fixed | prescribed)
    if not bool(moving.any().item()):
        moving = ~fixed
    stress = ~prescribed
    if not bool(stress.any().item()):
        stress = torch.ones_like(prescribed, dtype=torch.bool)
    return moving, stress


def _trajectory_stress_p95_sums(
    truth_parts: list[torch.Tensor],
    residual_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool one trajectory, then select its truth-defined top-five-percent."""

    if len(truth_parts) != len(residual_parts) or not truth_parts:
        raise ValueError(
            "truth/residual trajectory parts must be non-empty and aligned"
        )
    truth = torch.cat([part.reshape(-1) for part in truth_parts], dim=0)
    residual = torch.cat([part.reshape(-1) for part in residual_parts], dim=0)
    if truth.shape != residual.shape:
        raise ValueError("truth and residual trajectory stress shapes differ")
    threshold = torch.quantile(truth.abs(), 0.95)
    peak = truth.abs() >= threshold
    return (residual[peak] ** 2).sum(), (truth[peak] ** 2).sum()


def _load_rollout_native_reference(cfg: dict[str, Any]) -> dict[str, float]:
    """Load the exact validation-only native artifact used for checkpointing."""

    path = get_cfg(cfg, "validation.native_reference_file", None)
    if not path:
        raise ValueError(
            "four_metric_native_ratio_minimax requires "
            "validation.native_reference_file"
        )
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    steps = get_cfg(cfg, "training.rollout_validation_steps", None)
    validate_reference_protocol(
        payload,
        split_file=get_cfg(cfg, "data.split_file"),
        split=str(get_cfg(cfg, "data.val_split", "val")),
        case_count=int(get_cfg(cfg, "training.rollout_validation_cases", 20)),
        frame_count=None if steps is None else int(steps) + 1,
        case_selection="even",
    )
    values: Any = payload
    for container in ("summary", "aggregate", "rollout"):
        if isinstance(values, dict) and isinstance(values.get(container), dict):
            values = values[container]
            break
    aliases = {
        "moving_displacement_relative_rmse": ("displacement_relative_rmse",),
        "final_displacement_relative_rmse": ("final_relative_rmse",),
        "stress_p95_relative_rmse": ("p95_stress_relative_rmse",),
    }
    reference: dict[str, float] = {}
    for key in ROLLOUT_METRIC_KEYS:
        value = values.get(key)
        for alias in aliases.get(key, ()):
            if value is None:
                value = values.get(alias)
        if value is None:
            raise KeyError(f"native validation reference is missing {key}")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                f"native validation reference {key} must be finite and positive"
            )
        reference[key] = numeric
    return reference


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
    data_contract: dict[str, Any] | None = None,
) -> None:
    model_state = model.state_dict()
    artifact_role = (
        "best"
        if path.name == "best.pt"
        else "latest"
        if path.name == "latest.pt"
        else "epoch"
    )
    state = {
        **checkpoint_metadata(
            cfg,
            model_state,
            data_contract,
            output_dim,
            artifact_role=artifact_role,
        ),
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg,
        "normalizers": normalizers.state_dict() if normalizers is not None else None,
        "output_dim": int(output_dim),
        "epoch": int(epoch),
        "score": float(score),
    }
    atomic_torch_save(state, path)


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

