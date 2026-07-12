"""CUDA/BF16 training protocol for the two-level MultiScale MGN baseline."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from valgraphnet.config import get_cfg
from valgraphnet.data.case import read_split_file
from valgraphnet.fair_train import (
    ROLLOUT_METRIC_KEYS,
    evaluate_fair_rollouts,
    fit_fair_statistics,
    load_native_reference,
    minimax_checkpoint_score,
    require_cuda_bf16,
    train_fair_epoch,
)
from valgraphnet.multiscale_baseline import MultiscaleDeformingPlateBaseline
from valgraphnet.normalization import Normalizers
from valgraphnet.stress_transform import AsinhStressTransform
from valgraphnet.train import build_datasets


MULTISCALE_CHECKPOINT_SCHEMA_VERSION = 2
MULTISCALE_MODEL_FAMILY = "two_level_bistride_deforming_plate_mgn"


def validate_multiscale_development_protocol(cfg: dict[str, Any]) -> None:
    """Fail closed if checkpoint selection can touch test or a short rollout."""

    train_split = str(get_cfg(cfg, "data.train_split", "train"))
    val_split = str(get_cfg(cfg, "data.val_split", "val"))
    test_split = str(get_cfg(cfg, "data.test_split", "test"))
    if train_split == test_split or val_split == test_split:
        raise ValueError("the test split cannot be used for training or checkpoint selection")
    if int(get_cfg(cfg, "validation.cases", -1)) != 20:
        raise ValueError("MultiScale checkpoint selection requires validation.cases=20")
    if int(get_cfg(cfg, "validation.steps", -1)) != 399:
        raise ValueError("MultiScale checkpoint selection requires all 400 frames")
    if str(get_cfg(cfg, "validation.native_reference_split", "")) != val_split:
        raise ValueError("the native checkpoint reference must use the validation split")
    if str(get_cfg(cfg, "validation.native_reference_case_selection", "")) != "even":
        raise ValueError("the validation subset must use deterministic even selection")
    if not bool(get_cfg(cfg, "validation.require_native_reference", False)):
        raise ValueError("four-metric minimax selection requires a native reference")
    if not bool(
        get_cfg(cfg, "validation.require_native_reference_provenance", False)
    ):
        raise ValueError("native validation provenance must be verified")
    if int(get_cfg(cfg, "model.num_mesh_levels", -1)) != 2:
        raise ValueError("the comparison architecture requires exactly two mesh levels")
    split_file = get_cfg(cfg, "data.split_file", None)
    if split_file is None:
        raise ValueError("an explicit train/val/test split file is required")
    train_ids = set(read_split_file(split_file, train_split))
    val_ids = set(read_split_file(split_file, val_split))
    test_ids = set(read_split_file(split_file, test_split))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("train, validation, and test trajectory ids must be disjoint")


def run_multiscale_training(cfg: dict[str, Any]) -> Path:
    """Train and return the strictly rollout-validated minimax checkpoint."""

    validate_multiscale_development_protocol(cfg)
    device = require_cuda_bf16(cfg)
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    # Validate the reference artifact before loading 23+ GiB of trajectory
    # metadata or fitting statistics.  A missing/mismatched val20 reference is
    # a protocol error, not an invitation to fall back to one-step loss.
    native_reference = load_native_reference(cfg)
    train_dataset, val_dataset = build_datasets(cfg)
    if val_dataset is None:
        raise ValueError("MultiScale checkpointing requires a validation split")
    short_or_long = [
        case.case_id
        for dataset in (train_dataset, val_dataset)
        for case in dataset.cases
        if int(case.num_steps) != 400
    ]
    if short_or_long:
        raise ValueError(
            "the full400 baseline requires exactly 400 frames per trajectory: "
            + ", ".join(short_or_long[:3])
        )
    normalizers, stress_transform = fit_fair_statistics(train_dataset, cfg)
    train_dataset.normalizers = normalizers
    val_dataset.normalizers = normalizers

    model = MultiscaleDeformingPlateBaseline(cfg).to(device)
    optimizer_kwargs: dict[str, Any] = {
        "lr": float(get_cfg(cfg, "training.lr", 1.0e-4)),
        "weight_decay": float(get_cfg(cfg, "training.weight_decay", 0.0)),
    }
    if bool(get_cfg(cfg, "training.fused_optimizer", True)):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    epochs = int(get_cfg(cfg, "training.epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(get_cfg(cfg, "training.min_lr", 1.0e-6)),
    )
    output_dir = Path(
        get_cfg(
            cfg,
            "training.output_dir",
            "outputs/deforming_plate_multiscale_mgn",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    history_path = output_dir / "history.json"
    start_epoch = 1
    best_score = float("inf")
    history: list[dict[str, Any]] = []

    resume = get_cfg(cfg, "training.resume_from", None)
    resume_path = (
        latest_path
        if str(resume).lower() == "auto"
        else Path(resume)
        if resume
        else None
    )
    if resume_path is not None and resume_path.exists():
        checkpoint = _torch_load(resume_path, device)
        _validate_multiscale_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        normalizers = Normalizers.from_state_dict(checkpoint["normalizers"])
        stress_transform = AsinhStressTransform.from_state_dict(
            checkpoint["stress_transform"]
        )
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
        if _validation_due(epoch, epochs, validation_every):
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

        payload = _checkpoint_payload(
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
            min(best_score, score),
        )
        if rollout_metrics is not None and math.isfinite(score) and score < best_score:
            best_score = score
            payload["best_score"] = best_score
            torch.save(payload, best_path)
        payload["best_score"] = best_score
        torch.save(payload, latest_path)
        if save_every > 0 and epoch % save_every == 0:
            torch.save(payload, output_dir / f"epoch_{epoch:04d}.pt")
    if not best_path.exists() or not math.isfinite(best_score):
        raise RuntimeError(
            "two-level MultiScale MGN produced no finite rollout-validated checkpoint"
        )
    return best_path


def load_multiscale_checkpoint(
    cfg: dict[str, Any], path: str | Path
) -> tuple[
    MultiscaleDeformingPlateBaseline,
    Normalizers,
    AsinhStressTransform,
    dict[str, Any],
]:
    """Load a frozen MultiScale baseline directly onto the required GPU."""

    device = require_cuda_bf16(cfg)
    checkpoint = _torch_load(path, device)
    _validate_multiscale_checkpoint(checkpoint)
    model = MultiscaleDeformingPlateBaseline(checkpoint.get("cfg", cfg)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (
        model,
        Normalizers.from_state_dict(checkpoint["normalizers"]),
        AsinhStressTransform.from_state_dict(checkpoint["stress_transform"]),
        checkpoint,
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
        "schema_version": MULTISCALE_CHECKPOINT_SCHEMA_VERSION,
        "model_family": MULTISCALE_MODEL_FAMILY,
        "architecture": "physicsnemo_bistride_mgn_two_topology_levels",
        "hierarchy_edge_source": "static_mesh_only_no_contact_edges",
        "output_contract": "normalized_delta_x_plus_asinh_stress",
        "state_integration": "velocity_and_acceleration_derived_from_delta_x",
        "training_device": "cuda",
        "training_precision": "bfloat16",
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": cfg,
        "normalizers": normalizers.state_dict(),
        "stress_transform": stress_transform.state_dict(),
        "native_reference": native_reference,
        "rollout_metrics": rollout_metrics,
        "checkpoint_metric": "four_metric_native_ratio_minimax",
        "checkpoint_split": str(get_cfg(cfg, "data.val_split", "val")),
        "checkpoint_case_selection": "even",
        "checkpoint_frames": 400,
        "epoch": int(epoch),
        "score": float(score),
        "best_score": float(best_score),
    }


def _validate_multiscale_checkpoint(checkpoint: dict[str, Any]) -> None:
    if (
        int(checkpoint.get("schema_version", 0))
        != MULTISCALE_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported MultiScale baseline checkpoint schema")
    if checkpoint.get("model_family") != MULTISCALE_MODEL_FAMILY:
        raise ValueError("checkpoint is not the two-level MultiScale MGN baseline")
    if checkpoint.get("checkpoint_split") == "test":
        raise ValueError("test-selected checkpoints are scientifically invalid")


def _validation_due(epoch: int, total_epochs: int, every: int) -> bool:
    return int(every) > 0 and (
        int(epoch) == int(total_epochs) or int(epoch) % int(every) == 0
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    train = ", ".join(
        f"{key}={value:.5g}" for key, value in sorted(row["train"].items())
    )
    rollout = row["rollout"]
    if rollout:
        selected = ", ".join(
            f"{key}={rollout[key]:.5g}"
            for key in (*ROLLOUT_METRIC_KEYS, "minimax_score")
        )
        return f"epoch {row['epoch']:04d} | train: {train} | rollout: {selected}"
    return f"epoch {row['epoch']:04d} | train: {train}"
