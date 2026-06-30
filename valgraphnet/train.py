"""Training loop for ValGraphNet."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from valgraphnet.config import get_cfg
from valgraphnet.data import ValveGraphDataset, collate_valve_graphs
from valgraphnet.losses import valve_loss
from valgraphnet.model import build_model
from valgraphnet.normalization import fit_normalizers


def run_training(cfg: dict[str, Any]) -> Path:
    """Run model training and return the best checkpoint path."""

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(get_cfg(cfg, "training.device", "auto")))
    output_dir = Path(get_cfg(cfg, "training.output_dir", "outputs/valve_hybrid"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "config_snapshot.json", cfg)

    train_dataset, val_dataset = build_datasets(cfg)
    normalizers = None
    if bool(get_cfg(cfg, "normalization.enabled", True)):
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
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(get_cfg(cfg, "training.batch_size", 1)),
        shuffle=True,
        num_workers=int(get_cfg(cfg, "training.num_workers", 0)),
        collate_fn=collate_valve_graphs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(get_cfg(cfg, "training.batch_size", 1)),
            shuffle=False,
            num_workers=int(get_cfg(cfg, "training.num_workers", 0)),
            collate_fn=collate_valve_graphs,
        )

    best_loss = float("inf")
    best_path = output_dir / "best.pt"
    epochs = int(get_cfg(cfg, "training.epochs", 100))
    save_every = int(get_cfg(cfg, "training.save_every", 10))

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_metrics = evaluate(model, val_loader, device, cfg) if val_loader is not None else None
        score = val_metrics["total"] if val_metrics else train_metrics["total"]

        print(_format_epoch(epoch, train_metrics, val_metrics))
        if score < best_loss:
            best_loss = score
            save_checkpoint(best_path, model, optimizer, cfg, normalizers, train_dataset.output_dim, epoch, best_loss)
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                cfg,
                normalizers,
                train_dataset.output_dim,
                epoch,
                score,
            )

    return best_path


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


def train_one_epoch(model, loader, optimizer, device, cfg: dict[str, Any]) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss, metrics = valve_loss(pred, batch, cfg)
        loss.backward()
        clip_norm = get_cfg(cfg, "training.grad_clip_norm", None)
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
        optimizer.step()
        _accumulate(totals, metrics)
        count += 1
    return _average(totals, count)


@torch.no_grad()
def evaluate(model, loader, device, cfg: dict[str, Any]) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = batch.to(device)
        pred = model(batch)
        _, metrics = valve_loss(pred, batch, cfg)
        _accumulate(totals, metrics)
        count += 1
    return _average(totals, count)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    cfg: dict[str, Any],
    normalizers,
    output_dim: int,
    epoch: int,
    score: float,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
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


def _accumulate(totals: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def _average(totals: dict[str, float], count: int) -> dict[str, float]:
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def _format_epoch(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float] | None) -> str:
    train_part = ", ".join(f"{k}={v:.5g}" for k, v in sorted(train_metrics.items()))
    if val_metrics:
        val_part = ", ".join(f"{k}={v:.5g}" for k, v in sorted(val_metrics.items()))
        return f"epoch {epoch:04d} | train: {train_part} | val: {val_part}"
    return f"epoch {epoch:04d} | train: {train_part}"


def _save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

