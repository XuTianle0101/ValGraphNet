"""Train the native deforming_plate HybridMeshGraphNet example."""

from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from valgraphnet.config import get_cfg, load_config
from valgraphnet.data.case import read_split_file
from valgraphnet.train import resolve_device

from .dataset import (
    DeformingPlateSequence,
    fit_stats,
    load_stats,
    make_graph_sample,
    save_stats,
)
from .preprocess import run_preprocess


class CachedDeformingPlateDataset(Dataset):
    """Dataset over preprocess sequence cache files."""

    def __init__(self, cache_dir: str | Path, split: str) -> None:
        self.split_dir = Path(cache_dir) / split
        manifest_path = self.split_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing preprocess manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.entries = manifest["entries"]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        samples = _load_sequence_cache(str(self.split_dir / entry["file"]))
        return samples[int(entry["step"])]


@lru_cache(maxsize=16)
def _load_sequence_cache(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collate_one(items: list[dict[str, Any]]) -> dict[str, Any]:
    """The example uses batch_size=1, matching PhysicsNeMo deforming_plate."""

    if len(items) != 1:
        raise ValueError("deforming_plate native example currently expects batch_size=1")
    return items[0]


class CaseBackedDeformingPlateDataset(Dataset):
    """Lazy dataset over converted ValGraphNet deforming-plate cases."""

    def __init__(
        self,
        case_root: str | Path,
        split_file: str | Path,
        split: str,
        edge_stats: dict[str, torch.Tensor],
        node_stats: dict[str, torch.Tensor],
        cfg: dict[str, Any],
        add_noise: bool,
    ) -> None:
        self.case_root = Path(case_root)
        self.case_ids = read_split_file(split_file, split)
        self.edge_stats = edge_stats
        self.node_stats = node_stats
        self.world_edge_radius = float(get_cfg(cfg, "graph.world_edge_radius", 0.03))
        self.max_world_neighbors = get_cfg(cfg, "graph.max_world_neighbors", None)
        self.add_noise = bool(add_noise)
        self.noise_std = float(get_cfg(cfg, "data.noise_std", 0.003))
        self.samples: list[tuple[Path, int]] = []
        for case_id in self.case_ids:
            case_dir = self.case_root / case_id
            times = np.load(case_dir / "times.npy", allow_pickle=False, mmap_mode="r")
            for step in range(int(times.shape[0]) - 1):
                self.samples.append((case_dir, step))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        case_dir, step = self.samples[index]
        sequence = _load_case_sequence(str(case_dir))
        sample = make_graph_sample(
            sequence=sequence,
            step=step,
            edge_stats=self.edge_stats,
            node_stats=self.node_stats,
            world_edge_radius=self.world_edge_radius,
            max_world_neighbors=self.max_world_neighbors,
            add_noise=self.add_noise,
            noise_std=self.noise_std,
        )
        return {
            "graph": sample.graph,
            "mesh_edge_features": sample.mesh_edge_features,
            "world_edge_features": sample.world_edge_features,
        }


@lru_cache(maxsize=32)
def _load_case_sequence(path: str) -> DeformingPlateSequence:
    case_dir = Path(path)
    mesh_pos = np.load(case_dir / "nodes.npy", allow_pickle=False, mmap_mode="r")
    displacement = np.load(case_dir / "U.npy", allow_pickle=False, mmap_mode="r")
    cells = np.load(case_dir / "cells.npy", allow_pickle=False, mmap_mode="r")
    node_type = np.load(case_dir / "node_type.npy", allow_pickle=False, mmap_mode="r")
    stress = np.load(case_dir / "S.npy", allow_pickle=False, mmap_mode="r")
    world_pos = np.asarray(mesh_pos[None, :, :] + displacement, dtype=np.float32)
    return DeformingPlateSequence(
        sample_id=case_dir.name,
        mesh_pos=np.array(mesh_pos, dtype=np.float32, copy=True),
        world_pos=world_pos,
        cells=np.array(cells, dtype=np.int64, copy=True),
        node_type=np.array(node_type, dtype=np.int64, copy=True).reshape(-1),
        stress=np.array(stress, dtype=np.float32, copy=True),
    )


def run_training(cfg: dict[str, Any]) -> Path:
    """Train native deforming_plate model and return the best checkpoint path."""

    _set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(get_cfg(cfg, "training.device", "auto")))
    output_dir = Path(get_cfg(cfg, "training.output_dir", "outputs/deforming_plate"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(
        get_cfg(cfg, "data.preprocess_output_dir", "preprocessed_dataset/deforming_plate")
    )
    if get_cfg(cfg, "data.case_dir", None):
        train_dataset, val_dataset = _build_case_backed_datasets(cfg, cache_dir)
    else:
        if not (cache_dir / "train" / "manifest.json").exists():
            run_preprocess(cfg)
        train_dataset = CachedDeformingPlateDataset(cache_dir, "train")
        val_dataset = CachedDeformingPlateDataset(cache_dir, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(get_cfg(cfg, "training.batch_size", 1)),
        shuffle=True,
        num_workers=int(get_cfg(cfg, "training.num_workers", 0)),
        collate_fn=collate_one,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(get_cfg(cfg, "training.batch_size", 1)),
        shuffle=False,
        num_workers=int(get_cfg(cfg, "training.num_workers", 0)),
        collate_fn=collate_one,
    )

    model = build_deforming_plate_model(cfg).to(device)
    optimizer_kwargs = {
        "lr": float(get_cfg(cfg, "training.lr", 1.0e-4)),
        "weight_decay": float(get_cfg(cfg, "training.weight_decay", 0.0)),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(get_cfg(cfg, "training.lr_decay_rate", 0.9999917)),
    )
    criterion = torch.nn.MSELoss()
    amp_enabled, amp_dtype = _amp_settings(cfg, device)
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

    best_loss = float("inf")
    best_path = output_dir / "best.pt"
    epochs = int(get_cfg(cfg, "training.epochs", 30))
    save_every = int(get_cfg(cfg, "training.save_every", 5))
    save_latest = bool(get_cfg(cfg, "training.save_latest", False))
    start_epoch = 1
    resume_path = _resolve_resume_path(cfg, output_dir)
    if resume_path is not None:
        checkpoint = _torch_load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("score", best_loss))
        if best_path.exists():
            best_loss = float(_torch_load(best_path, map_location="cpu").get("score", best_loss))
        print(f"resumed checkpoint: {resume_path} at epoch {start_epoch - 1}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            cfg,
        )
        val_loss = _evaluate(model, val_loader, criterion, device, cfg)
        print(f"epoch {epoch:04d} | train={train_loss:.6g} | val={val_loss:.6g}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, cfg, epoch, best_loss)
        if save_latest:
            save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                epoch,
                val_loss,
            )
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                epoch,
                val_loss,
            )
    return best_path


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


def _build_case_backed_datasets(
    cfg: dict[str, Any],
    cache_dir: Path,
) -> tuple[CaseBackedDeformingPlateDataset, CaseBackedDeformingPlateDataset]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    edge_path = cache_dir / "edge_stats.pt"
    node_path = cache_dir / "node_stats.pt"
    if edge_path.exists() and node_path.exists():
        edge_stats = load_stats(edge_path)
        node_stats = load_stats(node_path)
    else:
        edge_stats, node_stats = _fit_case_backed_stats(cfg)
        save_stats(edge_path, edge_stats)
        save_stats(node_path, node_stats)

    case_root = Path(get_cfg(cfg, "data.case_dir"))
    split_file = Path(get_cfg(cfg, "data.case_split_file", case_root / "splits.json"))
    train_dataset = CaseBackedDeformingPlateDataset(
        case_root=case_root,
        split_file=split_file,
        split=str(get_cfg(cfg, "data.train_case_split", "train")),
        edge_stats=edge_stats,
        node_stats=node_stats,
        cfg=cfg,
        add_noise=True,
    )
    val_dataset = CaseBackedDeformingPlateDataset(
        case_root=case_root,
        split_file=split_file,
        split=str(get_cfg(cfg, "data.val_case_split", "val")),
        edge_stats=edge_stats,
        node_stats=node_stats,
        cfg=cfg,
        add_noise=False,
    )
    return train_dataset, val_dataset


def _fit_case_backed_stats(
    cfg: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    case_root = Path(get_cfg(cfg, "data.case_dir"))
    split_file = Path(get_cfg(cfg, "data.case_split_file", case_root / "splits.json"))
    train_ids = read_split_file(split_file, str(get_cfg(cfg, "data.train_case_split", "train")))
    sequences = (
        _load_case_sequence(str(case_root / case_id))
        for case_id in tqdm(train_ids, desc="load case-backed stats")
    )
    return fit_stats(
        sequences,
        world_edge_radius=float(get_cfg(cfg, "graph.world_edge_radius", 0.03)),
        max_world_neighbors=get_cfg(cfg, "graph.max_world_neighbors", None),
    )


def build_deforming_plate_model(cfg: dict[str, Any]):
    """Build the native deforming_plate HybridMeshGraphNet."""

    try:
        from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
    except ImportError as exc:
        raise ImportError("PhysicsNeMo is required for the deforming_plate example") from exc

    model_cfg = cfg.get("model", {})
    return HybridMeshGraphNet(
        input_dim_nodes=3,
        input_dim_edges=8,
        output_dim=4,
        processor_size=int(model_cfg.get("processor_size", 15)),
        hidden_dim_processor=int(model_cfg.get("hidden_dim_processor", 128)),
        aggregation=str(model_cfg.get("aggregation", "sum")),
        mlp_activation_fn=str(model_cfg.get("activation", "relu")),
        num_layers_node_processor=int(model_cfg.get("node_layers", 2)),
        num_layers_edge_processor=int(model_cfg.get("edge_layers", 2)),
        num_layers_node_decoder=int(model_cfg.get("decoder_layers", 2)),
        do_concat_trick=bool(model_cfg.get("do_concat_trick", False)),
        num_processor_checkpoint_segments=int(
            model_cfg.get("num_processor_checkpoint_segments", 0)
        ),
        checkpoint_offloading=bool(model_cfg.get("checkpoint_offloading", False)),
        recompute_activation=bool(model_cfg.get("recompute_activation", False)),
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    cfg: dict[str, Any],
    epoch: int,
    score: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "epoch": int(epoch),
            "score": float(score),
        },
        path,
    )


def _train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    device,
    cfg: dict[str, Any],
) -> float:
    model.train()
    total = 0.0
    count = 0
    for item in tqdm(loader, desc="train", leave=False):
        graph = item["graph"].to(device)
        mesh_edge_features = item["mesh_edge_features"].to(device)
        world_edge_features = item["world_edge_features"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(cfg, device):
            pred = model(graph.x, mesh_edge_features, world_edge_features, graph)
            loss = criterion(pred, graph.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device, cfg: dict[str, Any]) -> float:
    model.eval()
    total = 0.0
    count = 0
    for item in tqdm(loader, desc="val", leave=False):
        graph = item["graph"].to(device)
        with autocast_context(cfg, device):
            pred = model(
                graph.x,
                item["mesh_edge_features"].to(device),
                item["world_edge_features"].to(device),
                graph,
            )
            loss = criterion(pred, graph.y)
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def autocast_context(cfg: dict[str, Any], device: torch.device):
    """Return the configured CUDA autocast context."""

    enabled, dtype = _amp_settings(cfg, device)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _amp_settings(
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train native deforming_plate example.")
    parser.add_argument("--config", default="examples/deforming_plate/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    best = run_training(cfg)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
