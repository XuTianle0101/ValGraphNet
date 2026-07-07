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
from valgraphnet.train import resolve_device

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


def run_training(cfg: dict[str, Any]) -> Path:
    """Train native deforming_plate model and return the best checkpoint path."""

    _set_seed(int(cfg.get("seed", 42)))
    cache_dir = Path(
        get_cfg(cfg, "data.preprocess_output_dir", "preprocessed_dataset/deforming_plate")
    )
    if not (cache_dir / "train" / "manifest.json").exists():
        run_preprocess(cfg)

    device = resolve_device(str(get_cfg(cfg, "training.device", "auto")))
    output_dir = Path(get_cfg(cfg, "training.output_dir", "outputs/deforming_plate"))
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(get_cfg(cfg, "training.lr_decay_rate", 0.9999917)),
    )
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    best_path = output_dir / "best.pt"
    epochs = int(get_cfg(cfg, "training.epochs", 30))
    save_every = int(get_cfg(cfg, "training.save_every", 5))
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, criterion, device, cfg)
        val_loss = _evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch:04d} | train={train_loss:.6g} | val={val_loss:.6g}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, cfg, epoch, best_loss)
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                val_loss,
            )
    return best_path


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
        recompute_activation=bool(model_cfg.get("recompute_activation", False)),
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    cfg: dict[str, Any],
    epoch: int,
    score: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
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
    criterion,
    device,
    cfg: dict[str, Any],
) -> float:
    model.train()
    amp = bool(get_cfg(cfg, "training.amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total = 0.0
    count = 0
    for item in tqdm(loader, desc="train", leave=False):
        graph = item["graph"].to(device)
        mesh_edge_features = item["mesh_edge_features"].to(device)
        world_edge_features = item["world_edge_features"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
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
def _evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for item in tqdm(loader, desc="val", leave=False):
        graph = item["graph"].to(device)
        pred = model(
            graph.x,
            item["mesh_edge_features"].to(device),
            item["world_edge_features"].to(device),
            graph,
        )
        total += float(criterion(pred, graph.y).detach().cpu())
        count += 1
    return total / max(count, 1)


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
