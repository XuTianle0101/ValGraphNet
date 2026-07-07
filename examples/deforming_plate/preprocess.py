"""Preprocess DeepMind deforming_plate TFRecords for native training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from valgraphnet.config import get_cfg, load_config

from .dataset import fit_stats, load_sequences, load_stats, make_graph_sample, save_stats


def run_preprocess(cfg: dict[str, Any]) -> Path:
    """Create cached native deforming_plate graph samples and stats."""

    data_dir = Path(get_cfg(cfg, "data.data_dir"))
    out_dir = Path(
        get_cfg(cfg, "data.preprocess_output_dir", "preprocessed_dataset/deforming_plate")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sequences = load_sequences(
        data_dir=data_dir,
        split=str(get_cfg(cfg, "data.train_split", "train")),
        num_samples=int(get_cfg(cfg, "data.num_training_samples", 1000)),
        num_steps=int(get_cfg(cfg, "data.num_training_time_steps", 200)),
    )
    edge_stats, node_stats = fit_stats(
        train_sequences,
        world_edge_radius=float(get_cfg(cfg, "graph.world_edge_radius", 0.03)),
    )
    save_stats(out_dir / "edge_stats.pt", edge_stats)
    save_stats(out_dir / "node_stats.pt", node_stats)

    _write_split_cache(
        cfg=cfg,
        split_name="train",
        source_split=str(get_cfg(cfg, "data.train_split", "train")),
        sequences=train_sequences,
        edge_stats=edge_stats,
        node_stats=node_stats,
        out_dir=out_dir,
        add_noise=True,
    )
    _write_split_cache(
        cfg=cfg,
        split_name="val",
        source_split=str(get_cfg(cfg, "data.val_split", "valid")),
        sequences=None,
        edge_stats=edge_stats,
        node_stats=node_stats,
        out_dir=out_dir,
        add_noise=False,
    )
    _write_split_cache(
        cfg=cfg,
        split_name="test",
        source_split=str(get_cfg(cfg, "data.test_split", "test")),
        sequences=None,
        edge_stats=edge_stats,
        node_stats=node_stats,
        out_dir=out_dir,
        add_noise=False,
    )
    return out_dir


def _write_split_cache(
    cfg: dict[str, Any],
    split_name: str,
    source_split: str,
    sequences,
    edge_stats: dict[str, torch.Tensor],
    node_stats: dict[str, torch.Tensor],
    out_dir: Path,
    add_noise: bool,
) -> None:
    data_dir = Path(get_cfg(cfg, "data.data_dir"))
    if sequences is None:
        sample_key = split_name if split_name != "val" else "validation"
        sequences = load_sequences(
            data_dir=data_dir,
            split=source_split,
            num_samples=int(get_cfg(cfg, f"data.num_{sample_key}_samples", 5)),
            num_steps=int(get_cfg(cfg, f"data.num_{sample_key}_time_steps", 200)),
        )
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for seq_idx, sequence in enumerate(tqdm(sequences, desc=f"preprocess {split_name}")):
        samples = []
        for step in range(sequence.num_steps - 1):
            sample = make_graph_sample(
                sequence=sequence,
                step=step,
                edge_stats=edge_stats,
                node_stats=node_stats,
                world_edge_radius=float(get_cfg(cfg, "graph.world_edge_radius", 0.03)),
                add_noise=add_noise,
                noise_std=float(get_cfg(cfg, "data.noise_std", 0.003)),
            )
            samples.append(
                {
                    "graph": sample.graph,
                    "mesh_edge_features": sample.mesh_edge_features,
                    "world_edge_features": sample.world_edge_features,
                }
            )
            entries.append({"file": f"sequence_{seq_idx:05d}.pt", "step": step})
        torch.save(samples, split_dir / f"sequence_{seq_idx:05d}.pt")

    manifest = {
        "split": split_name,
        "source_split": source_split,
        "num_sequences": len(sequences),
        "num_samples": len(entries),
        "entries": entries,
    }
    with (split_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_preprocess_stats(
    cache_dir: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load edge and node stats from a preprocess cache directory."""

    cache = Path(cache_dir)
    return load_stats(cache / "edge_stats.pt"), load_stats(cache / "node_stats.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess deforming_plate TFRecords.")
    parser.add_argument("--config", default="examples/deforming_plate/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = run_preprocess(cfg)
    print(f"preprocessed deforming_plate data written to: {out}")


if __name__ == "__main__":
    main()
