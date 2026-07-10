"""Autoregressive rollout evaluation for the native deforming_plate example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from valgraphnet.config import get_cfg, load_config
from valgraphnet.train import resolve_device

from .dataset import denormalize, load_sequences, load_stats, make_graph_sample, rollout_masks
from .preprocess import preprocess_cache_is_compatible, run_preprocess
from .train import (
    _build_case_backed_datasets,
    _load_case_sequence,
    autocast_context,
    build_deforming_plate_model,
)


@torch.no_grad()
def run_rollout_eval(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    out_dir: str | Path | None = None,
) -> Path:
    """Run native deforming_plate rollout evaluation and write metrics/artifacts."""

    cache_dir = Path(
        get_cfg(cfg, "data.preprocess_output_dir", "preprocessed_dataset/deforming_plate")
    )
    if get_cfg(cfg, "data.case_dir", None) and not (cache_dir / "edge_stats.pt").exists():
        _build_case_backed_datasets(cfg, cache_dir)
    elif not preprocess_cache_is_compatible(cfg, cache_dir):
        run_preprocess(cfg)

    edge_stats = load_stats(cache_dir / "edge_stats.pt")
    node_stats = load_stats(cache_dir / "node_stats.pt")
    checkpoint = _torch_load(checkpoint_path)
    ckpt_cfg = checkpoint.get("cfg", cfg)
    device = resolve_device(str(get_cfg(ckpt_cfg, "training.device", "auto")))

    model = build_deforming_plate_model(ckpt_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(
        out_dir
        or get_cfg(ckpt_cfg, "rollout.output_dir", "outputs/deforming_plate/rollout_eval")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = _load_rollout_sequences(ckpt_cfg)

    all_metrics = []
    for seq_idx, sequence in enumerate(sequences):
        pred_pos, exact_pos, pred_stress, exact_stress = _rollout_sequence(
            sequence=sequence,
            model=model,
            edge_stats=edge_stats,
            node_stats=node_stats,
            cfg=ckpt_cfg,
            device=device,
        )
        metrics = _sequence_metrics(pred_pos, exact_pos, pred_stress, exact_stress)
        metrics["sample_id"] = sequence.sample_id
        all_metrics.append(metrics)
        np.savez_compressed(
            output_dir / f"{sequence.sample_id}.npz",
            pred_world_pos=pred_pos,
            exact_world_pos=exact_pos,
            pred_stress=pred_stress,
            exact_stress=exact_stress,
            cells=sequence.cells,
            node_type=sequence.node_type,
        )
        if seq_idx == 0 and bool(get_cfg(ckpt_cfg, "rollout.make_gif", False)):
            _save_scatter_gif(
                output_dir / str(get_cfg(ckpt_cfg, "rollout.gif_name", "animation.gif")),
                pred_pos,
                exact_pos,
                frame_skip=int(get_cfg(ckpt_cfg, "rollout.frame_skip", 20)),
            )

    summary = _aggregate_metrics(all_metrics)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "sequences": all_metrics}, f, indent=2)
    return output_dir


def _load_rollout_sequences(cfg: dict[str, Any]):
    if get_cfg(cfg, "data.case_dir", None):
        from valgraphnet.data.case import read_split_file

        case_root = Path(get_cfg(cfg, "data.case_dir"))
        split_file = Path(get_cfg(cfg, "data.case_split_file", case_root / "splits.json"))
        case_ids = read_split_file(split_file, str(get_cfg(cfg, "data.test_case_split", "test")))
        limit = int(get_cfg(cfg, "data.num_test_samples", len(case_ids)))
        return [_load_case_sequence(str(case_root / case_id)) for case_id in case_ids[:limit]]
    return load_sequences(
        data_dir=get_cfg(cfg, "data.data_dir"),
        split=str(get_cfg(cfg, "data.test_split", "test")),
        num_samples=int(get_cfg(cfg, "data.num_test_samples", 5)),
        num_steps=int(get_cfg(cfg, "data.num_test_time_steps", 200)),
    )


def _rollout_sequence(
    sequence,
    model,
    edge_stats: dict[str, torch.Tensor],
    node_stats: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_pos = torch.as_tensor(sequence.world_pos[0], dtype=torch.float32)
    node_type = torch.as_tensor(sequence.node_type, dtype=torch.long)
    _, object_mask, clamped_mask = rollout_masks(node_type)
    boundary_mask = (object_mask | clamped_mask).float()

    pred_positions = [current_pos.numpy()]
    exact_positions = [sequence.world_pos[0].astype(np.float32)]
    pred_stresses = []
    exact_stresses = []
    for step in range(sequence.num_steps - 1):
        sample = make_graph_sample(
            sequence=sequence,
            step=step,
            edge_stats=edge_stats,
            node_stats=node_stats,
            world_pos_override=current_pos,
            world_edge_radius=float(get_cfg(cfg, "graph.world_edge_radius", 0.03)),
            max_world_neighbors=get_cfg(cfg, "graph.max_world_neighbors", None),
        )
        graph = sample.graph.to(device)
        with autocast_context(cfg, device):
            pred_norm = model(
                graph.x,
                sample.mesh_edge_features.to(device),
                sample.world_edge_features.to(device),
                graph,
            )
        pred_delta = denormalize(
            pred_norm[:, 0:3],
            node_stats["velocity_mean"],
            node_stats["velocity_std"],
        ).cpu()
        pred_stress = denormalize(
            pred_norm[:, 3:4],
            node_stats["stress_mean"],
            node_stats["stress_std"],
        ).cpu()
        exact_next = torch.as_tensor(sequence.world_pos[step + 1], dtype=torch.float32)
        next_pos = current_pos + pred_delta
        next_pos = next_pos * (1.0 - boundary_mask) + exact_next * boundary_mask
        current_pos = next_pos.detach()

        pred_positions.append(current_pos.numpy())
        exact_positions.append(exact_next.numpy())
        pred_stresses.append(pred_stress.numpy())
        stress = sequence.stress[step + 1].astype(np.float32)
        exact_stresses.append(stress[:, None] if stress.ndim == 1 else stress[:, :1])

    return (
        np.asarray(pred_positions, dtype=np.float32),
        np.asarray(exact_positions, dtype=np.float32),
        np.asarray(pred_stresses, dtype=np.float32),
        np.asarray(exact_stresses, dtype=np.float32),
    )


def _sequence_metrics(
    pred_pos: np.ndarray,
    exact_pos: np.ndarray,
    pred_stress: np.ndarray,
    exact_stress: np.ndarray,
) -> dict[str, float]:
    displacement_rmse = float(np.sqrt(np.mean((pred_pos - exact_pos) ** 2)))
    final_rmse = float(np.sqrt(np.mean((pred_pos[-1] - exact_pos[-1]) ** 2)))
    stress_rmse = (
        float(np.sqrt(np.mean((pred_stress - exact_stress) ** 2)))
        if pred_stress.size
        else 0.0
    )
    return {
        "displacement_rmse": displacement_rmse,
        "rollout_rmse": final_rmse,
        "stress_rmse": stress_rmse,
    }


def _aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = ["displacement_rmse", "rollout_rmse", "stress_rmse"]
    if not metrics:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_scatter_gif(
    path: Path,
    pred_pos: np.ndarray,
    exact_pos: np.ndarray,
    frame_skip: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError as exc:
        raise ImportError("matplotlib is required when rollout.make_gif is true") from exc

    frames = list(range(0, pred_pos.shape[0], max(frame_skip, 1)))
    fig = plt.figure(figsize=(10, 5))
    pred_ax = fig.add_subplot(1, 2, 1, projection="3d")
    exact_ax = fig.add_subplot(1, 2, 2, projection="3d")

    mins = np.minimum(pred_pos.min(axis=(0, 1)), exact_pos.min(axis=(0, 1)))
    maxs = np.maximum(pred_pos.max(axis=(0, 1)), exact_pos.max(axis=(0, 1)))

    def animate(frame_idx):
        pred_ax.cla()
        exact_ax.cla()
        p = pred_pos[frame_idx]
        e = exact_pos[frame_idx]
        pred_ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1)
        exact_ax.scatter(e[:, 0], e[:, 1], e[:, 2], s=1)
        pred_ax.set_title("Predicted")
        exact_ax.set_title("Exact")
        for ax in (pred_ax, exact_ax):
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=50)
    ani.save(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate native deforming_plate rollout.")
    parser.add_argument("--config", default="examples/deforming_plate/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = run_rollout_eval(cfg, args.checkpoint, args.out)
    print(f"rollout evaluation written to: {out}")


if __name__ == "__main__":
    main()
