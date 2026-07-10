"""Convert DeepMind deforming_plate TFRecords to ValGraphNet .npy cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from valgraphnet.config import get_cfg, load_config

from .dataset import NODE_TYPE_CLAMPED, NODE_TYPE_OBJECT, SequenceDataset, cells_to_edges


def convert_to_cases(cfg: dict[str, Any], out_dir: str | Path) -> Path:
    """Convert train/val/test deforming_plate sequences to ValGraphNet cases."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    split_specs = [
        (
            "train",
            str(get_cfg(cfg, "data.train_split", "train")),
            int(get_cfg(cfg, "data.num_training_samples", 1000)),
            int(get_cfg(cfg, "data.num_training_time_steps", 200)),
        ),
        (
            "val",
            str(get_cfg(cfg, "data.val_split", "valid")),
            int(get_cfg(cfg, "data.num_validation_samples", 100)),
            int(get_cfg(cfg, "data.num_validation_time_steps", 200)),
        ),
        (
            "test",
            str(get_cfg(cfg, "data.test_split", "test")),
            int(get_cfg(cfg, "data.num_test_samples", 5)),
            int(get_cfg(cfg, "data.num_test_time_steps", 200)),
        ),
    ]

    for target_split, source_split, num_samples, num_steps in split_specs:
        sequences = SequenceDataset(
            data_dir=get_cfg(cfg, "data.data_dir"),
            split=source_split,
            num_samples=num_samples,
            num_steps=num_steps,
        )
        for idx, sequence in enumerate(
            tqdm(sequences, desc=f"convert {target_split}", total=num_samples)
        ):
            case_id = f"{target_split}_{idx:05d}"
            _write_case(root / case_id, case_id, sequence)
            splits[target_split].append(case_id)

    with (root / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    return root


def _write_case(case_dir: Path, case_id: str, sequence) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    nodes = sequence.mesh_pos.astype(np.float32)
    world_pos = sequence.world_pos.astype(np.float32)
    displacement = world_pos - nodes[None, :, :]
    times = np.arange(sequence.num_steps, dtype=np.float32)
    velocity = np.zeros_like(displacement, dtype=np.float32)
    velocity[1:] = displacement[1:] - displacement[:-1]
    acceleration = np.zeros_like(displacement, dtype=np.float32)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    stress = sequence.stress.astype(np.float32)
    if stress.ndim == 2:
        stress = stress[:, :, None]
    stress = stress[:, :, :1]

    edge_elements = cells_to_edges(sequence.cells).t().cpu().numpy().astype(np.int64)
    fixed_mask = (sequence.node_type == NODE_TYPE_CLAMPED).astype(bool)
    prescribed_mask = (sequence.node_type == NODE_TYPE_OBJECT).astype(bool)
    pressure_mask = np.zeros(sequence.num_nodes, dtype=bool)
    leaflet_id = sequence.node_type.astype(np.int64)
    thickness = np.ones(sequence.num_nodes, dtype=np.float32)
    pressure = np.zeros(sequence.num_steps, dtype=np.float32)

    np.save(case_dir / "nodes.npy", nodes)
    np.save(case_dir / "elements.npy", edge_elements)
    np.save(case_dir / "times.npy", times)
    np.save(case_dir / "pressure.npy", pressure)
    np.save(case_dir / "U.npy", displacement)
    np.save(case_dir / "V.npy", velocity)
    np.save(case_dir / "A.npy", acceleration)
    np.save(case_dir / "S.npy", stress)
    np.save(case_dir / "fixed_mask.npy", fixed_mask)
    np.save(case_dir / "prescribed_mask.npy", prescribed_mask)
    np.save(case_dir / "pressure_mask.npy", pressure_mask)
    np.save(case_dir / "leaflet_id.npy", leaflet_id)
    np.save(case_dir / "thickness.npy", thickness)
    np.save(case_dir / "node_type.npy", sequence.node_type.astype(np.int64))
    np.save(case_dir / "cells.npy", sequence.cells.astype(np.int64))

    metadata = {
        "case_id": case_id,
        "source": "DeepMind deforming_plate",
        "sample_id": sequence.sample_id,
        "node_type_values": {
            "0": "moving",
            "1": "object",
            "3": "clamped",
        },
        "element_representation": "unique two-node mesh edges derived from tetrahedral cells",
    }
    with (case_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert deforming_plate TFRecords to ValGraphNet cases."
    )
    parser.add_argument("--config", default="examples/deforming_plate/config.yaml")
    parser.add_argument("--out", default="data/deforming_plate_cases")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = convert_to_cases(cfg, args.out)
    print(f"deforming_plate cases written to: {out}")


if __name__ == "__main__":
    main()
