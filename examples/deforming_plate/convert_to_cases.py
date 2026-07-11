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
    if stress.ndim != 3:
        raise ValueError(f"Expected deforming_plate stress [T, N, C], got {stress.shape}")

    cells = sequence.cells.astype(np.int64)
    edge_elements = cells_to_edges(cells).t().cpu().numpy().astype(np.int64)
    dm_inv, reference_volume, shape_gradients = _tetra_reference_geometry(nodes, cells)
    density = np.ones((cells.shape[0], 1), dtype=np.float32)
    lumped_mass = _lumped_mass(nodes.shape[0], cells, reference_volume, density)
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
    np.save(case_dir / "cells.npy", cells)
    np.save(case_dir / "Dm_inv.npy", dm_inv)
    np.save(case_dir / "reference_volume.npy", reference_volume)
    np.save(case_dir / "shape_gradients.npy", shape_gradients)
    np.save(case_dir / "lumped_mass.npy", lumped_mass)
    np.save(case_dir / "density.npy", density)

    metadata = {
        "case_id": case_id,
        "schema_version": 2,
        "source": "DeepMind deforming_plate",
        "sample_id": sequence.sample_id,
        "node_type_values": {
            "0": "moving",
            "1": "object",
            "3": "clamped",
        },
        "element_representation": "unique two-node mesh edges derived from tetrahedral cells",
        "cell_representation": "four-node linear tetrahedra",
        "stress_representation": "all nodal channels from the source TFRecord",
        "constitutive_fields": {
            "Dm_inv": "inverse reference edge matrix",
            "reference_volume": "absolute tetrahedral reference volume",
            "shape_gradients": "reference gradients of the four linear shape functions",
            "lumped_mass": "unit-density tetrahedral mass lumped equally to vertices",
            "density": "unit density in source units",
        },
    }
    with (case_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _tetra_reference_geometry(
    nodes: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(nodes, dtype=np.float64)[np.asarray(cells, dtype=np.int64)]
    dm = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=-1,
    )
    determinant = np.linalg.det(dm)
    if np.any(np.abs(determinant) <= np.finfo(np.float64).eps):
        bad = np.flatnonzero(np.abs(determinant) <= np.finfo(np.float64).eps)[:5].tolist()
        raise ValueError(f"Degenerate tetrahedral cells at indices {bad}")
    dm_inv = np.linalg.inv(dm)
    shape_gradients = np.empty((cells.shape[0], 4, 3), dtype=np.float64)
    shape_gradients[:, 1:, :] = dm_inv
    shape_gradients[:, 0, :] = -dm_inv.sum(axis=1)
    reference_volume = (np.abs(determinant) / 6.0)[:, None]
    return (
        dm_inv.astype(np.float32),
        reference_volume.astype(np.float32),
        shape_gradients.astype(np.float32),
    )


def _lumped_mass(
    num_nodes: int,
    cells: np.ndarray,
    reference_volume: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    mass = np.zeros((num_nodes, 1), dtype=np.float32)
    contribution = (reference_volume[:, 0] * density[:, 0] / 4.0).astype(np.float32)
    for local_idx in range(4):
        np.add.at(mass[:, 0], cells[:, local_idx], contribution)
    return mass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert deforming_plate TFRecords to ValGraphNet cases."
    )
    parser.add_argument("--config", default="examples/deforming_plate/config.yaml")
    parser.add_argument(
        "--out",
        default=None,
        help="Output root. Defaults to data.case_dir from the config.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = args.out or get_cfg(cfg, "data.case_dir", "data/deforming_plate_cases")
    out = convert_to_cases(cfg, out_dir)
    print(f"deforming_plate cases written to: {out}")


if __name__ == "__main__":
    main()
