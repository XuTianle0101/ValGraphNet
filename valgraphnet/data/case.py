"""Load exported Abaqus case directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from valgraphnet.geometry import compute_node_normals_areas, mesh_edges_from_elements


@dataclass
class ValveCase:
    case_id: str
    root: Path
    metadata: dict[str, Any]
    nodes: np.ndarray
    elements: np.ndarray
    times: np.ndarray
    pressure: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    stress: np.ndarray
    fixed_mask: np.ndarray
    prescribed_mask: np.ndarray
    pressure_mask: np.ndarray
    leaflet_id: np.ndarray
    thickness: np.ndarray
    normals: np.ndarray
    nodal_area: np.ndarray
    mesh_edge_index: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.times.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def stress_dim(self) -> int:
        return int(self.stress.shape[-1]) if self.stress.ndim == 3 else 0


def load_case(case_dir: str | Path) -> ValveCase:
    """Load one exported case directory."""

    root = Path(case_dir)
    metadata_path = root / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    nodes = _load_required(root, "nodes.npy").astype(np.float32, copy=False)
    elements = _load_required(root, "elements.npy").astype(np.int64, copy=False)
    times = _load_required(root, "times.npy").astype(np.float32, copy=False)
    pressure = _load_required(root, "pressure.npy").astype(np.float32, copy=False)
    displacement = _load_required(root, "U.npy").astype(np.float32, copy=False)
    velocity = _load_required(root, "V.npy").astype(np.float32, copy=False)
    acceleration = _load_required(root, "A.npy").astype(np.float32, copy=False)
    stress = _load_optional(root, "S.npy", np.zeros((*displacement.shape[:2], 0), dtype=np.float32))
    fixed_mask = _load_optional(
        root, "fixed_mask.npy", np.zeros(nodes.shape[0], dtype=bool)
    ).astype(bool, copy=False)
    prescribed_path = root / "prescribed_mask.npy"
    node_type_path = root / "node_type.npy"
    if prescribed_path.exists():
        prescribed_mask = np.load(
            prescribed_path, allow_pickle=False, mmap_mode="r"
        ).astype(bool, copy=False)
    elif node_type_path.exists():
        node_type = np.load(node_type_path, allow_pickle=False, mmap_mode="r")
        prescribed_mask = np.asarray(node_type).reshape(-1) == 1
    else:
        prescribed_mask = np.zeros(nodes.shape[0], dtype=bool)
    pressure_mask = _load_optional(
        root, "pressure_mask.npy", np.zeros(nodes.shape[0], dtype=bool)
    ).astype(bool, copy=False)
    leaflet_id = _load_optional(
        root, "leaflet_id.npy", np.zeros(nodes.shape[0], dtype=np.int64)
    ).astype(np.int64, copy=False)
    if metadata.get("source") == "DeepMind deforming_plate" and node_type_path.exists():
        leaflet_id = np.load(node_type_path, allow_pickle=False, mmap_mode="r").reshape(-1)
    thickness = _load_optional(
        root, "thickness.npy", np.ones(nodes.shape[0], dtype=np.float32)
    ).astype(np.float32, copy=False)
    if thickness.ndim == 0:
        thickness = np.full(nodes.shape[0], float(thickness), dtype=np.float32)
    fixed_mask = np.array(fixed_mask, dtype=bool, copy=True)
    prescribed_mask = np.array(prescribed_mask, dtype=bool, copy=True)
    pressure_mask = np.array(pressure_mask, dtype=bool, copy=True)
    leaflet_id = np.array(leaflet_id, dtype=np.int64, copy=True)
    thickness = np.array(thickness, dtype=np.float32, copy=True)

    _validate_case(root, nodes, times, pressure, displacement, velocity, acceleration, stress)
    if (
        elements.ndim == 2
        and elements.shape[1] == 2
        and metadata.get("element_representation")
        == "unique two-node mesh edges derived from tetrahedral cells"
    ):
        mesh_edge_index = np.asarray(elements, dtype=np.int64).T
    else:
        mesh_edge_index = mesh_edges_from_elements(elements)
    if elements.ndim == 2 and elements.shape[1] >= 3:
        normals, nodal_area = compute_node_normals_areas(nodes, elements)
    else:
        normals = np.zeros_like(nodes, dtype=np.float32)
        nodal_area = np.ones(nodes.shape[0], dtype=np.float32)

    return ValveCase(
        case_id=metadata.get("case_id", root.name),
        root=root,
        metadata=metadata,
        nodes=nodes,
        elements=elements,
        times=times,
        pressure=pressure,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
        stress=stress.astype(np.float32, copy=False),
        fixed_mask=fixed_mask,
        prescribed_mask=prescribed_mask,
        pressure_mask=pressure_mask,
        leaflet_id=leaflet_id,
        thickness=thickness,
        normals=normals,
        nodal_area=nodal_area,
        mesh_edge_index=mesh_edge_index,
    )


def discover_case_dirs(data_root: str | Path, case_ids: list[str] | None = None) -> list[Path]:
    """Discover case directories under a data root."""

    root = Path(data_root)
    if case_ids:
        return [root / case_id for case_id in case_ids]
    if (root / "nodes.npy").exists():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "nodes.npy").exists())


def read_split_file(path: str | Path, split: str) -> list[str]:
    """Read case ids from a split JSON file."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if split not in data:
        raise KeyError(f"Split '{split}' not found in {path}")
    return [str(case_id) for case_id in data[split]]


def _load_required(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required case file: {path}")
    return np.load(path, allow_pickle=False, mmap_mode="r")


def _load_optional(root: Path, name: str, default: np.ndarray) -> np.ndarray:
    path = root / name
    if path.exists():
        return np.load(path, allow_pickle=False, mmap_mode="r")
    return default


def _validate_case(
    root: Path,
    nodes: np.ndarray,
    times: np.ndarray,
    pressure: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    stress: np.ndarray,
) -> None:
    n_nodes = nodes.shape[0]
    n_steps = times.shape[0]
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"{root}: nodes.npy must have shape [N, 3]")
    if pressure.shape[0] != n_steps:
        raise ValueError(f"{root}: pressure.npy length must match times.npy")
    for name, array in {"U": displacement, "V": velocity, "A": acceleration}.items():
        if array.shape != (n_steps, n_nodes, 3):
            raise ValueError(f"{root}: {name}.npy must have shape [{n_steps}, {n_nodes}, 3]")
    if stress.ndim != 3 or stress.shape[0] != n_steps or stress.shape[1] != n_nodes:
        raise ValueError(f"{root}: S.npy must have shape [T, N, C]")

