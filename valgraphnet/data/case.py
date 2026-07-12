"""Load exported Abaqus case directories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    contact_surface_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=bool)
    )
    cells: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.int64)
    )
    dm_inv: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3, 3), dtype=np.float32)
    )
    reference_volume: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 1), dtype=np.float32)
    )
    shape_gradients: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4, 3), dtype=np.float32)
    )
    lumped_mass: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 1), dtype=np.float32)
    )
    cell_stress: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.float32)
    )
    integration_point_stress: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0, 6), dtype=np.float32)
    )
    integration_point_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=bool)
    )
    cell_strain: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.float32)
    )
    material: dict[str, Any] = field(default_factory=dict)
    material_features: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    density: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 1), dtype=np.float32)
    )
    fiber_direction: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    solver_nodal_force: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float32)
    )

    @property
    def num_steps(self) -> int:
        return int(self.times.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def stress_dim(self) -> int:
        return int(self.stress.shape[-1]) if self.stress.ndim == 3 else 0

    @property
    def num_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def has_constitutive_data(self) -> bool:
        return self.num_cells > 0 and self.dm_inv.shape == (self.num_cells, 3, 3)

    @property
    def has_solver_nodal_force(self) -> bool:
        return self.solver_nodal_force.shape == (
            self.num_steps,
            self.num_nodes,
            3,
        )


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
    cells = _load_cells(root, nodes.shape[0])
    num_cells = int(cells.shape[0])
    material = _load_json_optional(root / "material.json")
    density = _normalize_cell_scalar(
        root,
        "density.npy",
        _load_optional(root, "density.npy", np.ones((num_cells, 1), dtype=np.float32)),
        num_cells,
    )
    derived = _tetra_reference_geometry(root, nodes, cells)
    dm_inv = _load_or_derived(root, "Dm_inv.npy", derived[0], (num_cells, 3, 3))
    reference_volume = _normalize_cell_scalar(
        root,
        "reference_volume.npy",
        _load_optional(root, "reference_volume.npy", derived[1]),
        num_cells,
    )
    shape_gradients = _load_or_derived(
        root,
        "shape_gradients.npy",
        derived[2],
        (num_cells, 4, 3),
    )
    default_mass = _lumped_tetra_mass(nodes.shape[0], cells, reference_volume, density)
    lumped_mass = _normalize_nodal_scalar(
        root,
        "lumped_mass.npy",
        _load_optional(root, "lumped_mass.npy", default_mass),
        nodes.shape[0],
    )
    material_features = _normalize_cell_matrix(
        root,
        "material_features.npy",
        _load_optional(
            root,
            "material_features.npy",
            np.zeros((num_cells, 0), dtype=np.float32),
        ),
        num_cells,
    )
    fiber_direction = _normalize_fiber_direction(
        root,
        _load_optional(
            root,
            "fiber_direction.npy",
            np.zeros((num_cells, 3), dtype=np.float32),
        ),
        num_cells,
    )
    cell_stress = _normalize_cell_history(
        root,
        "S_cell.npy",
        _load_optional(
            root,
            "S_cell.npy",
            np.zeros((times.shape[0], num_cells, 0), dtype=np.float32),
        ),
        times.shape[0],
        num_cells,
    )
    cell_strain = _normalize_cell_history(
        root,
        "LE_cell.npy",
        _load_optional(
            root,
            "LE_cell.npy",
            np.zeros((times.shape[0], num_cells, 0), dtype=np.float32),
        ),
        times.shape[0],
        num_cells,
    )
    integration_point_stress, integration_point_mask = _load_integration_point_stress(
        root,
        times.shape[0],
        num_cells,
    )
    solver_force_path = root / "solver_nodal_force.npy"
    if solver_force_path.is_file():
        solver_nodal_force = np.load(
            solver_force_path, allow_pickle=False, mmap_mode="r"
        ).astype(np.float32, copy=False)
        expected_solver_force = (times.shape[0], nodes.shape[0], 3)
        if solver_nodal_force.shape != expected_solver_force:
            raise ValueError(
                f"{root}: solver_nodal_force.npy must have shape "
                f"{expected_solver_force}; found {solver_nodal_force.shape}"
            )
        if not bool(np.isfinite(solver_nodal_force).all()):
            raise ValueError(f"{root}: solver_nodal_force.npy contains non-finite values")
    else:
        solver_nodal_force = np.zeros((0, 0, 3), dtype=np.float32)
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
    contact_surface_path = root / "contact_surface_mask.npy"
    if contact_surface_path.exists():
        contact_surface_mask = np.load(
            contact_surface_path, allow_pickle=False, mmap_mode="r"
        ).astype(bool, copy=False)
        if contact_surface_mask.shape != (nodes.shape[0],):
            raise ValueError(
                f"{root}: contact_surface_mask.npy must have shape [{nodes.shape[0]}]"
            )
    else:
        # Empty distinguishes an absent optional mask from an explicitly empty
        # contact surface. CHP can then retain its tetra-boundary fallback.
        contact_surface_mask = np.zeros((0,), dtype=bool)
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
    contact_surface_mask = np.array(contact_surface_mask, dtype=bool, copy=True)
    leaflet_id = np.array(leaflet_id, dtype=np.int64, copy=True)
    thickness = np.array(thickness, dtype=np.float32, copy=True)

    _validate_case(root, nodes, times, pressure, displacement, velocity, acceleration, stress)
    if (
        elements.ndim == 2
        and elements.shape[1] == 2
        and metadata.get("element_representation")
        in {
            "unique two-node mesh edges derived from tetrahedral cells",
            "unique two-node mesh edges derived from tetrahedral cells and indenter triangles",
            "directed two-node mesh edges derived from tetrahedral cells and indenter triangles",
        }
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
        contact_surface_mask=contact_surface_mask,
        cells=cells,
        dm_inv=dm_inv,
        reference_volume=reference_volume,
        shape_gradients=shape_gradients,
        lumped_mass=lumped_mass,
        cell_stress=cell_stress,
        integration_point_stress=integration_point_stress,
        integration_point_mask=integration_point_mask,
        cell_strain=cell_strain,
        material=material,
        material_features=material_features,
        density=density,
        fiber_direction=fiber_direction,
        solver_nodal_force=solver_nodal_force,
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


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: material JSON must contain an object")
    return value


def _load_cells(root: Path, num_nodes: int) -> np.ndarray:
    path = root / "cells.npy"
    if path.exists():
        cells = np.load(path, allow_pickle=False, mmap_mode="r")
    else:
        cells = np.zeros((0, 4), dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError(f"{root}: cells.npy must have shape [M, 4]")
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= num_nodes):
        raise ValueError(f"{root}: cells.npy contains node indices outside [0, {num_nodes})")
    return cells


def _tetra_reference_geometry(
    root: Path,
    nodes: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_cells = int(cells.shape[0])
    if num_cells == 0:
        return (
            np.zeros((0, 3, 3), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 4, 3), dtype=np.float32),
        )
    vertices = np.asarray(nodes, dtype=np.float64)[cells]
    dm = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=-1,
    )
    determinant = np.linalg.det(dm)
    scale = np.maximum(np.linalg.norm(dm, axis=(1, 2)), 1.0)
    singular = np.abs(determinant) <= np.finfo(np.float64).eps * scale**3
    if np.any(singular):
        indices = np.flatnonzero(singular)[:5].tolist()
        raise ValueError(f"{root}: degenerate tetrahedral cells at indices {indices}")
    dm_inv = np.linalg.inv(dm)
    shape_gradients = np.empty((num_cells, 4, 3), dtype=np.float64)
    shape_gradients[:, 1:, :] = dm_inv
    shape_gradients[:, 0, :] = -dm_inv.sum(axis=1)
    volume = (np.abs(determinant) / 6.0)[:, None]
    return (
        dm_inv.astype(np.float32),
        volume.astype(np.float32),
        shape_gradients.astype(np.float32),
    )


def _load_or_derived(
    root: Path,
    name: str,
    default: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = _load_optional(root, name, default)
    array = np.asarray(array, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{root}: {name} must have shape {list(expected_shape)}")
    return array


def _normalize_cell_scalar(
    root: Path,
    name: str,
    value: np.ndarray,
    num_cells: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((num_cells, 1), float(array), dtype=np.float32)
    elif array.shape == (1,):
        array = np.full((num_cells, 1), float(array[0]), dtype=np.float32)
    elif array.shape == (num_cells,):
        array = array[:, None]
    elif array.shape == (1, 1):
        array = np.full((num_cells, 1), float(array[0, 0]), dtype=np.float32)
    if array.shape != (num_cells, 1):
        raise ValueError(f"{root}: {name} must be scalar, [M], or [M, 1]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{root}: {name} must contain finite values")
    if name in {"density.npy", "reference_volume.npy"} and np.any(array <= 0.0):
        raise ValueError(f"{root}: {name} must contain positive values")
    return array


def _normalize_nodal_scalar(
    root: Path,
    name: str,
    value: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (num_nodes,):
        array = array[:, None]
    if array.shape != (num_nodes, 1):
        raise ValueError(f"{root}: {name} must have shape [N] or [N, 1]")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{root}: {name} must contain finite non-negative values")
    return array


def _normalize_cell_matrix(
    root: Path,
    name: str,
    value: np.ndarray,
    num_cells: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = np.broadcast_to(array[None, :], (num_cells, array.shape[0])).copy()
    elif array.ndim == 2 and array.shape[0] == 1 and num_cells != 1:
        array = np.broadcast_to(array, (num_cells, array.shape[1])).copy()
    if array.ndim != 2 or array.shape[0] != num_cells:
        raise ValueError(f"{root}: {name} must have shape [P] or [M, P]")
    return array


def _normalize_fiber_direction(
    root: Path,
    value: np.ndarray,
    num_cells: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (3,):
        array = np.broadcast_to(array[None, :], (num_cells, 3)).copy()
    elif array.shape == (1, 3) and num_cells != 1:
        array = np.broadcast_to(array, (num_cells, 3)).copy()
    if array.shape != (num_cells, 3):
        raise ValueError(f"{root}: fiber_direction.npy must have shape [3] or [M, 3]")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0.0
    array = array.copy()
    array[nonzero] /= norms[nonzero]
    return array


def _normalize_cell_history(
    root: Path,
    name: str,
    value: np.ndarray,
    num_steps: int,
    num_cells: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[:2] != (num_steps, num_cells):
        raise ValueError(f"{root}: {name} must have shape [T, M, C]")
    if array.shape[2] not in (0, 6):
        raise ValueError(f"{root}: {name} must store zero or six symmetric tensor components")
    return array


def _load_integration_point_stress(
    root: Path,
    num_steps: int,
    num_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    stress = _load_optional(
        root,
        "S_integration_point.npy",
        np.zeros((num_steps, num_cells, 0, 6), dtype=np.float32),
    )
    stress = np.asarray(stress, dtype=np.float32)
    if stress.ndim != 4 or stress.shape[:2] != (num_steps, num_cells):
        raise ValueError(f"{root}: S_integration_point.npy must have shape [T, M, I, C]")
    if stress.shape[2] and stress.shape[3] != 6:
        raise ValueError(f"{root}: S_integration_point.npy must store six tensor components")
    default_mask = np.ones(stress.shape[:3], dtype=bool)
    mask = _load_optional(root, "integration_point_mask.npy", default_mask)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == stress.shape[1:3]:
        mask = np.broadcast_to(mask[None, :, :], stress.shape[:3]).copy()
    if mask.shape != stress.shape[:3]:
        raise ValueError(f"{root}: integration_point_mask.npy must have shape [M, I] or [T, M, I]")
    return stress, mask


def _lumped_tetra_mass(
    num_nodes: int,
    cells: np.ndarray,
    volume: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    mass = np.zeros((num_nodes, 1), dtype=np.float32)
    if cells.size:
        contribution = (volume[:, 0] * density[:, 0] / 4.0).astype(np.float32)
        for local_idx in range(4):
            np.add.at(mass[:, 0], cells[:, local_idx], contribution)
    return mass


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

