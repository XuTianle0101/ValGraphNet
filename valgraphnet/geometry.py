"""Geometry and graph feature utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valgraphnet.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


@dataclass(frozen=True)
class ContactConfig:
    enabled: bool = True
    radius: float | None = None
    radius_factor: float = 2.5
    max_edges: int = 200_000
    different_leaflets_only: bool = True


def valid_element_nodes(element: np.ndarray) -> np.ndarray:
    """Return valid node ids from a padded element row."""

    element = np.asarray(element, dtype=np.int64)
    return element[element >= 0]


def mesh_edges_from_elements(elements: np.ndarray) -> np.ndarray:
    """Build directed, unique mesh edges from polygonal shell elements."""

    undirected: set[tuple[int, int]] = set()
    for raw in np.asarray(elements):
        elem = valid_element_nodes(raw)
        if elem.size < 2:
            continue
        for i, src in enumerate(elem):
            dst = elem[(i + 1) % elem.size]
            if src == dst:
                continue
            a, b = sorted((int(src), int(dst)))
            undirected.add((a, b))

    directed = []
    for a, b in sorted(undirected):
        directed.append((a, b))
        directed.append((b, a))
    if not directed:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(directed, dtype=np.int64).T


def undirected_edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    """Return an undirected edge set from a directed edge_index array."""

    if edge_index.size == 0:
        return set()
    edges = np.asarray(edge_index).T
    return {tuple(sorted((int(src), int(dst)))) for src, dst in edges if src != dst}


def compute_node_normals_areas(
    nodes: np.ndarray,
    elements: np.ndarray,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference nodal normals and nodal areas from shell elements."""

    nodes = np.asarray(nodes, dtype=np.float64)
    normals = np.zeros_like(nodes, dtype=np.float64)
    areas = np.zeros((nodes.shape[0],), dtype=np.float64)

    for raw in np.asarray(elements):
        elem = valid_element_nodes(raw)
        if elem.size < 3:
            continue

        anchor = elem[0]
        for i in range(1, elem.size - 1):
            tri = np.array([anchor, elem[i], elem[i + 1]], dtype=np.int64)
            v1 = nodes[tri[1]] - nodes[tri[0]]
            v2 = nodes[tri[2]] - nodes[tri[0]]
            area_vec = 0.5 * np.cross(v1, v2)
            area = float(np.linalg.norm(area_vec))
            if area <= eps:
                continue
            for node_id in tri:
                normals[node_id] += area_vec
                areas[node_id] += area / 3.0

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(norm, eps), out=np.zeros_like(normals), where=norm > eps)
    return normals.astype(np.float32), areas.astype(np.float32)


def median_mesh_edge_length(nodes: np.ndarray, edge_index: np.ndarray) -> float:
    """Return the median directed mesh-edge length."""

    if edge_index.size == 0:
        return 0.0
    src, dst = edge_index
    lengths = np.linalg.norm(nodes[dst] - nodes[src], axis=1)
    return float(np.median(lengths)) if lengths.size else 0.0


def build_contact_edges(
    current_pos: np.ndarray,
    leaflet_id: np.ndarray,
    mesh_edge_index: np.ndarray,
    cfg: ContactConfig,
    fixed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build directed world/contact edges between nearby non-mesh nodes."""

    if not cfg.enabled:
        return np.zeros((2, 0), dtype=np.int64)

    current_pos = np.asarray(current_pos, dtype=np.float64)
    leaflet_id = np.asarray(leaflet_id, dtype=np.int64)
    if current_pos.shape[0] < 2:
        return np.zeros((2, 0), dtype=np.int64)
    if cfg.different_leaflets_only and np.unique(leaflet_id).size < 2:
        return np.zeros((2, 0), dtype=np.int64)

    excluded = undirected_edge_set(mesh_edge_index)
    radius = cfg.radius
    if radius is None:
        ref_len = median_mesh_edge_length(current_pos, mesh_edge_index)
        radius = cfg.radius_factor * ref_len if ref_len > 0.0 else 0.0
    if radius <= 0.0:
        return np.zeros((2, 0), dtype=np.int64)

    if fixed_mask is None:
        fixed_mask = np.zeros((current_pos.shape[0],), dtype=bool)
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)

    pairs = _radius_pairs(current_pos, radius)
    directed: list[tuple[int, int]] = []
    for i, j in pairs:
        if i == j:
            continue
        if fixed_mask[i] and fixed_mask[j]:
            continue
        if cfg.different_leaflets_only and leaflet_id[i] == leaflet_id[j]:
            continue
        key = tuple(sorted((int(i), int(j))))
        if key in excluded:
            continue
        directed.append((int(i), int(j)))
        directed.append((int(j), int(i)))
        if len(directed) >= cfg.max_edges:
            break

    if not directed:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(directed, dtype=np.int64).T


def _radius_pairs(points: np.ndarray, radius: float) -> list[tuple[int, int]]:
    """Return unordered point pairs within a radius."""

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        return [(int(i), int(j)) for i, j in tree.query_pairs(radius)]
    except Exception:
        return _radius_pairs_chunked(points, radius)


def _radius_pairs_chunked(points: np.ndarray, radius: float, chunk_size: int = 1024) -> list[tuple[int, int]]:
    """Fallback O(N^2) radius search with modest memory use."""

    radius2 = float(radius * radius)
    pairs: list[tuple[int, int]] = []
    n_points = points.shape[0]
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        diff = points[start:stop, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        local_i, local_j = np.where(dist2 <= radius2)
        for li, j in zip(local_i, local_j, strict=False):
            i = start + int(li)
            j = int(j)
            if i < j:
                pairs.append((i, j))
    return pairs


def build_node_features(
    nodes: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    fixed_mask: np.ndarray,
    pressure_mask: np.ndarray,
    leaflet_id: np.ndarray,
    normals: np.ndarray,
    nodal_area: np.ndarray,
    thickness: np.ndarray,
    pressure_k: float,
    pressure_next: float,
    pressure_rate: float,
    phase: float,
    pressure_sign: float = 1.0,
) -> np.ndarray:
    """Build per-node input features."""

    n_nodes = nodes.shape[0]
    features = np.zeros((n_nodes, NODE_FEATURE_DIM), dtype=np.float32)
    pressure_mask_f = pressure_mask.astype(np.float32)
    fixed_mask_f = fixed_mask.astype(np.float32)
    traction = (
        pressure_sign
        * float(pressure_k)
        * normals
        * nodal_area[:, None]
        * pressure_mask_f[:, None]
    )

    max_leaflet = max(float(np.max(leaflet_id)), 1.0)
    phase_angle = 2.0 * np.pi * float(phase)

    features[:, 0:3] = nodes
    features[:, 3:6] = displacement
    features[:, 6:9] = velocity
    features[:, 9:12] = acceleration
    features[:, 12] = fixed_mask_f
    features[:, 13] = pressure_mask_f
    features[:, 14] = leaflet_id.astype(np.float32) / max_leaflet
    features[:, 15:18] = normals
    features[:, 18] = nodal_area.astype(np.float32)
    features[:, 19] = thickness.astype(np.float32)
    features[:, 20] = float(pressure_k)
    features[:, 21] = float(pressure_next)
    features[:, 22] = float(pressure_rate)
    features[:, 23] = np.sin(phase_angle)
    features[:, 24] = np.cos(phase_angle)
    features[:, 25:28] = traction.astype(np.float32)
    return features


def build_edge_features(
    nodes: np.ndarray,
    current_pos: np.ndarray,
    normals: np.ndarray,
    edge_index: np.ndarray,
    pressure_mask: np.ndarray,
    fixed_mask: np.ndarray,
    leaflet_id: np.ndarray,
    is_world_edge: bool,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Build fixed-width edge features for mesh or world edges."""

    if edge_index.size == 0:
        return np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

    src, dst = edge_index.astype(np.int64)
    ref_rel = nodes[dst] - nodes[src]
    cur_rel = current_pos[dst] - current_pos[src]
    ref_len = np.linalg.norm(ref_rel, axis=1, keepdims=True)
    cur_len = np.linalg.norm(cur_rel, axis=1, keepdims=True)
    normal_dot = np.sum(normals[src] * normals[dst], axis=1, keepdims=True)
    pressure_pair = 0.5 * (
        pressure_mask[src].astype(np.float32)[:, None] + pressure_mask[dst].astype(np.float32)[:, None]
    )
    fixed_pair = (
        fixed_mask[src].astype(np.float32)[:, None] * fixed_mask[dst].astype(np.float32)[:, None]
    )
    same_leaflet = (leaflet_id[src] == leaflet_id[dst]).astype(np.float32)[:, None]

    features = np.zeros((edge_index.shape[1], EDGE_FEATURE_DIM), dtype=np.float32)
    features[:, 0:3] = ref_rel
    features[:, 3:4] = np.maximum(ref_len, eps)
    features[:, 4:7] = cur_rel
    features[:, 7:8] = np.maximum(cur_len, eps)
    features[:, 8:9] = normal_dot
    features[:, 9:10] = pressure_pair
    features[:, 10:11] = fixed_pair
    features[:, 11:12] = cur_len if is_world_edge else 0.0
    features[:, 12:13] = same_leaflet
    features[:, 13] = 1.0 if is_world_edge else 0.0
    return features

