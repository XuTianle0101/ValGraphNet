"""CUDA graph construction and autoregressive state updates."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from valgraphnet.config import get_cfg
from valgraphnet.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from valgraphnet.data.case import ValveCase


class GpuGraphBuilder:
    """Build deforming graphs on CUDA while retaining a small trajectory cache."""

    def __init__(self, cfg: dict[str, Any], stress_dim: int) -> None:
        self.cfg = cfg
        self.stress_dim = int(stress_dim)
        self.cache_size = int(get_cfg(cfg, "training.gpu_case_cache_size", 8))
        self._cache: OrderedDict[tuple[str, str], dict[str, torch.Tensor]] = OrderedDict()

    def state(
        self, case: ValveCase, step: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        tensors = self.case_tensors(case, device)
        return {
            "U": tensors["U"][step],
            "V": tensors["V"][step],
            "A": tensors["A"][step],
        }

    def case_tensors(
        self, case: ValveCase, device: torch.device
    ) -> dict[str, torch.Tensor]:
        key = (str(case.root.resolve()), str(device))
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached

        def tensor(array, dtype=None):
            value = torch.from_numpy(np.array(array, copy=True))
            if dtype is not None:
                value = value.to(dtype=dtype)
            return value.to(device=device, non_blocking=True)

        cached = {
            "nodes": tensor(case.nodes, torch.float32),
            "times": tensor(case.times, torch.float32),
            "pressure": tensor(case.pressure, torch.float32),
            "U": tensor(case.displacement, torch.float32),
            "V": tensor(case.velocity, torch.float32),
            "A": tensor(case.acceleration, torch.float32),
            "S": tensor(case.stress, torch.float32),
            "fixed": tensor(case.fixed_mask, torch.bool),
            "prescribed": tensor(case.prescribed_mask, torch.bool),
            "pressure_mask": tensor(case.pressure_mask, torch.bool),
            "leaflet": tensor(case.leaflet_id, torch.long),
            "thickness": tensor(case.thickness, torch.float32),
            "normals": tensor(case.normals, torch.float32),
            "area": tensor(case.nodal_area, torch.float32),
            "mesh_edges": tensor(case.mesh_edge_index, torch.long),
        }
        self._cache[key] = cached
        while len(self._cache) > max(self.cache_size, 1):
            self._cache.popitem(last=False)
        return cached

    def make_graph(
        self,
        case: ValveCase,
        step: int,
        device: torch.device,
        state: dict[str, torch.Tensor] | None = None,
    ):
        """Create a device-resident PyG graph for one autoregressive step."""

        from torch_geometric.data import Data

        t = self.case_tensors(case, device)
        u = t["U"][step] if state is None else state["U"]
        v = t["V"][step] if state is None else state["V"]
        a = t["A"][step] if state is None else state["A"]
        current_pos = t["nodes"] + u
        dt = t["times"][step + 1] - t["times"][step]
        if bool((dt <= 0).item()):
            raise ValueError(f"{case.root}: non-positive dt at step {step}")

        pressure_sign = float(get_cfg(self.cfg, "data.pressure_sign", 1.0))
        pressure_k = pressure_sign * t["pressure"][step]
        pressure_next = pressure_sign * t["pressure"][step + 1]
        pressure_rate = (pressure_next - pressure_k) / dt
        span = (t["times"][-1] - t["times"][0]).clamp_min(1.0e-12)
        phase = (t["times"][step] - t["times"][0]) / span

        node_features = _node_features(
            t, u, v, a, pressure_k, pressure_next, pressure_rate, phase
        )
        world_edges = _contact_edges(current_pos, t, self.cfg)
        mesh_features = _edge_features(t, current_pos, t["mesh_edges"], False)
        world_features = _edge_features(t, current_pos, world_edges, True)
        edge_index = torch.cat([t["mesh_edges"], world_edges], dim=1)

        target_stress = t["S"][step + 1]
        if target_stress.shape[1] < self.stress_dim:
            target_stress = torch.nn.functional.pad(
                target_stress, (0, self.stress_dim - target_stress.shape[1])
            )
        target = torch.cat(
            [
                t["U"][step + 1] - u,
                t["V"][step + 1] - v,
                t["A"][step + 1],
                target_stress[:, : self.stress_dim],
            ],
            dim=1,
        )
        data = Data(edge_index=edge_index, num_nodes=case.num_nodes)
        data.node_features = node_features
        data.mesh_edge_features = mesh_features
        data.world_edge_features = world_features
        data.target = target
        data.fixed_mask = t["fixed"]
        data.prescribed_mask = t["prescribed"]
        data.pressure_mask = t["pressure_mask"]
        data.nodal_area = t["area"]
        data.case_id = case.case_id
        data.step = int(step)
        data.dt = dt
        data.mesh_edge_count = int(t["mesh_edges"].shape[1])
        data.world_edge_count = int(world_edges.shape[1])
        return data


def update_state(
    prediction: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    case_tensors: dict[str, torch.Tensor],
    next_step: int,
) -> dict[str, torch.Tensor]:
    """Apply one physical prediction and exact boundary conditions on GPU."""

    fixed = case_tensors["fixed"][:, None]
    prescribed = case_tensors["prescribed"][:, None]
    u = state["U"] + torch.where(
        fixed, torch.zeros_like(prediction["delta_u"]), prediction["delta_u"]
    )
    v = state["V"] + torch.where(
        fixed, torch.zeros_like(prediction["delta_v"]), prediction["delta_v"]
    )
    a = torch.where(
        fixed, torch.zeros_like(prediction["accel"]), prediction["accel"]
    )
    return {
        "U": torch.where(prescribed, case_tensors["U"][next_step], u),
        "V": torch.where(prescribed, case_tensors["V"][next_step], v),
        "A": torch.where(prescribed, case_tensors["A"][next_step], a),
    }


def _node_features(t, u, v, a, pressure_k, pressure_next, pressure_rate, phase):
    count = t["nodes"].shape[0]
    out = torch.zeros(
        (count, NODE_FEATURE_DIM), dtype=torch.float32, device=t["nodes"].device
    )
    fixed = t["fixed"].float()
    pressure_mask = t["pressure_mask"].float()
    max_leaflet = t["leaflet"].max().float().clamp_min(1.0)
    angle = 2.0 * torch.pi * phase
    traction = pressure_k * t["normals"] * t["area"][:, None] * pressure_mask[:, None]
    out[:, 0:3] = t["nodes"]
    out[:, 3:6] = u
    out[:, 6:9] = v
    out[:, 9:12] = a
    out[:, 12] = fixed
    out[:, 13] = pressure_mask
    out[:, 14] = t["leaflet"].float() / max_leaflet
    out[:, 15:18] = t["normals"]
    out[:, 18] = t["area"]
    out[:, 19] = t["thickness"]
    out[:, 20] = pressure_k
    out[:, 21] = pressure_next
    out[:, 22] = pressure_rate
    out[:, 23] = torch.sin(angle)
    out[:, 24] = torch.cos(angle)
    out[:, 25:28] = traction
    return out


def _edge_features(t, current_pos, edge_index, is_world_edge):
    if edge_index.numel() == 0:
        return torch.zeros(
            (0, EDGE_FEATURE_DIM), dtype=torch.float32, device=current_pos.device
        )
    src, dst = edge_index
    ref_rel = t["nodes"][dst] - t["nodes"][src]
    cur_rel = current_pos[dst] - current_pos[src]
    ref_len = torch.linalg.vector_norm(ref_rel, dim=1, keepdim=True).clamp_min(1.0e-12)
    cur_len = torch.linalg.vector_norm(cur_rel, dim=1, keepdim=True).clamp_min(1.0e-12)
    out = torch.zeros(
        (edge_index.shape[1], EDGE_FEATURE_DIM),
        dtype=torch.float32,
        device=current_pos.device,
    )
    out[:, 0:3] = ref_rel
    out[:, 3:4] = ref_len
    out[:, 4:7] = cur_rel
    out[:, 7:8] = cur_len
    out[:, 8:9] = (t["normals"][src] * t["normals"][dst]).sum(1, keepdim=True)
    out[:, 9:10] = 0.5 * (
        t["pressure_mask"][src].float()[:, None]
        + t["pressure_mask"][dst].float()[:, None]
    )
    out[:, 10:11] = (
        t["fixed"][src].float() * t["fixed"][dst].float()
    )[:, None]
    if is_world_edge:
        out[:, 11:12] = cur_len
        out[:, 13] = 1.0
    out[:, 12:13] = (t["leaflet"][src] == t["leaflet"][dst]).float()[:, None]
    return out


def _contact_edges(current_pos, t, cfg):
    if not bool(get_cfg(cfg, "contact.enabled", True)):
        return torch.zeros((2, 0), dtype=torch.long, device=current_pos.device)
    radius = get_cfg(cfg, "contact.radius", None)
    if radius is None:
        src, dst = t["mesh_edges"]
        lengths = torch.linalg.vector_norm(current_pos[dst] - current_pos[src], dim=1)
        radius = float(get_cfg(cfg, "contact.radius_factor", 2.5)) * float(
            lengths.median().item()
        )
    radius = float(radius)
    if radius <= 0.0 or current_pos.shape[0] < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=current_pos.device)

    max_neighbors = int(get_cfg(cfg, "contact.max_neighbors", 32) or 32)
    from physicsnemo.nn.functional import radius_search

    points = current_pos.detach().float()
    count = int(points.shape[0])
    # Warp does not promise that a truncated radius result contains the nearest
    # points. Enumerate all in-radius candidates, then perform an exact GPU top-k
    # so this path matches the CPU cKDTree graph definition.
    query_k = count
    neighbors = radius_search(
        points,
        points,
        radius=radius,
        max_points=query_k,
        return_dists=False,
        return_points=False,
    ).long()
    sources = torch.arange(count, device=points.device)[:, None].expand_as(neighbors)
    adjacency = torch.zeros((count, count), dtype=torch.bool, device=points.device)
    adjacency[t["mesh_edges"][0], t["mesh_edges"][1]] = True
    valid = (neighbors != sources) & ~adjacency[sources, neighbors]
    valid &= ~(
        t["fixed"][sources] & t["fixed"][neighbors]
    )
    if bool(get_cfg(cfg, "contact.different_leaflets_only", True)):
        valid &= t["leaflet"][sources] != t["leaflet"][neighbors]
    distance_sq = ((points[sources] - points[neighbors]) ** 2).sum(dim=-1)
    valid &= distance_sq <= radius * radius

    sorted_neighbors, order = neighbors.sort(dim=1)
    sorted_distance = distance_sq.gather(1, order)
    sorted_valid = valid.gather(1, order)
    duplicate = torch.zeros_like(sorted_valid)
    duplicate[:, 1:] = sorted_neighbors[:, 1:] == sorted_neighbors[:, :-1]
    sorted_distance.masked_fill_(~(sorted_valid & ~duplicate), torch.inf)
    keep_count = min(max_neighbors, query_k)
    best_distance, positions = torch.topk(
        sorted_distance, keep_count, dim=1, largest=False, sorted=True
    )
    best_neighbors = sorted_neighbors.gather(1, positions)
    keep = torch.isfinite(best_distance)
    edge_sources = torch.arange(count, device=points.device)[:, None].expand(
        -1, keep_count
    )[keep]
    edges = torch.stack([edge_sources, best_neighbors[keep]], dim=0)
    max_edges = int(get_cfg(cfg, "contact.max_edges", 200_000))
    return edges[:, :max_edges].long()
