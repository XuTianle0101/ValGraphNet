"""Custom batching for PhysicsNeMo HybridMeshGraphNet."""

from __future__ import annotations

import torch


def collate_valve_graphs(items: list):
    """Batch graph samples while keeping all mesh edges before all world edges."""

    if not items:
        raise ValueError("Cannot collate an empty batch")
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("torch-geometric is required for graph batching") from exc

    node_offset = 0
    mesh_edges = []
    world_edges = []
    node_features = []
    mesh_edge_features = []
    world_edge_features = []
    targets = []
    fixed_masks = []
    pressure_masks = []
    nodal_areas = []
    batch_vec = []
    steps = []
    dts = []
    case_ids = []

    for batch_id, item in enumerate(items):
        mesh_count = int(item.mesh_edge_count)
        world_count = int(item.world_edge_count)
        if item.edge_index.shape[1] != mesh_count + world_count:
            raise ValueError("edge_index length does not match mesh/world edge counts")

        mesh_edges.append(item.edge_index[:, :mesh_count] + node_offset)
        world_edges.append(item.edge_index[:, mesh_count:] + node_offset)
        node_features.append(item.node_features)
        mesh_edge_features.append(item.mesh_edge_features)
        world_edge_features.append(item.world_edge_features)
        targets.append(item.target)
        fixed_masks.append(item.fixed_mask)
        pressure_masks.append(item.pressure_mask)
        nodal_areas.append(item.nodal_area)
        batch_vec.append(torch.full((item.num_nodes,), batch_id, dtype=torch.long))
        steps.append(int(item.step))
        dts.append(float(item.dt))
        case_ids.append(str(item.case_id))
        node_offset += int(item.num_nodes)

    batched_mesh_edges = _cat_edges(mesh_edges)
    batched_world_edges = _cat_edges(world_edges)
    edge_index = torch.cat([batched_mesh_edges, batched_world_edges], dim=1)

    data = Data(edge_index=edge_index, num_nodes=node_offset)
    data.node_features = torch.cat(node_features, dim=0)
    data.mesh_edge_features = _cat_features(mesh_edge_features, items[0].mesh_edge_features)
    data.world_edge_features = _cat_features(world_edge_features, items[0].world_edge_features)
    data.target = torch.cat(targets, dim=0)
    data.fixed_mask = torch.cat(fixed_masks, dim=0)
    data.pressure_mask = torch.cat(pressure_masks, dim=0)
    data.nodal_area = torch.cat(nodal_areas, dim=0)
    data.batch = torch.cat(batch_vec, dim=0)
    data.step = torch.tensor(steps, dtype=torch.long)
    data.dt = torch.tensor(dts, dtype=torch.float32)
    data.case_id = case_ids
    data.mesh_edge_count = int(data.mesh_edge_features.shape[0])
    data.world_edge_count = int(data.world_edge_features.shape[0])
    return data


def _cat_edges(edges: list[torch.Tensor]) -> torch.Tensor:
    non_empty = [edge for edge in edges if edge.numel() > 0]
    if not non_empty:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.cat(non_empty, dim=1)


def _cat_features(features: list[torch.Tensor], template: torch.Tensor) -> torch.Tensor:
    non_empty = [feat for feat in features if feat.numel() > 0]
    if not non_empty:
        return template.new_zeros((0, template.shape[1]))
    return torch.cat(non_empty, dim=0)

