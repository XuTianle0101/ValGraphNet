"""Two-level topology-bi-stride MeshGraphNet comparison baseline.

The neural network is PhysicsNeMo's :class:`BiStrideMeshGraphNet`.  Its
optional dataset helper depends on ``sparse_dot_mkl``, so this module builds
the same kind of static, alternate-BFS topology hierarchy directly.  Contact
edges are deliberately present only in the fine MeshGraphNet processor; they
never participate in coarsening.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from valgraphnet.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


@dataclass(frozen=True)
class BiStrideTopology:
    """CPU topology tensors consumed by PhysicsNeMo's bi-stride U-Net."""

    edge_indices: tuple[torch.Tensor, ...]
    retained_ids: tuple[torch.Tensor, ...]
    node_counts: tuple[int, ...]

    @property
    def levels(self) -> int:
        return len(self.retained_ids)

    def to(
        self, device: torch.device | str
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return (
            [edge.to(device=device, non_blocking=True) for edge in self.edge_indices],
            [ids.to(device=device, non_blocking=True) for ids in self.retained_ids],
        )


def build_bistride_topology(
    mesh_edge_index: torch.Tensor,
    num_nodes: int,
    *,
    levels: int = 2,
    reference_position: torch.Tensor | None = None,
) -> BiStrideTopology:
    """Build deterministic topology-only alternate-frontier coarse graphs.

    Each pass retains the smaller BFS parity in every connected component.
    Coarse edges connect retained vertices separated by at most two fine
    edges, matching the adjacency-squaring construction used by BSMS-GNN.
    Returned retained ids are relative to their immediate parent level, as
    required by ``BiStrideMeshGraphNet``.
    """

    if int(num_nodes) <= 0:
        raise ValueError("num_nodes must be positive")
    if int(levels) <= 0:
        raise ValueError("levels must be positive")
    edge = _canonical_undirected_edges(
        mesh_edge_index.detach().long().cpu(), int(num_nodes)
    )
    position = None
    if reference_position is not None:
        position = reference_position.detach().float().cpu()
        if tuple(position.shape) != (int(num_nodes), 3):
            raise ValueError("reference_position must have shape [num_nodes, 3]")
    edge_indices = [edge]
    retained_ids: list[torch.Tensor] = []
    node_counts = [int(num_nodes)]
    for _ in range(int(levels)):
        retained = _alternate_frontier_ids(
            edge, node_counts[-1], reference_position=position
        )
        edge = _squared_restricted_edges(edge, retained, node_counts[-1])
        if position is not None:
            position = position[retained]
        retained_ids.append(retained)
        edge_indices.append(edge)
        node_counts.append(int(retained.numel()))
    return BiStrideTopology(
        edge_indices=tuple(edge_indices),
        retained_ids=tuple(retained_ids),
        node_counts=tuple(node_counts),
    )


class BiStrideHierarchyCache:
    """Static CPU cache plus a bounded device LRU for trajectory rollouts."""

    def __init__(self, *, levels: int = 2, device_capacity: int = 3) -> None:
        self.levels = int(levels)
        self.device_capacity = max(int(device_capacity), 1)
        self._cpu: dict[tuple[str, int, int], BiStrideTopology] = {}
        self._device: OrderedDict[
            tuple[tuple[str, int, int], str],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = OrderedDict()

    def get(
        self,
        *,
        case_id: str,
        mesh_edge_index: torch.Tensor,
        reference_position: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        key = (str(case_id), int(num_nodes), int(mesh_edge_index.shape[1]))
        topology = self._cpu.get(key)
        if topology is None:
            topology = build_bistride_topology(
                mesh_edge_index,
                int(num_nodes),
                levels=self.levels,
                reference_position=reference_position,
            )
            self._cpu[key] = topology
        device_key = (key, str(device))
        cached = self._device.pop(device_key, None)
        if cached is None:
            cached = topology.to(device)
        self._device[device_key] = cached
        while len(self._device) > self.device_capacity:
            self._device.popitem(last=False)
        return cached

    def clear_device(self) -> None:
        self._device.clear()


class MultiscaleDeformingPlateBaseline(nn.Module):
    """Fair four-output MGN with one two-level topology bi-stride U-Net."""

    output_dim = 4
    hierarchy_levels = 2

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        configured_levels = int(model_cfg.get("num_mesh_levels", 2))
        if configured_levels != self.hierarchy_levels:
            raise ValueError(
                "the comparison baseline is fixed to exactly two coarse mesh levels"
            )
        try:
            from physicsnemo.models.meshgraphnet import BiStrideMeshGraphNet
        except ImportError as exc:
            raise ImportError(
                "PhysicsNeMo with BiStrideMeshGraphNet is required for the "
                "two-level baseline"
            ) from exc
        hidden = int(model_cfg.get("hidden_dim_processor", 128))
        activation = str(model_cfg.get("activation", "relu"))
        recompute_activation = bool(
            model_cfg.get("recompute_activation", False)
        )
        if recompute_activation and activation.lower() not in {"silu", "swish"}:
            raise ValueError(
                "PhysicsNeMo activation recomputation requires SiLU; disable it "
                "when matching the ReLU fair baseline"
            )
        self.net = BiStrideMeshGraphNet(
            input_dim_nodes=NODE_FEATURE_DIM,
            input_dim_edges=EDGE_FEATURE_DIM,
            output_dim=self.output_dim,
            processor_size=int(model_cfg.get("processor_size", 15)),
            mlp_activation_fn=activation,
            num_layers_node_processor=int(model_cfg.get("node_layers", 2)),
            num_layers_edge_processor=int(model_cfg.get("edge_layers", 2)),
            num_mesh_levels=self.hierarchy_levels,
            bistride_pos_dim=3,
            num_layers_bistride=int(model_cfg.get("num_layers_bistride", 2)),
            bistride_unet_levels=int(model_cfg.get("bistride_unet_levels", 1)),
            hidden_dim_processor=hidden,
            hidden_dim_node_encoder=int(
                model_cfg.get("hidden_dim_node_encoder", hidden)
            ),
            num_layers_node_encoder=int(model_cfg.get("node_encoder_layers", 2)),
            hidden_dim_edge_encoder=int(
                model_cfg.get("hidden_dim_edge_encoder", hidden)
            ),
            num_layers_edge_encoder=int(model_cfg.get("edge_encoder_layers", 2)),
            hidden_dim_node_decoder=int(
                model_cfg.get("hidden_dim_node_decoder", 128)
            ),
            num_layers_node_decoder=int(model_cfg.get("decoder_layers", 2)),
            aggregation=str(model_cfg.get("aggregation", "sum")),
            do_concat_trick=bool(model_cfg.get("do_concat_trick", False)),
            num_processor_checkpoint_segments=int(
                model_cfg.get("num_processor_checkpoint_segments", 0)
            ),
            recompute_activation=recompute_activation,
        )
        self.hierarchy_cache = BiStrideHierarchyCache(
            levels=self.hierarchy_levels,
            device_capacity=int(
                cfg.get("training", {}).get("gpu_hierarchy_cache_size", 3)
            ),
        )

    def forward(self, data) -> dict[str, torch.Tensor]:
        if not hasattr(data, "pos") or data.pos is None:
            raise ValueError(
                "multiscale graphs require unnormalized current coordinates in data.pos"
            )
        if data.pos.ndim != 2 or tuple(data.pos.shape) != (int(data.num_nodes), 3):
            raise ValueError("data.pos must have shape [num_nodes, 3]")
        if data.pos.device != data.node_features.device:
            raise ValueError("data.pos and neural features must share a device")
        if not hasattr(data, "reference_pos") or tuple(data.reference_pos.shape) != (
            int(data.num_nodes),
            3,
        ):
            raise ValueError("data.reference_pos must have shape [num_nodes, 3]")
        if data.reference_pos.device != data.node_features.device:
            raise ValueError("reference coordinates and neural features must share a device")
        mesh_count = int(data.mesh_edge_count)
        if mesh_count <= 0 or mesh_count > int(data.edge_index.shape[1]):
            raise ValueError("mesh_edge_count is inconsistent with edge_index")
        case_id = getattr(data, "case_id", None)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("one-trajectory multiscale graphs require a string case_id")
        mesh_edges = data.edge_index[:, :mesh_count]
        ms_edges, ms_ids = self.hierarchy_cache.get(
            case_id=case_id,
            mesh_edge_index=mesh_edges,
            reference_position=data.reference_pos,
            num_nodes=int(data.num_nodes),
            device=data.node_features.device,
        )
        edge_features = torch.cat(
            [data.mesh_edge_features, data.world_edge_features], dim=0
        )
        if int(edge_features.shape[0]) != int(data.edge_index.shape[1]):
            raise ValueError("edge features are not aligned with mesh/contact edges")
        output = self.net(
            data.node_features,
            edge_features,
            data,
            ms_edges=ms_edges,
            ms_ids=ms_ids,
        )
        delta_x = output[:, :3]
        fixed = data.fixed_mask[:, None]
        return {
            "delta_x": torch.where(fixed, torch.zeros_like(delta_x), delta_x),
            "stress_transformed": output[:, 3:4],
        }


def _canonical_undirected_edges(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("mesh_edge_index must have shape [2, E]")
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(num_nodes):
            raise ValueError("mesh edge index is outside the node range")
        src, dst = edge_index
        keep = src != dst
        edge_index = torch.cat(
            [
                torch.stack([src[keep], dst[keep]], dim=0),
                torch.stack([dst[keep], src[keep]], dim=0),
            ],
            dim=1,
        )
    return _coalesce_with_isolated_self_loops(edge_index, int(num_nodes))


def _alternate_frontier_ids(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    reference_position: torch.Tensor | None,
) -> torch.Tensor:
    adjacency = _adjacency(edge_index, num_nodes)
    seen = [False] * int(num_nodes)
    retained: list[int] = []
    for root in range(int(num_nodes)):
        if seen[root]:
            continue
        # Discover the component first, then use the vertex nearest its
        # reference-space centroid as the BSMS parity seed.  Node id is the
        # deterministic fallback when positions were not supplied.
        queue: deque[int] = deque([root])
        seen[root] = True
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        seed = min(component)
        if reference_position is not None:
            component_ids = torch.tensor(component, dtype=torch.long)
            points = reference_position[component_ids]
            center = points.mean(dim=0, keepdim=True)
            distance_to_center = torch.linalg.vector_norm(points - center, dim=1)
            minimum = distance_to_center.min()
            candidates = component_ids[distance_to_center == minimum]
            seed = int(candidates.min())
        distance = {seed: 0}
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        even = [node for node in component if distance[node] % 2 == 0]
        odd = [node for node in component if distance[node] % 2 == 1]
        retained.extend(even if len(even) <= len(odd) or not odd else odd)
    return torch.tensor(sorted(retained), dtype=torch.long)


def _squared_restricted_edges(
    edge_index: torch.Tensor,
    retained: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    adjacency = _adjacency(edge_index, num_nodes)
    retained_list = retained.tolist()
    remap = {old: new for new, old in enumerate(retained_list)}
    pairs: list[tuple[int, int]] = []
    for old_src in retained_list:
        reachable = set(adjacency[old_src])
        for middle in adjacency[old_src]:
            reachable.update(adjacency[middle])
        new_src = remap[old_src]
        for old_dst in sorted(reachable):
            if old_dst != old_src and old_dst in remap:
                pairs.append((new_src, remap[old_dst]))
    if pairs:
        coarse = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    else:
        coarse = torch.zeros((2, 0), dtype=torch.long)
    return _coalesce_with_isolated_self_loops(coarse, len(retained_list))


def _adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
    adjacency = [set() for _ in range(int(num_nodes))]
    for src, dst in edge_index.t().tolist():
        if int(src) != int(dst):
            adjacency[int(src)].add(int(dst))
            adjacency[int(dst)].add(int(src))
    return adjacency


def _coalesce_with_isolated_self_loops(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    if edge_index.numel():
        encoded = edge_index[0].long() * int(num_nodes) + edge_index[1].long()
        encoded = torch.unique(encoded, sorted=True)
        edge = torch.stack(
            [encoded // int(num_nodes), encoded % int(num_nodes)], dim=0
        )
    else:
        edge = torch.zeros((2, 0), dtype=torch.long)
    degree = torch.bincount(edge[0], minlength=int(num_nodes))
    isolated = torch.nonzero(degree == 0, as_tuple=False).flatten()
    if isolated.numel():
        edge = torch.cat([edge, torch.stack([isolated, isolated])], dim=1)
        encoded = edge[0] * int(num_nodes) + edge[1]
        encoded = torch.unique(encoded, sorted=True)
        edge = torch.stack(
            [encoded // int(num_nodes), encoded % int(num_nodes)], dim=0
        )
    return edge.contiguous()
