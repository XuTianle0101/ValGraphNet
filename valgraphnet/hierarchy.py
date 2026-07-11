"""Topology-only graph hierarchy and rotation-equivariant scalar/vector blocks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass
class TopologyHierarchy:
    """Static fine-to-coarse assignments and directed edges at every level."""

    edge_indices: list[torch.Tensor]
    assignments: list[torch.Tensor]
    node_counts: list[int]

    def to(self, device: torch.device | str) -> "TopologyHierarchy":
        return TopologyHierarchy(
            edge_indices=[edge.to(device) for edge in self.edge_indices],
            assignments=[assignment.to(device) for assignment in self.assignments],
            node_counts=self.node_counts,
        )


def build_topology_hierarchy(
    num_nodes: int,
    edge_index: torch.Tensor,
    ratios: tuple[int, ...] = (4, 4),
) -> TopologyHierarchy:
    """Build deterministic BFS coarsenings without geometry-based shortcut edges."""

    edges = edge_index.detach().long().cpu()
    hierarchy_edges = [_coalesce_directed(edges, int(num_nodes))]
    assignments: list[torch.Tensor] = []
    counts = [int(num_nodes)]
    for ratio in ratios:
        assignment, coarse_count = _bfs_assignment(
            counts[-1], hierarchy_edges[-1], max(int(ratio), 2)
        )
        assignments.append(assignment)
        hierarchy_edges.append(_coarse_edges(hierarchy_edges[-1], assignment, coarse_count))
        counts.append(coarse_count)
    return TopologyHierarchy(hierarchy_edges, assignments, counts)


def pool_mean(value: torch.Tensor, assignment: torch.Tensor, count: int) -> torch.Tensor:
    """Mean-pool an arbitrary trailing-dimensional tensor."""

    assignment = assignment.to(value.device)
    out = value.new_zeros((int(count), *value.shape[1:]))
    out.index_add_(0, assignment, value)
    degree = torch.bincount(assignment, minlength=int(count)).to(value.dtype)
    shape = (int(count),) + (1,) * (value.ndim - 1)
    return out / degree.clamp_min(1.0).reshape(shape)


def unpool(value: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    return value[assignment.to(value.device)]


class ScalarVectorBlock(nn.Module):
    """Equivariant message passing with invariant scalar filters."""

    def __init__(self, scalar_dim: int = 96, vector_dim: int = 16) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        invariant_dim = 2 * self.scalar_dim + 1 + 4 * self.vector_dim
        hidden = max(self.scalar_dim, 64)
        self.message = nn.Sequential(
            nn.Linear(invariant_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.scalar_dim + self.vector_dim),
        )
        self.vector_mix = nn.Linear(self.vector_dim, self.vector_dim, bias=False)
        self.scalar_update = nn.Sequential(
            nn.Linear(2 * self.scalar_dim + self.vector_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.scalar_dim),
        )
        self.vector_gate = nn.Sequential(
            nn.Linear(self.scalar_dim, self.vector_dim), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(self.scalar_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return scalar, vector
        src, dst = edge_index.long()
        relative = position[dst] - position[src]
        distance = torch.linalg.vector_norm(relative, dim=1, keepdim=True).clamp_min(1.0e-8)
        direction = relative / distance
        src_vector = vector[src]
        dst_vector = vector[dst]
        invariants = torch.cat(
            [
                scalar[src],
                scalar[dst],
                distance,
                torch.linalg.vector_norm(src_vector, dim=-1),
                torch.linalg.vector_norm(dst_vector, dim=-1),
                (src_vector * direction[:, None, :]).sum(-1),
                (dst_vector * direction[:, None, :]).sum(-1),
            ],
            dim=1,
        )
        raw = self.message(invariants)
        scalar_message = raw[:, : self.scalar_dim].to(scalar.dtype)
        direction_weight = raw[:, self.scalar_dim :].to(vector.dtype)
        mixed = self.vector_mix(src_vector.transpose(1, 2)).transpose(1, 2).to(vector.dtype)
        vector_message = (
            mixed + direction_weight[:, :, None] * direction[:, None, :]
        ).to(vector.dtype)

        scalar_agg = scalar.new_zeros(scalar.shape)
        vector_agg = vector.new_zeros(vector.shape)
        scalar_agg.index_add_(0, dst, scalar_message)
        vector_agg.index_add_(0, dst, vector_message)
        degree = torch.bincount(dst, minlength=scalar.shape[0]).to(scalar.dtype).clamp_min(1.0)
        scalar_agg = scalar_agg / degree[:, None]
        vector_agg = vector_agg / degree[:, None, None]
        vector_norm = torch.linalg.vector_norm(vector_agg, dim=-1)
        scalar_delta = self.scalar_update(torch.cat([scalar, scalar_agg, vector_norm], dim=1))
        scalar_new = self.norm(scalar + scalar_delta)
        gate = self.vector_gate(scalar_new)[:, :, None]
        vector_new = vector + gate * vector_agg
        return scalar_new, vector_new


class HierarchicalScalarVectorProcessor(nn.Module):
    """Two fine and four coarse blocks followed by two fine refinement blocks."""

    def __init__(
        self,
        scalar_dim: int = 96,
        vector_dim: int = 16,
        *,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.fine_in = nn.ModuleList(
            [ScalarVectorBlock(scalar_dim, vector_dim) for _ in range(2)]
        )
        self.coarse_one = nn.ModuleList(
            [ScalarVectorBlock(scalar_dim, vector_dim) for _ in range(2)]
        )
        self.coarse_two = nn.ModuleList(
            [ScalarVectorBlock(scalar_dim, vector_dim) for _ in range(2)]
        )
        self.fine_out = nn.ModuleList(
            [ScalarVectorBlock(scalar_dim, vector_dim) for _ in range(2)]
        )
        self.coarse_one_fuse = nn.Linear(2 * scalar_dim, scalar_dim)
        self.fine_fuse = nn.Linear(2 * scalar_dim, scalar_dim)

    def _blocks(
        self,
        blocks: nn.ModuleList,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def run_pair(
            pair: nn.ModuleList,
            scalar_value: torch.Tensor,
            vector_value: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            for block in pair:
                scalar_value, vector_value = block(
                    scalar_value, vector_value, edge_index, position
                )
            return scalar_value, vector_value

        # Every list currently contains two blocks. Keeping the grouping generic
        # makes the memory policy stable if a later ablation changes the depth.
        for start in range(0, len(blocks), 2):
            pair = blocks[start : start + 2]
            if self.activation_checkpointing and self.training:
                scalar, vector = checkpoint(
                    lambda s, v, pair=pair: run_pair(pair, s, v),
                    scalar,
                    vector,
                    use_reentrant=False,
                )
            else:
                scalar, vector = run_pair(pair, scalar, vector)
        return scalar, vector

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        position: torch.Tensor,
        hierarchy: TopologyHierarchy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edges = [edge.to(scalar.device) for edge in hierarchy.edge_indices]
        assignments = [value.to(scalar.device) for value in hierarchy.assignments]
        scalar, vector = self._blocks(
            self.fine_in, scalar, vector, edges[0], position
        )

        scalar_one = pool_mean(scalar, assignments[0], hierarchy.node_counts[1])
        vector_one = pool_mean(vector, assignments[0], hierarchy.node_counts[1])
        position_one = pool_mean(position, assignments[0], hierarchy.node_counts[1])
        scalar_one, vector_one = self._blocks(
            self.coarse_one, scalar_one, vector_one, edges[1], position_one
        )

        scalar_two = pool_mean(scalar_one, assignments[1], hierarchy.node_counts[2])
        vector_two = pool_mean(vector_one, assignments[1], hierarchy.node_counts[2])
        position_two = pool_mean(position_one, assignments[1], hierarchy.node_counts[2])
        scalar_two, vector_two = self._blocks(
            self.coarse_two, scalar_two, vector_two, edges[2], position_two
        )

        scalar_one = self.coarse_one_fuse(
            torch.cat([scalar_one, unpool(scalar_two, assignments[1])], dim=1)
        )
        vector_one = vector_one + unpool(vector_two, assignments[1])
        scalar = self.fine_fuse(
            torch.cat([scalar, unpool(scalar_one, assignments[0])], dim=1)
        )
        vector = vector + unpool(vector_one, assignments[0])
        scalar, vector = self._blocks(
            self.fine_out, scalar, vector, edges[0], position
        )
        return scalar, vector


def _bfs_assignment(
    num_nodes: int, edge_index: torch.Tensor, ratio: int
) -> tuple[torch.Tensor, int]:
    adjacency = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        if src != dst:
            adjacency[int(src)].add(int(dst))
            adjacency[int(dst)].add(int(src))
    order: list[int] = []
    seen = [False] * num_nodes
    for root in range(num_nodes):
        if seen[root]:
            continue
        queue = deque([root])
        seen[root] = True
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in sorted(adjacency[node]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
    representatives = order[::ratio] or [0]
    assignment = torch.full((num_nodes,), -1, dtype=torch.long)
    queue: deque[int] = deque()
    for coarse, node in enumerate(representatives):
        assignment[node] = coarse
        queue.append(node)
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if assignment[neighbor] < 0:
                assignment[neighbor] = assignment[node]
                queue.append(neighbor)
    for node in range(num_nodes):
        if assignment[node] < 0:
            assignment[node] = len(representatives)
            representatives.append(node)
    return assignment, len(representatives)


def _coarse_edges(
    edge_index: torch.Tensor, assignment: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    src = assignment[edge_index[0]]
    dst = assignment[edge_index[1]]
    keep = src != dst
    return _coalesce_directed(torch.stack([src[keep], dst[keep]]), num_nodes)


def _coalesce_directed(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    encoded = edge_index[0].long() * int(num_nodes) + edge_index[1].long()
    encoded = torch.unique(encoded, sorted=True)
    return torch.stack([encoded // int(num_nodes), encoded % int(num_nodes)], dim=0)
