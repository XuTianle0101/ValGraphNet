"""Constitutive-consistent Hierarchical Potential Graph Neural Simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from valgraphnet.config import get_cfg
from valgraphnet.hierarchy import HierarchicalScalarVectorProcessor, TopologyHierarchy
from valgraphnet.mechanics import (
    AnalyticPotential,
    assemble_internal_force,
    deformation_gradient,
    invariants,
    project_cell_to_nodes,
    semi_implicit_step,
    von_mises,
)


@dataclass
class CHPStatic:
    reference_position: torch.Tensor
    cells: torch.Tensor
    mesh_edge_index: torch.Tensor
    dm_inv: torch.Tensor
    volume: torch.Tensor
    shape_gradients: torch.Tensor
    lumped_mass: torch.Tensor
    fixed_mask: torch.Tensor
    prescribed_mask: torch.Tensor
    material_features: torch.Tensor
    fiber_direction: torch.Tensor
    hierarchy: TopologyHierarchy

    @property
    def num_nodes(self) -> int:
        return int(self.reference_position.shape[0])


@dataclass
class CHPState:
    position: torch.Tensor
    velocity: torch.Tensor


@dataclass
class PhysicalStep:
    next_position: torch.Tensor
    next_velocity: torch.Tensor
    acceleration: torch.Tensor
    nodal_stress: torch.Tensor
    cell_stress_tensor: torch.Tensor
    internal_force: torch.Tensor
    contact_force: torch.Tensor
    damping_force: torch.Tensor
    residual_force: torch.Tensor
    energy_diagnostics: dict[str, torch.Tensor]

    def legacy_predictions(self, state: CHPState) -> dict[str, torch.Tensor]:
        return {
            "delta_u": self.next_position - state.position,
            "delta_v": self.next_velocity - state.velocity,
            "accel": self.acceleration,
            "stress": self.nodal_stress,
        }


class VectorChannelLinear(nn.Module):
    """Mix vector channels without mixing Cartesian coordinates."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_channels, input_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oc,nci->noi", self.weight, value)


class CellNodeBlock(nn.Module):
    def __init__(self, node_dim: int = 96, cell_dim: int = 64) -> None:
        super().__init__()
        self.cell_update = nn.Sequential(
            nn.Linear(cell_dim + node_dim, cell_dim),
            nn.SiLU(),
            nn.Linear(cell_dim, cell_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + cell_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.cell_norm = nn.LayerNorm(cell_dim)
        self.node_norm = nn.LayerNorm(node_dim)

    def forward(
        self, node: torch.Tensor, cell: torch.Tensor, cells: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_at_cell = node[cells].mean(dim=1)
        cell = self.cell_norm(cell + self.cell_update(torch.cat([cell, node_at_cell], 1)))
        messages = cell[:, None, :].expand(-1, 4, -1)
        node_message = node.new_zeros((node.shape[0], cell.shape[1]))
        node_message.index_add_(0, cells.reshape(-1), messages.reshape(-1, cell.shape[1]))
        degree = torch.bincount(cells.reshape(-1), minlength=node.shape[0]).to(node.dtype)
        node_message = node_message / degree.clamp_min(1.0)[:, None]
        node = self.node_norm(node + self.node_update(torch.cat([node, node_message], 1)))
        return node, cell


class PairForceHeads(nn.Module):
    """Non-negative damping/contact magnitudes assembled as pair forces."""

    def __init__(self, scalar_dim: int) -> None:
        super().__init__()
        symmetric_dim = 2 * scalar_dim + 4
        self.damping = nn.Sequential(
            nn.Linear(symmetric_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, 1)
        )
        self.contact = nn.Sequential(
            nn.Linear(symmetric_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, 2)
        )

    @staticmethod
    def invariants(
        scalar: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        pairs: torch.Tensor,
        reference_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = pairs
        relative = position[dst] - position[src]
        distance = torch.linalg.vector_norm(relative, dim=1, keepdim=True).clamp_min(1.0e-8)
        direction = relative / distance
        relative_velocity = velocity[dst] - velocity[src]
        normal_velocity = (relative_velocity * direction).sum(1, keepdim=True)
        if reference_position is None:
            stretch = torch.zeros_like(distance)
        else:
            reference_length = torch.linalg.vector_norm(
                reference_position[dst] - reference_position[src], dim=1, keepdim=True
            ).clamp_min(1.0e-8)
            stretch = distance / reference_length - 1.0
        features = torch.cat(
            [
                scalar[src] + scalar[dst],
                (scalar[src] - scalar[dst]).abs(),
                distance,
                stretch,
                normal_velocity,
                torch.linalg.vector_norm(relative_velocity, dim=1, keepdim=True),
            ],
            dim=1,
        )
        return features, direction, relative_velocity, normal_velocity

    @staticmethod
    def scatter_pair(force_on_src: torch.Tensor, pairs: torch.Tensor, count: int) -> torch.Tensor:
        result = force_on_src.new_zeros((count, 3))
        result.index_add_(0, pairs[0], force_on_src)
        result.index_add_(0, pairs[1], -force_on_src)
        return result

    def mesh_damping(
        self,
        scalar: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        mesh_pairs: torch.Tensor,
        reference_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mesh_pairs.numel() == 0:
            zero = position.new_zeros((position.shape[0], 3))
            return zero, position.new_zeros(())
        features, direction, _, normal_velocity = self.invariants(
            scalar, position, velocity, mesh_pairs, reference_position
        )
        coefficient = F.softplus(self.damping(features))
        force_src = coefficient * normal_velocity * direction
        force = self.scatter_pair(force_src, mesh_pairs, position.shape[0])
        dissipation = (coefficient * normal_velocity.square()).sum()
        return force, dissipation

    def contact_force(
        self,
        scalar: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        contact_pairs: torch.Tensor,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if contact_pairs.numel() == 0:
            zero = position.new_zeros((position.shape[0], 3))
            scalar_zero = position.new_zeros(())
            return zero, scalar_zero, scalar_zero
        features, direction, relative_velocity, normal_velocity = self.invariants(
            scalar, position, velocity, contact_pairs
        )
        raw = self.contact(features)
        penetration = ((float(radius) - features[:, -4:-3]) / max(float(radius), 1.0e-8)).clamp_min(0.0)
        normal_magnitude = F.softplus(raw[:, :1]) * penetration
        normal_force_src = -normal_magnitude * direction
        tangential_velocity = relative_velocity - normal_velocity * direction
        tangential_coefficient = F.softplus(raw[:, 1:2])
        tangential_force_src = tangential_coefficient * tangential_velocity
        force_src = normal_force_src + tangential_force_src
        force = self.scatter_pair(force_src, contact_pairs, position.shape[0])
        dissipation = (tangential_coefficient * tangential_velocity.square().sum(1, keepdim=True)).sum()
        return force, penetration.max(), dissipation


class CHPGNS(nn.Module):
    """Hierarchical potential simulator whose stress and dynamics share mechanics."""

    checkpoint_schema_version = 2

    def __init__(self, cfg: dict[str, Any], material_dim: int = 0) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        self.scalar_dim = int(model_cfg.get("scalar_dim", 96))
        self.vector_dim = int(model_cfg.get("vector_dim", 16))
        self.cell_dim = int(model_cfg.get("cell_dim", 64))
        self.material_dim = int(material_dim)
        self.contact_radius = float(get_cfg(cfg, "contact.radius", 0.03))
        self.contact_substeps = int(model_cfg.get("contact_substeps", 2))
        self.residual_limit = float(model_cfg.get("residual_force_limit", 0.1))

        self.node_encoder = nn.Sequential(
            nn.Linear(6, self.scalar_dim),
            nn.SiLU(),
            nn.Linear(self.scalar_dim, self.scalar_dim),
            nn.LayerNorm(self.scalar_dim),
        )
        self.vector_encoder = VectorChannelLinear(3, self.vector_dim)
        self.processor = HierarchicalScalarVectorProcessor(
            self.scalar_dim, self.vector_dim
        )
        cell_input_dim = self.scalar_dim + 6 + self.material_dim
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_input_dim, self.cell_dim),
            nn.SiLU(),
            nn.Linear(self.cell_dim, self.cell_dim),
            nn.LayerNorm(self.cell_dim),
        )
        self.cell_blocks = nn.ModuleList(
            [CellNodeBlock(self.scalar_dim, self.cell_dim) for _ in range(4)]
        )
        self.force_heads = PairForceHeads(self.scalar_dim)
        self.residual_channel = VectorChannelLinear(self.vector_dim, 1)
        self.potential = AnalyticPotential(
            order=int(model_cfg.get("potential_order", 2)),
            fiber_order=int(model_cfg.get("fiber_order", 0)),
            inversion_stiffness=float(model_cfg.get("inversion_stiffness", 10.0)),
            minimum_j=float(model_cfg.get("minimum_j", 0.0)),
        )
        self.material_scale = (
            nn.Sequential(nn.Linear(self.material_dim, 32), nn.SiLU(), nn.Linear(32, 1))
            if self.material_dim
            else None
        )

    def forward(
        self,
        static: CHPStatic,
        state: CHPState,
        *,
        contact_pairs: torch.Tensor | None = None,
        external_force: torch.Tensor | None = None,
        dt: float | torch.Tensor = 1.0,
        time_fraction: float | torch.Tensor = 0.0,
        prescribed_position: torch.Tensor | None = None,
        prescribed_velocity: torch.Tensor | None = None,
    ) -> PhysicalStep:
        position = state.position.float()
        velocity = state.velocity.float()
        reference = static.reference_position.float()
        displacement = position - reference
        if external_force is None:
            external_force = torch.zeros_like(position)
        else:
            external_force = external_force.float()
        time = torch.as_tensor(time_fraction, device=position.device, dtype=position.dtype)
        scalar_input = torch.stack(
            [
                static.fixed_mask.float(),
                static.prescribed_mask.float(),
                (~(static.fixed_mask | static.prescribed_mask)).float(),
                torch.log1p(static.lumped_mass.float().reshape(-1)),
                torch.sin(2.0 * torch.pi * time).expand(position.shape[0]),
                torch.cos(2.0 * torch.pi * time).expand(position.shape[0]),
            ],
            dim=1,
        )
        scalar = self.node_encoder(scalar_input)
        vector_input = torch.stack([displacement, velocity, external_force], dim=1)
        vector = self.vector_encoder(vector_input)
        scalar, vector = self.processor(scalar, vector, position, static.hierarchy)

        deformation = deformation_gradient(position, static.cells, static.dm_inv)
        strain = invariants(deformation)
        invariant_features = torch.stack(
            [
                strain.i1_bar - 3.0,
                strain.i2_bar - 3.0,
                strain.j - 1.0,
                torch.log(strain.j.clamp_min(1.0e-8)),
                strain.i1 - 3.0,
                strain.i2 - 3.0,
            ],
            dim=1,
        )
        cell_node = scalar[static.cells].mean(1)
        material = static.material_features.float()
        if material.shape[1] != self.material_dim:
            raise ValueError(
                f"Expected {self.material_dim} material features, got {material.shape[1]}"
            )
        cell = self.cell_encoder(torch.cat([cell_node, invariant_features, material], 1))
        for block in self.cell_blocks:
            scalar, cell = block(scalar, cell, static.cells)

        fiber = static.fiber_direction if self.potential.fiber_order else None
        response = self.potential(deformation, fiber_direction=fiber)
        if self.material_scale is None:
            scale = torch.ones((static.cells.shape[0], 1), device=position.device)
        else:
            scale = F.softplus(self.material_scale(material)) + 1.0e-6
        first_piola = response.first_piola * scale[:, :, None]
        cauchy = response.cauchy_stress * scale[:, :, None]
        energy_density = response.energy_density * scale[:, 0]
        internal = assemble_internal_force(
            first_piola,
            static.cells,
            static.volume,
            static.shape_gradients,
            static.num_nodes,
        )

        mesh_pairs = unique_undirected_pairs(static.mesh_edge_index, static.num_nodes)
        damping, damping_energy = self.force_heads.mesh_damping(
            scalar, position, velocity, mesh_pairs, reference
        )
        if contact_pairs is None:
            contact_pairs = position.new_zeros((2, 0), dtype=torch.long)
        contact_pairs = unique_undirected_pairs(contact_pairs, static.num_nodes)
        contact, max_penetration, contact_dissipation = self.force_heads.contact_force(
            scalar, position, velocity, contact_pairs, self.contact_radius
        )
        raw_residual = self.residual_channel(vector)[:, 0]
        residual_norm = torch.linalg.vector_norm(raw_residual, dim=1, keepdim=True)
        residual = (
            self.residual_limit
            * torch.tanh(residual_norm)
            * raw_residual
            / residual_norm.clamp_min(1.0e-8)
        )
        activity = (
            torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
            + torch.linalg.vector_norm(velocity, dim=1, keepdim=True)
            + torch.linalg.vector_norm(external_force, dim=1, keepdim=True)
        )
        residual = residual * torch.tanh(activity)
        total_force = internal + damping + contact + residual + external_force
        active_prescribed_mask = (
            static.prescribed_mask
            if bool(static.prescribed_mask.any().item())
            else None
        )
        integrated = semi_implicit_step(
            position,
            velocity,
            total_force,
            static.lumped_mass.reshape(-1),
            dt,
            substeps=self.contact_substeps,
            fixed_mask=static.fixed_mask,
            prescribed_mask=active_prescribed_mask,
            prescribed_position=prescribed_position,
            prescribed_velocity=prescribed_velocity,
        )
        cell_vm = von_mises(cauchy)
        nodal_stress = project_cell_to_nodes(
            cell_vm[:, None], static.cells, static.num_nodes, weights=static.volume
        )
        kinetic = 0.5 * (
            static.lumped_mass.reshape(-1, 1) * integrated.velocity.square()
        ).sum()
        potential = (static.volume * energy_density).sum()
        return PhysicalStep(
            next_position=integrated.position,
            next_velocity=integrated.velocity,
            acceleration=integrated.acceleration,
            nodal_stress=nodal_stress,
            cell_stress_tensor=cauchy,
            internal_force=internal,
            contact_force=contact,
            damping_force=damping,
            residual_force=residual,
            energy_diagnostics={
                "potential": potential,
                "kinetic": kinetic,
                "damping_dissipation": damping_energy,
                "contact_dissipation": contact_dissipation,
                "max_penetration": max_penetration,
                "negative_j": response.inversion_barrier.mean(),
                "residual_norm": residual.square().mean().sqrt(),
            },
        )


def unique_undirected_pairs(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    src = torch.minimum(edge_index[0].long(), edge_index[1].long())
    dst = torch.maximum(edge_index[0].long(), edge_index[1].long())
    keep = src != dst
    encoded = torch.unique(src[keep] * int(num_nodes) + dst[keep], sorted=True)
    return torch.stack([encoded // int(num_nodes), encoded % int(num_nodes)], dim=0)


def build_chp_static(case, device: torch.device | str) -> CHPStatic:
    """Move one constitutive case and its cached topology hierarchy to a device."""

    def tensor(value, dtype):
        return torch.from_numpy(np.array(value, copy=True)).to(device=device, dtype=dtype)

    reference = tensor(case.nodes, torch.float32)
    cells = tensor(case.cells, torch.long)
    mesh_edges = tensor(case.mesh_edge_index, torch.long)
    hierarchy = build_case_hierarchy(case).to(device)
    return CHPStatic(
        reference_position=reference,
        cells=cells,
        mesh_edge_index=mesh_edges,
        dm_inv=tensor(case.dm_inv, torch.float32),
        volume=tensor(case.reference_volume.reshape(-1), torch.float32),
        shape_gradients=tensor(case.shape_gradients, torch.float32),
        lumped_mass=tensor(case.lumped_mass.reshape(-1), torch.float32),
        fixed_mask=tensor(case.fixed_mask, torch.bool),
        prescribed_mask=tensor(case.prescribed_mask, torch.bool),
        material_features=tensor(case.material_features, torch.float32),
        fiber_direction=tensor(case.fiber_direction, torch.float32),
        hierarchy=hierarchy,
    )


def build_case_hierarchy(case) -> TopologyHierarchy:
    from valgraphnet.hierarchy import build_topology_hierarchy

    return build_topology_hierarchy(
        case.num_nodes,
        torch.from_numpy(np.array(case.mesh_edge_index, dtype=np.int64, copy=True)),
    )


def radius_contact_pairs(
    position: torch.Tensor,
    mesh_edge_index: torch.Tensor,
    fixed_mask: torch.Tensor,
    radius: float,
    max_neighbors: int = 32,
) -> torch.Tensor:
    """Build exact bounded contact pairs on CUDA using PhysicsNeMo Warp search."""

    if not position.is_cuda:
        raise ValueError("radius_contact_pairs requires CUDA positions")
    from physicsnemo.nn.functional import radius_search

    count = int(position.shape[0])
    points = position.detach().float()
    neighbors = radius_search(
        points,
        points,
        radius=float(radius),
        max_points=count,
        return_dists=False,
        return_points=False,
    ).long()
    sources = torch.arange(count, device=position.device)[:, None].expand_as(neighbors)
    adjacency = torch.zeros((count, count), dtype=torch.bool, device=position.device)
    adjacency[mesh_edge_index[0], mesh_edge_index[1]] = True
    valid = (neighbors != sources) & ~adjacency[sources, neighbors]
    valid &= ~(fixed_mask[sources] & fixed_mask[neighbors])
    distance_sq = ((points[sources] - points[neighbors]) ** 2).sum(-1)
    valid &= distance_sq <= float(radius) ** 2
    distance_sq = distance_sq.masked_fill(~valid, torch.inf)
    keep_count = min(int(max_neighbors), count)
    best_distance, positions = torch.topk(
        distance_sq, keep_count, dim=1, largest=False, sorted=True
    )
    best_neighbors = neighbors.gather(1, positions)
    keep = torch.isfinite(best_distance)
    src = sources[:, :keep_count][keep]
    dst = best_neighbors[keep]
    return unique_undirected_pairs(torch.stack([src, dst]), count)
