import pytest
import torch

from valgraphnet.hierarchy import (
    HierarchicalScalarVectorProcessor,
    ScalarVectorBlock,
    _single_bistride_assignment,
    build_topology_hierarchy,
    pool_mean,
)


def _chain_edges(count: int) -> torch.Tensor:
    forward = torch.stack([torch.arange(count - 1), torch.arange(1, count)])
    return torch.cat([forward, forward.flip(0)], dim=1)


def test_topology_hierarchy_is_deterministic_and_coarsens_without_geometry():
    edges = _chain_edges(64)
    first = build_topology_hierarchy(64, edges)
    second = build_topology_hierarchy(64, edges)

    assert first.node_counts == [64, 16, 4]
    assert first.coarsening == "topology_bistride"
    assert all(torch.equal(a, b) for a, b in zip(first.assignments, second.assignments))
    assert all(torch.equal(a, b) for a, b in zip(first.edge_indices, second.edge_indices))
    value = torch.arange(64, dtype=torch.float32)[:, None]
    pooled = pool_mean(value, first.assignments[0], first.node_counts[1])
    assert pooled.shape == (16, 1)


def test_bistride_keeps_alternate_bfs_frontiers_and_rejects_non_power_ratio():
    assignment, count = _single_bistride_assignment(10, _chain_edges(10))
    assert count == 5
    assert assignment.unique().numel() == 5
    assert torch.equal(assignment[::2], torch.arange(5))

    with pytest.raises(ValueError, match="powers of two"):
        build_topology_hierarchy(10, _chain_edges(10), ratios=(3,))


def test_scalar_vector_block_is_rotation_equivariant():
    torch.manual_seed(4)
    count = 8
    scalar = torch.randn(count, 12)
    vector = torch.randn(count, 4, 3)
    position = torch.randn(count, 3)
    edges = _chain_edges(count)
    block = ScalarVectorBlock(12, 4).eval()
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    scalar_a, vector_a = block(scalar, vector, edges, position)
    scalar_b, vector_b = block(
        scalar,
        vector @ rotation.T,
        edges,
        position @ rotation.T,
    )

    assert torch.allclose(scalar_a, scalar_b, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(vector_a @ rotation.T, vector_b, atol=2.0e-5, rtol=2.0e-5)


def test_hierarchy_activation_checkpointing_preserves_gradients():
    hierarchy = build_topology_hierarchy(8, _chain_edges(8))
    processor = HierarchicalScalarVectorProcessor(
        scalar_dim=8, vector_dim=3, activation_checkpointing=True
    ).train()
    scalar = torch.randn(8, 8, requires_grad=True)
    vector = torch.randn(8, 3, 3, requires_grad=True)
    position = torch.randn(8, 3)
    scalar_out, vector_out = processor(scalar, vector, position, hierarchy)
    (scalar_out.square().mean() + vector_out.square().mean()).backward()
    assert scalar.grad is not None and torch.isfinite(scalar.grad).all()
    assert vector.grad is not None and torch.isfinite(vector.grad).all()
