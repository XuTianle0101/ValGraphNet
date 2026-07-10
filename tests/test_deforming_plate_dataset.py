import torch

from examples.deforming_plate.dataset import radius_world_edges


def test_radius_world_edges_limits_neighbors_per_source():
    world_pos = torch.tensor(
        [
            [0.00, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.04, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mesh_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    edges = radius_world_edges(
        world_pos,
        radius=0.1,
        mesh_edge_index=mesh_edge_index,
        max_neighbors=2,
    )

    sources, destinations = edges
    counts = torch.bincount(sources, minlength=world_pos.shape[0])
    assert int(counts.max()) <= 2
    assert not torch.any((sources == 0) & (destinations == 1))
    assert not torch.any((sources == 1) & (destinations == 0))


def test_radius_world_edges_neighbor_limit_is_deterministic():
    torch.manual_seed(0)
    world_pos = torch.rand(20, 3)
    mesh_edge_index = torch.zeros((2, 0), dtype=torch.long)

    first = radius_world_edges(world_pos, 2.0, mesh_edge_index, max_neighbors=4)
    second = radius_world_edges(world_pos, 2.0, mesh_edge_index, max_neighbors=4)

    assert torch.equal(first, second)
