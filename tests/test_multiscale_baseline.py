from copy import deepcopy

import pytest
import torch

from valgraphnet.config import load_config
from valgraphnet.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from valgraphnet.multiscale_baseline import build_bistride_topology
from valgraphnet.multiscale_train import validate_multiscale_development_protocol


def _bidirectional_cycle(start: int, count: int) -> torch.Tensor:
    src = torch.arange(start, start + count, dtype=torch.long)
    dst = start + (torch.arange(count, dtype=torch.long) + 1) % count
    return torch.cat(
        [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
    )


def test_two_level_bistride_is_deterministic_and_component_safe():
    edges = torch.cat(
        [_bidirectional_cycle(0, 8), _bidirectional_cycle(8, 8)], dim=1
    )

    first = build_bistride_topology(edges, 16, levels=2)
    second = build_bistride_topology(edges.flip(1), 16, levels=2)

    assert first.node_counts == (16, 8, 4)
    assert first.levels == 2
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.edge_indices, second.edge_indices)
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.retained_ids, second.retained_ids)
    )
    # Each original component contributes four then two retained nodes.  No
    # coarse edge may jump between those components.
    for edge, boundary in zip(first.edge_indices[1:], (4, 2)):
        crosses = (edge[0] < boundary) != (edge[1] < boundary)
        assert not bool(crosses.any())


def test_topology_builder_never_receives_dynamic_contact_edges():
    mesh = _bidirectional_cycle(0, 16)
    contact = torch.tensor([[0, 8], [8, 0]], dtype=torch.long)
    topology = build_bistride_topology(mesh, 16, levels=2)

    fine_pairs = set(map(tuple, topology.edge_indices[0].t().tolist()))
    assert (0, 8) not in fine_pairs
    assert (8, 0) not in fine_pairs
    assert contact.shape[1] == 2  # documents the deliberately excluded pair


def test_multiscale_forward_backward_and_fixed_boundary():
    pyg = pytest.importorskip("torch_geometric")
    pytest.importorskip("physicsnemo.models.meshgraphnet")
    from valgraphnet.multiscale_baseline import MultiscaleDeformingPlateBaseline

    torch.manual_seed(7)
    num_nodes = 16
    mesh = _bidirectional_cycle(0, num_nodes)
    contact = torch.tensor([[0, 8], [8, 0]], dtype=torch.long)
    graph = pyg.data.Data(
        edge_index=torch.cat([mesh, contact], dim=1), num_nodes=num_nodes
    )
    graph.pos = torch.randn(num_nodes, 3)
    graph.reference_pos = graph.pos.clone()
    graph.node_features = torch.randn(num_nodes, NODE_FEATURE_DIM)
    graph.mesh_edge_features = torch.randn(mesh.shape[1], EDGE_FEATURE_DIM)
    graph.world_edge_features = torch.randn(contact.shape[1], EDGE_FEATURE_DIM)
    graph.mesh_edge_count = int(mesh.shape[1])
    graph.fixed_mask = torch.tensor([True] + [False] * (num_nodes - 1))
    graph.case_id = "synthetic-cycle"
    model = MultiscaleDeformingPlateBaseline(
        {
            "model": {
                "num_mesh_levels": 2,
                "processor_size": 1,
                "hidden_dim_processor": 16,
                "hidden_dim_node_encoder": 16,
                "hidden_dim_edge_encoder": 16,
                "hidden_dim_node_decoder": 16,
                "num_layers_bistride": 1,
                "bistride_unet_levels": 1,
                "num_processor_checkpoint_segments": 0,
            },
            "training": {"gpu_hierarchy_cache_size": 1},
        }
    )

    prediction = model(graph)
    loss = (
        prediction["delta_x"].square().mean()
        + prediction["stress_transformed"].square().mean()
    )
    loss.backward()

    assert prediction["delta_x"].shape == (num_nodes, 3)
    assert prediction["stress_transformed"].shape == (num_nodes, 1)
    assert torch.equal(prediction["delta_x"][0], torch.zeros(3))
    assert any(parameter.grad is not None for parameter in model.parameters())
    cached = next(iter(model.hierarchy_cache._cpu.values()))
    assert cached.node_counts == (16, 8, 4)
    assert (0, 8) not in set(map(tuple, cached.edge_indices[0].t().tolist()))


def test_full400_config_is_gpu_bf16_and_val_only_minimax():
    cfg = load_config("configs/deforming_plate_multiscale_mgn.full400.yaml")
    validate_multiscale_development_protocol(cfg)

    assert cfg["training"]["device"] == "cuda"
    assert cfg["training"]["amp"] is True
    assert cfg["training"]["amp_dtype"] == "bfloat16"
    assert cfg["model"]["activation"] == "relu"
    assert cfg["model"]["recompute_activation"] is False
    assert cfg["validation"]["cases"] == 20
    assert cfg["validation"]["steps"] == 399
    assert cfg["validation"]["native_reference_case_selection"] == "even"
    assert cfg["validation"]["native_reference_split"] == "val"

    leaked = deepcopy(cfg)
    leaked["data"]["val_split"] = leaked["data"]["test_split"]
    leaked["validation"]["native_reference_split"] = "test"
    with pytest.raises(ValueError, match="test split"):
        validate_multiscale_development_protocol(leaked)

    shortened = deepcopy(cfg)
    shortened["validation"]["steps"] = 199
    with pytest.raises(ValueError, match="400 frames"):
        validate_multiscale_development_protocol(shortened)
