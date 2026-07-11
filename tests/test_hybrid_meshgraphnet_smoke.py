import pytest

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")
pytest.importorskip("physicsnemo.models.meshgraphnet")


def test_physicsnemo_hybrid_meshgraphnet_pyg_smoke():
    from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

    torch.manual_seed(0)
    num_nodes = 8
    mesh_edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    world_edge_index = torch.tensor(
        [[0, 2, 4, 6], [7, 5, 3, 1]],
        dtype=torch.long,
    )
    graph = pyg.data.Data(
        edge_index=torch.cat([mesh_edge_index, world_edge_index], dim=1),
        num_nodes=num_nodes,
    )
    model = HybridMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        output_dim=2,
        processor_size=1,
        hidden_dim_processor=16,
    )

    out = model(
        torch.randn(num_nodes, 4),
        torch.randn(mesh_edge_index.shape[1], 3),
        torch.randn(world_edge_index.shape[1], 3),
        graph,
    )

    assert out.shape == (num_nodes, 2)


def test_valgraphnet_wrapper_hybrid_forward_smoke():
    from valgraphnet.constants import BASE_OUTPUT_DIM, EDGE_FEATURE_DIM, NODE_FEATURE_DIM
    from valgraphnet.model import ValveGraphNet

    torch.manual_seed(0)
    num_nodes = 6
    mesh_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    world_edge_index = torch.tensor([[0, 5], [5, 0]], dtype=torch.long)
    graph = pyg.data.Data(
        edge_index=torch.cat([mesh_edge_index, world_edge_index], dim=1),
        num_nodes=num_nodes,
    )
    graph.node_features = torch.randn(num_nodes, NODE_FEATURE_DIM)
    graph.mesh_edge_features = torch.randn(mesh_edge_index.shape[1], EDGE_FEATURE_DIM)
    graph.world_edge_features = torch.randn(world_edge_index.shape[1], EDGE_FEATURE_DIM)
    graph.fixed_mask = torch.zeros(num_nodes, dtype=torch.bool)

    model = ValveGraphNet(
        {
            "model": {
                "type": "hybrid",
                "processor_size": 1,
                "hidden_dim_processor": 16,
                "aggregation": "sum",
            }
        },
        output_dim=BASE_OUTPUT_DIM + 1,
    )
    pred = model(graph)

    assert pred["delta_u"].shape == (num_nodes, 3)
    assert pred["delta_v"].shape == (num_nodes, 3)
    assert pred["accel"].shape == (num_nodes, 3)
    assert pred["stress"].shape == (num_nodes, 1)


def test_independent_stress_decoder_migrates_joint_decoder():
    from valgraphnet.constants import BASE_OUTPUT_DIM, EDGE_FEATURE_DIM, NODE_FEATURE_DIM
    from valgraphnet.model import ValveGraphNet

    torch.manual_seed(1)
    output_dim = BASE_OUTPUT_DIM + 1
    base_cfg = {
        "model": {
            "type": "hybrid",
            "processor_size": 1,
            "hidden_dim_processor": 16,
            "decoder_layers": 2,
        }
    }
    joint = ValveGraphNet(base_cfg, output_dim=output_dim)
    split_cfg = {"model": {**base_cfg["model"], "independent_stress_decoder": True}}
    split = ValveGraphNet(split_cfg, output_dim=output_dim)
    split.load_compatible_state_dict(joint.state_dict())

    joint_final = joint.net.node_decoder.model[-1]
    dynamics_final = split.net.node_decoder.model[-1]
    stress_final = split.stress_decoder.model[-1]
    assert torch.equal(dynamics_final.weight, joint_final.weight[:BASE_OUTPUT_DIM])
    assert torch.equal(stress_final.weight, joint_final.weight[BASE_OUTPUT_DIM:])

    num_nodes = 6
    mesh_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    world_edge_index = torch.tensor([[0, 5], [5, 0]], dtype=torch.long)
    graph = pyg.data.Data(
        edge_index=torch.cat([mesh_edge_index, world_edge_index], dim=1),
        num_nodes=num_nodes,
    )
    graph.node_features = torch.randn(num_nodes, NODE_FEATURE_DIM)
    graph.mesh_edge_features = torch.randn(mesh_edge_index.shape[1], EDGE_FEATURE_DIM)
    graph.world_edge_features = torch.randn(world_edge_index.shape[1], EDGE_FEATURE_DIM)
    graph.fixed_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pred = split(graph)
    assert pred["stress"].shape == (num_nodes, 1)


def test_detached_stress_head_does_not_backpropagate_into_processor():
    from valgraphnet.constants import BASE_OUTPUT_DIM, EDGE_FEATURE_DIM, NODE_FEATURE_DIM
    from valgraphnet.model import ValveGraphNet

    graph = pyg.data.Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        num_nodes=4,
    )
    graph.node_features = torch.randn(4, NODE_FEATURE_DIM)
    graph.mesh_edge_features = torch.randn(4, EDGE_FEATURE_DIM)
    graph.world_edge_features = torch.zeros(0, EDGE_FEATURE_DIM)
    graph.fixed_mask = torch.zeros(4, dtype=torch.bool)
    model = ValveGraphNet(
        {
            "model": {
                "type": "hybrid",
                "processor_size": 1,
                "hidden_dim_processor": 16,
                "independent_stress_decoder": True,
                "detach_stress_latent": True,
            }
        },
        output_dim=BASE_OUTPUT_DIM + 1,
    )

    model(graph)["stress"].sum().backward()

    assert model.stress_decoder.model[-1].weight.grad is not None
    shared_grad = model.net.node_encoder.model[0].weight.grad
    assert shared_grad is None or torch.count_nonzero(shared_grad) == 0
