import json

import torch

from examples.deforming_plate.dataset import radius_world_edges
from examples.deforming_plate.preprocess import preprocess_cache_is_compatible


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


def test_preprocess_cache_rejects_stale_graph_settings(tmp_path):
    cfg = {
        "data": {
            "train_split": "train",
            "val_split": "valid",
            "test_split": "test",
            "num_training_samples": 2,
            "num_validation_samples": 1,
            "num_test_samples": 1,
            "num_training_time_steps": 4,
            "num_validation_time_steps": 3,
            "num_test_time_steps": 2,
            "noise_std": 0.003,
        },
        "graph": {"world_edge_radius": 0.03, "max_world_neighbors": 32},
    }
    specs = {
        "train": ("train", 2, 4),
        "val": ("valid", 1, 3),
        "test": ("test", 1, 2),
    }
    signature = {
        "world_edge_radius": 0.03,
        "max_world_neighbors": 32,
        "noise_std": 0.003,
    }
    for split, (source, sequences, steps) in specs.items():
        split_dir = tmp_path / split
        split_dir.mkdir()
        manifest = {
            "source_split": source,
            "num_sequences": sequences,
            "num_samples": sequences * (steps - 1),
            "cache_signature": signature,
        }
        (split_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "edge_stats.pt").touch()
    (tmp_path / "node_stats.pt").touch()

    assert preprocess_cache_is_compatible(cfg, tmp_path)
    cfg["graph"]["max_world_neighbors"] = 16
    assert not preprocess_cache_is_compatible(cfg, tmp_path)
