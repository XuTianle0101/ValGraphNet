import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from valgraphnet.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from valgraphnet.data import ValveGraphDataset, collate_valve_graphs


def test_dataset_and_collate_keep_hybrid_edge_order(tmp_path):
    case = tmp_path / "case_001"
    case.mkdir()

    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.05],
            [1.0, 0.0, 0.05],
            [0.0, 1.0, 0.05],
        ],
        dtype=np.float32,
    )
    elements = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    times = np.array([0.0, 0.01], dtype=np.float32)
    zeros = np.zeros((2, 6, 3), dtype=np.float32)
    pressure = np.array([0.0, 10.0], dtype=np.float32)
    leaflet_id = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)

    np.save(case / "nodes.npy", nodes)
    np.save(case / "elements.npy", elements)
    np.save(case / "times.npy", times)
    np.save(case / "pressure.npy", pressure)
    np.save(case / "U.npy", zeros)
    np.save(case / "V.npy", zeros)
    np.save(case / "A.npy", zeros)
    np.save(case / "S.npy", np.zeros((2, 6, 1), dtype=np.float32))
    np.save(case / "fixed_mask.npy", np.array([1, 1, 0, 1, 1, 0], dtype=bool))
    np.save(case / "pressure_mask.npy", np.array([0, 0, 0, 1, 1, 1], dtype=bool))
    np.save(case / "leaflet_id.npy", leaflet_id)
    np.save(case / "node_type.npy", np.array([0, 0, 3, 1, 1, 1], dtype=np.int64))
    np.save(case / "thickness.npy", np.ones(6, dtype=np.float32))
    (case / "metadata.json").write_text(
        json.dumps({"case_id": "case_001", "source": "DeepMind deforming_plate"}),
        encoding="utf-8",
    )

    cfg = {"contact": {"enabled": True, "radius": 0.08, "different_leaflets_only": True}}
    dataset = ValveGraphDataset(tmp_path, cfg=cfg)
    item = dataset[0]

    assert item.node_features.shape == (6, NODE_FEATURE_DIM)
    assert item.mesh_edge_features.shape[1] == EDGE_FEATURE_DIM
    assert item.world_edge_features.shape[1] == EDGE_FEATURE_DIM
    assert item.mesh_edge_count > 0
    assert item.world_edge_count > 0

    batch = collate_valve_graphs([item, item])
    assert batch.edge_index.shape[1] == batch.mesh_edge_count + batch.world_edge_count
    assert batch.mesh_edge_features.shape[0] == batch.mesh_edge_count
    assert batch.world_edge_features.shape[0] == batch.world_edge_count
    assert batch.target.shape[1] == 10
    assert int(batch.prescribed_mask.sum()) == 6

