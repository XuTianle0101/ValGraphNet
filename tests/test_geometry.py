import numpy as np

from valgraphnet.geometry import (
    ContactConfig,
    build_contact_edges,
    compute_node_normals_areas,
    mesh_edges_from_elements,
)


def test_mesh_edges_from_quad_are_bidirectional_unique():
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    edge_index = mesh_edges_from_elements(elements)

    assert edge_index.shape == (2, 8)
    directed = {tuple(edge) for edge in edge_index.T.tolist()}
    assert (0, 1) in directed
    assert (1, 0) in directed
    assert (3, 0) in directed
    assert (0, 3) in directed


def test_normals_and_areas_for_unit_square():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

    normals, areas = compute_node_normals_areas(nodes, elements)

    assert np.allclose(normals, np.array([[0.0, 0.0, 1.0]] * 4), atol=1.0e-6)
    assert np.isclose(areas.sum(), 1.0)


def test_contact_edges_skip_mesh_edges_and_same_leaflet():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.05],
            [1.0, 0.0, 0.05],
        ],
        dtype=np.float32,
    )
    mesh_edges = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
    leaflet_id = np.array([1, 1, 2, 2], dtype=np.int64)

    edge_index = build_contact_edges(
        current_pos=nodes,
        leaflet_id=leaflet_id,
        mesh_edge_index=mesh_edges,
        cfg=ContactConfig(enabled=True, radius=0.1, different_leaflets_only=True),
    )

    directed = {tuple(edge) for edge in edge_index.T.tolist()}
    assert (0, 2) in directed
    assert (2, 0) in directed
    assert (0, 1) not in directed
    assert (2, 3) not in directed

