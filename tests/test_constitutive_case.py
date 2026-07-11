import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.deforming_plate.convert_to_cases import _write_case
from scripts.abaqus_export_odb import (
    align_cell_values,
    canonical_tensor6,
    export_element_tensor_frames,
)
from valgraphnet.data.case import load_case


def _write_base_case(root, nodes: np.ndarray, elements: np.ndarray, steps: int = 2) -> None:
    root.mkdir()
    num_nodes = nodes.shape[0]
    zeros = np.zeros((steps, num_nodes, 3), dtype=np.float32)
    np.save(root / "nodes.npy", nodes.astype(np.float32))
    np.save(root / "elements.npy", elements.astype(np.int64))
    np.save(root / "times.npy", np.arange(steps, dtype=np.float32))
    np.save(root / "pressure.npy", np.zeros((steps,), dtype=np.float32))
    np.save(root / "U.npy", zeros)
    np.save(root / "V.npy", zeros)
    np.save(root / "A.npy", zeros)
    np.save(root / "S.npy", np.zeros((steps, num_nodes, 1), dtype=np.float32))


def test_load_case_derives_tetra_geometry_and_retains_material_fields(tmp_path):
    root = tmp_path / "tetra"
    nodes = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    _write_base_case(root, nodes, cells)
    np.save(root / "cells.npy", cells)
    np.save(root / "density.npy", np.asarray(2.0, dtype=np.float32))
    np.save(root / "fiber_direction.npy", np.asarray([2.0, 0.0, 0.0], dtype=np.float32))
    np.save(root / "material_features.npy", np.asarray([3.0, 4.0], dtype=np.float32))
    np.save(root / "S_cell.npy", np.ones((2, 1, 6), dtype=np.float32))
    np.save(root / "LE_cell.npy", np.full((2, 1, 6), 0.25, dtype=np.float32))
    np.save(root / "S_integration_point.npy", np.ones((2, 1, 2, 6), dtype=np.float32))
    np.save(root / "integration_point_mask.npy", np.asarray([[True, False]]))
    (root / "material.json").write_text(
        json.dumps({"model": "neo_hookean", "units": "SI"}),
        encoding="utf-8",
    )

    case = load_case(root)

    assert case.num_cells == 1
    assert case.has_constitutive_data
    np.testing.assert_array_equal(case.cells, cells)
    np.testing.assert_allclose(case.dm_inv, np.eye(3, dtype=np.float32)[None])
    np.testing.assert_allclose(case.reference_volume, [[1.0 / 6.0]])
    np.testing.assert_allclose(
        case.shape_gradients,
        [[[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]],
    )
    np.testing.assert_allclose(case.density, [[2.0]])
    np.testing.assert_allclose(case.lumped_mass, np.full((4, 1), 1.0 / 12.0))
    np.testing.assert_allclose(case.fiber_direction, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(case.material_features, [[3.0, 4.0]])
    assert case.cell_stress.shape == (2, 1, 6)
    assert case.cell_strain.shape == (2, 1, 6)
    assert case.integration_point_mask.shape == (2, 1, 2)
    assert case.material["model"] == "neo_hookean"


def test_load_legacy_shell_case_has_shape_safe_empty_constitutive_fields(tmp_path):
    root = tmp_path / "legacy"
    nodes = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    _write_base_case(root, nodes, np.asarray([[0, 1, 2]], dtype=np.int64))

    case = load_case(root)

    assert case.cells.shape == (0, 4)
    assert case.dm_inv.shape == (0, 3, 3)
    assert case.reference_volume.shape == (0, 1)
    assert case.lumped_mass.shape == (3, 1)
    assert case.material_features.shape == (0, 0)
    assert case.density.shape == (0, 1)
    assert case.fiber_direction.shape == (0, 3)
    assert case.cell_stress.shape == (2, 0, 0)


def test_deforming_plate_case_export_keeps_all_stress_channels(tmp_path):
    sequence = SimpleNamespace(
        mesh_pos=np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        ),
        world_pos=np.zeros((3, 4, 3), dtype=np.float32),
        cells=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        node_type=np.asarray([3, 0, 0, 1], dtype=np.int64),
        stress=np.arange(24, dtype=np.float32).reshape(3, 4, 2),
        num_steps=3,
        num_nodes=4,
        sample_id="sample-0",
    )
    sequence.world_pos[:] = sequence.mesh_pos
    root = tmp_path / "converted"

    _write_case(root, "converted", sequence)

    assert np.load(root / "S.npy").shape == (3, 4, 2)
    np.testing.assert_allclose(np.load(root / "Dm_inv.npy"), np.eye(3)[None])
    np.testing.assert_allclose(np.load(root / "reference_volume.npy"), [[1.0 / 6.0]])
    assert np.load(root / "lumped_mass.npy").shape == (4, 1)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2


def test_abaqus_element_tensor_export_preserves_integration_points_and_labels():
    components = ("S11", "S22", "S33", "S12", "S13", "S23")
    values = [
        SimpleNamespace(elementLabel=20, data=[20, 21, 22, 23, 24, 25]),
        SimpleNamespace(elementLabel=10, data=[1, 2, 3, 4, 5, 6]),
        SimpleNamespace(elementLabel=10, data=[3, 4, 5, 6, 7, 8]),
    ]

    class Field:
        componentLabels = components

        def getSubset(self, **_kwargs):
            return SimpleNamespace(values=values)

    frame = SimpleNamespace(fieldOutputs={"S": Field()}, frameValue=0.0)
    odb = SimpleNamespace(steps={"step": SimpleNamespace(frames=[frame])})

    mean, integration, mask = export_element_tensor_frames(
        odb,
        object(),
        np.asarray([10, 20], dtype=np.int64),
        "S",
    )

    assert integration.shape == (1, 2, 2, 6)
    np.testing.assert_allclose(mean[0, 0], [2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(mean[0, 1], [20, 21, 22, 23, 24, 25])
    np.testing.assert_array_equal(mask[0], [[True, True], [True, False]])
    np.testing.assert_allclose(
        canonical_tensor6([1, 2, 3, 4], ("S11", "S22", "S33", "S12"), "S"),
        [1, 2, 3, 4, 0, 0],
    )


def test_cell_sidecars_can_be_aligned_by_element_label():
    labels = np.asarray([20, 10], dtype=np.int64)
    values = np.asarray([[10, 1, 0, 0], [20, 0, 2, 0]], dtype=np.float32)

    aligned = align_cell_values(values, labels, width=3, name="fiber_direction")

    np.testing.assert_allclose(aligned, [[0, 2, 0], [1, 0, 0]])


def test_degenerate_tetrahedron_is_rejected(tmp_path):
    root = tmp_path / "degenerate"
    nodes = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    _write_base_case(root, nodes, cells)
    np.save(root / "cells.npy", cells)

    with pytest.raises(ValueError, match="degenerate tetrahedral"):
        load_case(root)
