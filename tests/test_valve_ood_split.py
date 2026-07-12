import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.abaqus_export_odb import (
    load_material_sidecars,
    parse_args,
    validate_material_feature_names,
)
from valgraphnet.config import load_config
from valgraphnet.chp_model import CHPGNS
from valgraphnet.data.case import load_case
from valgraphnet.data.dataset import ValveGraphDataset
from valgraphnet.data.valve_ood import (
    SPLIT_STRATEGY,
    build_strict_ood_split,
    validate_case_requirements,
    write_strict_ood_split,
)


def _write_metadata_case(
    root: Path,
    case_id: str,
    geometry_id: str,
    material_id: str,
) -> None:
    case = root / case_id
    case.mkdir(parents=True)
    (case / "metadata.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "geometry_id": geometry_id,
                "material_id": material_id,
            }
        ),
        encoding="utf-8",
    )


def _write_complete_constitutive_case(root: Path) -> None:
    root.mkdir(parents=True)
    nodes = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    steps = 2
    nodal_vector = np.zeros((steps, 4, 3), dtype=np.float32)
    np.save(root / "nodes.npy", nodes)
    np.save(root / "elements.npy", cells)
    np.save(root / "cells.npy", cells)
    np.save(root / "times.npy", np.asarray([0.0, 0.1], dtype=np.float32))
    np.save(root / "pressure.npy", np.zeros((steps,), dtype=np.float32))
    np.save(root / "U.npy", nodal_vector)
    np.save(root / "V.npy", nodal_vector)
    np.save(root / "A.npy", nodal_vector)
    np.save(root / "S.npy", np.zeros((steps, 4, 6), dtype=np.float32))
    np.save(root / "Dm_inv.npy", np.eye(3, dtype=np.float32)[None])
    np.save(root / "reference_volume.npy", np.asarray([[1.0 / 6.0]], dtype=np.float32))
    np.save(
        root / "shape_gradients.npy",
        np.asarray([[[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32),
    )
    np.save(root / "lumped_mass.npy", np.full((4, 1), 1.0 / 24.0, dtype=np.float32))
    np.save(root / "S_cell.npy", np.zeros((steps, 1, 6), dtype=np.float32))
    np.save(root / "S_integration_point.npy", np.zeros((steps, 1, 1, 6), dtype=np.float32))
    np.save(root / "integration_point_mask.npy", np.ones((steps, 1, 1), dtype=bool))
    np.save(root / "LE_cell.npy", np.zeros((steps, 1, 6), dtype=np.float32))
    np.save(root / "material_features.npy", np.asarray([[1.0, 2.0]], dtype=np.float32))
    np.save(root / "density.npy", np.asarray([[1000.0]], dtype=np.float32))
    np.save(root / "fiber_direction.npy", np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    (root / "material.json").write_text(
        json.dumps(
            {
                "model": "anisotropic_hyperelastic",
                "material_feature_names": ["young_modulus", "poisson_ratio"],
            }
        ),
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "case_id": root.name,
                "geometry_id": "geometry-a",
                "material_id": "material-a",
                "material_feature_names": ["young_modulus", "poisson_ratio"],
                "stress_tensor_components": ["S11", "S22", "S33", "S12", "S13", "S23"],
                "strain_tensor_components": [
                    "LE11",
                    "LE22",
                    "LE33",
                    "LE12",
                    "LE13",
                    "LE23",
                ],
            }
        ),
        encoding="utf-8",
    )


def _full_requirements_cfg() -> dict:
    return {
        "data": {
            "requirements": {
                "required_metadata": ["geometry_id", "material_id"],
                "required_files": [
                    "S_cell.npy",
                    "S_integration_point.npy",
                    "integration_point_mask.npy",
                    "material.json",
                    "material_features.npy",
                    "density.npy",
                    "fiber_direction.npy",
                ],
                "full_cell_stress_tensor": True,
                "full_integration_point_stress_tensor": True,
                "full_cell_strain_tensor": True,
                "material_json": True,
                "material_features": True,
                "material_feature_names": True,
                "density": True,
                "fiber_direction": True,
                "stress_tensor_components": [
                    "S11",
                    "S22",
                    "S33",
                    "S12",
                    "S13",
                    "S23",
                ],
                "strain_tensor_components": [
                    "LE11",
                    "LE22",
                    "LE33",
                    "LE12",
                    "LE13",
                    "LE23",
                ],
            }
        }
    }


def test_strict_valve_split_is_deterministic_and_keeps_combinations_together(tmp_path):
    root = tmp_path / "cases"
    combinations = [
        ("g0", "m0"),
        ("g0", "m1"),
        ("g1", "m0"),
        ("g1", "m1"),
        ("g2", "m0"),
        ("g2", "m1"),
    ]
    for index, (geometry, material) in enumerate(reversed(combinations)):
        _write_metadata_case(root, f"case_{index:02d}_a", geometry, material)
        _write_metadata_case(root, f"case_{index:02d}_b", geometry, material)

    first = build_strict_ood_split(
        root, seed=17, validation_fraction=0.2, test_fraction=0.2
    )
    second = build_strict_ood_split(
        root, seed=17, validation_fraction=0.2, test_fraction=0.2
    )

    assert first == second
    assert first["strategy"] == SPLIT_STRATEGY
    assert first["audit"]["combination_disjoint"] is True
    assert all(first[split] for split in ("train", "val", "test"))
    membership = {
        case_id: split
        for split in ("train", "val", "test")
        for case_id in first[split]
    }
    for index in range(len(combinations)):
        assert membership[f"case_{index:02d}_a"] == membership[f"case_{index:02d}_b"]

    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    write_strict_ood_split(root, one, seed=17, validation_fraction=0.2, test_fraction=0.2)
    write_strict_ood_split(root, two, seed=17, validation_fraction=0.2, test_fraction=0.2)
    assert one.read_bytes() == two.read_bytes()


def test_strict_valve_split_rejects_missing_ids_and_insufficient_groups(tmp_path):
    incomplete = tmp_path / "incomplete"
    case = incomplete / "case-a"
    case.mkdir(parents=True)
    (case / "metadata.json").write_text(
        json.dumps({"case_id": "case-a", "geometry_id": "g0"}), encoding="utf-8"
    )
    with pytest.raises(KeyError, match="material_id"):
        build_strict_ood_split(incomplete)

    too_small = tmp_path / "too-small"
    _write_metadata_case(too_small, "case-a", "g0", "m0")
    _write_metadata_case(too_small, "case-b", "g0", "m0")
    with pytest.raises(ValueError, match="at least three distinct"):
        build_strict_ood_split(too_small)


def test_full_tensor_valve_requirements_fail_closed(tmp_path):
    root = tmp_path / "case-a"
    _write_complete_constitutive_case(root)
    cfg = _full_requirements_cfg()
    validate_case_requirements(load_case(root), cfg)
    dataset = ValveGraphDataset(tmp_path, cfg)
    assert [case.case_id for case in dataset.cases] == ["case-a"]

    (root / "fiber_direction.npy").unlink()
    with pytest.raises(ValueError, match="fiber_direction.npy"):
        ValveGraphDataset(tmp_path, cfg)


@pytest.mark.parametrize(
    "metadata_key",
    ["stress_tensor_components", "strain_tensor_components"],
)
def test_valve_requirements_reject_noncanonical_tensor_component_order(
    tmp_path, metadata_key
):
    root = tmp_path / "case-a"
    _write_complete_constitutive_case(root)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[metadata_key][0], metadata[metadata_key][1] = (
        metadata[metadata_key][1],
        metadata[metadata_key][0],
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=metadata_key):
        ValveGraphDataset(tmp_path, _full_requirements_cfg())


def test_valve_requirements_reject_material_feature_schema_drift(tmp_path):
    first = tmp_path / "case-a"
    second = tmp_path / "case-b"
    _write_complete_constitutive_case(first)
    _write_complete_constitutive_case(second)
    metadata_path = second / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["material_feature_names"] = ["poisson_ratio", "young_modulus"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    material_path = second / "material.json"
    material = json.loads(material_path.read_text(encoding="utf-8"))
    material["material_feature_names"] = ["poisson_ratio", "young_modulus"]
    material_path.write_text(json.dumps(material), encoding="utf-8")

    with pytest.raises(ValueError, match="order differs"):
        ValveGraphDataset(tmp_path, _full_requirements_cfg())


def test_valve_requirements_reject_material_feature_name_width_mismatch(tmp_path):
    root = tmp_path / "case-a"
    _write_complete_constitutive_case(root)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["material_feature_names"] = ["young_modulus"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    material_path = root / "material.json"
    material = json.loads(material_path.read_text(encoding="utf-8"))
    material["material_feature_names"] = ["young_modulus"]
    material_path.write_text(json.dumps(material), encoding="utf-8")

    with pytest.raises(ValueError, match="width"):
        ValveGraphDataset(tmp_path, _full_requirements_cfg())


def test_valve_chp_template_requires_cuda_bf16_and_complete_fields():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "valve_chp.full_tensor_ood.yaml")

    assert cfg["training"]["device"] == "cuda"
    assert cfg["training"]["amp"] is True
    assert cfg["training"]["amp_dtype"] == "bfloat16"
    assert cfg["training"]["mechanics_dtype"] == "float32"
    assert cfg["model"]["fiber_order"] > 0
    assert "contact_substeps" not in cfg["model"]
    assert cfg["model"]["contact_iterations"] == 2
    assert cfg["model"]["integration_substeps"] == 1
    assert cfg["model"]["contact_predictor_stop_gradient"] is True
    assert cfg["model"]["contact_force_average"] == "trapezoidal"
    assert cfg["contact"]["refresh_each_iteration"] is True
    assert cfg["dynamics_pretraining"]["phases"] == [
        {"name": "physics_only", "epochs": 4},
        {"name": "residual_warmup", "epochs": 1},
        {"name": "joint", "epochs": 2},
    ]
    assert {
        key: cfg["dynamics_pretraining"][key]
        for key in (
            "physical_graph_lr",
            "pair_head_lr",
            "force_scale_lr",
            "inertia_lr",
            "constitutive_lr",
            "residual_warmup_lr",
            "residual_joint_lr",
        )
    } == {
        "physical_graph_lr": 1.0e-4,
        "pair_head_lr": 5.0e-4,
        "force_scale_lr": 5.0e-4,
        "inertia_lr": 1.0e-3,
        "constitutive_lr": 1.0e-5,
        "residual_warmup_lr": 1.0e-3,
        "residual_joint_lr": 5.0e-4,
    }
    assert cfg["dynamics_pretraining"]["phase_gate"] == {
        "active_acceleration_relative_rmse": 0.95,
        "active_acceleration_cosine": 0.05,
        "teacher_stress_relative_rmse": 0.50,
    }
    assert (
        cfg["validation"]["teacher_stress_minimum_admissible_coverage"]
        == 0.99
    )
    model = CHPGNS(cfg)
    assert model.dynamics_semantics_version == CHPGNS.dynamics_schema_version
    requirements = cfg["data"]["requirements"]
    assert requirements["full_cell_stress_tensor"] is True
    assert requirements["full_integration_point_stress_tensor"] is True
    assert requirements["full_cell_strain_tensor"] is True
    assert requirements["material_json"] is True
    assert requirements["material_features"] is True
    assert requirements["material_feature_names"] is True
    assert requirements["density"] is True
    assert requirements["fiber_direction"] is True
    assert {"geometry_id", "material_id"} <= set(requirements["required_metadata"])
    assert requirements["stress_tensor_components"] == [
        "S11",
        "S22",
        "S33",
        "S12",
        "S13",
        "S23",
    ]
    assert requirements["strain_tensor_components"] == [
        "LE11",
        "LE22",
        "LE33",
        "LE12",
        "LE13",
        "LE23",
    ]


def test_abaqus_export_accepts_explicit_ood_metadata_ids():
    args = parse_args(
        [
            "--odb",
            "case.odb",
            "--out",
            "case-a",
            "--fixed-set",
            "FIXED",
            "--pressure-surface",
            "PRESSURE",
            "--geometry-id",
            "geometry-a",
            "--material-id",
            "material-a",
        ]
    )
    assert args.geometry_id == "geometry-a"
    assert args.material_id == "material-a"


def test_abaqus_export_requires_ordered_material_feature_names():
    assert validate_material_feature_names(["c10", "bulk_modulus"], 2) == [
        "c10",
        "bulk_modulus",
    ]
    with pytest.raises(ValueError, match="must define"):
        validate_material_feature_names(None, 2)
    with pytest.raises(ValueError, match="length"):
        validate_material_feature_names(["c10"], 2)
    with pytest.raises(ValueError, match="unique"):
        validate_material_feature_names(["c10", "c10"], 2)


def test_abaqus_material_feature_names_flow_from_material_json(tmp_path):
    material_path = tmp_path / "material.json"
    material_path.write_text(
        json.dumps(
            {
                "density": 1000.0,
                "fiber_direction": [1.0, 0.0, 0.0],
                "material_features": [2.0, 5.0],
                "material_feature_names": ["c10", "bulk_modulus"],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        material_json=str(material_path),
        density=None,
        density_file=None,
        fiber_direction_file=None,
    )

    material, density, fiber, features, names = load_material_sidecars(
        args, np.asarray([10, 20], dtype=np.int64)
    )

    assert material["material_feature_names"] == ["c10", "bulk_modulus"]
    assert names == ["c10", "bulk_modulus"]
    np.testing.assert_allclose(density, [[1000.0], [1000.0]])
    np.testing.assert_allclose(fiber, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(features, [[2.0, 5.0], [2.0, 5.0]])
