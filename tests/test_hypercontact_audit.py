import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from scripts.audit_hypercontact3d import audit_hypercontact3d_dataset
from valgraphnet.calculix_results import convert_case


FIXTURES = Path(__file__).parent / "fixtures" / "hypercontact"


def _write_auditable_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    raw_case = raw / "cases" / "hc3d-fixture"
    raw_case.mkdir(parents=True)
    for name in ("model.inp", "model.frd"):
        shutil.copyfile(FIXTURES / name, raw_case / name)
    shutil.copyfile(FIXTURES / "model.dat.txt", raw_case / "model.dat")
    (raw_case / "model.sta").write_text("analysis completed\n", encoding="utf-8")
    deck = raw_case / "model.inp"
    entry = {
        "case_id": "hc3d-fixture",
        "deck": "cases/hc3d-fixture/model.inp",
        "deck_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
        "expected_outputs": [
            "cases/hc3d-fixture/model.frd",
            "cases/hc3d-fixture/model.dat",
            "cases/hc3d-fixture/model.sta",
        ],
        "parameters": {
            "material": {
                "c10_pa": 200000.0,
                "poisson_ratio": 0.45,
                "density_kg_m3": 1200.0,
            },
            "load": {"indentation_m": 0.001},
            "mesh": {"nx": 1, "ny": 1, "nz": 1},
        },
        "derived": {
            "imposed_indenter_displacement_m": -0.2,
            "indenter_density_kg_m3": 1000.0,
            "indenter_shell_thickness_m": 0.01,
            "step_duration": 1.0,
        },
        "benchmark_provenance": {
            "benchmark": "HyperContact-3D-test",
            "config_sha256": "1" * 64,
            "generator_version": "1.9-test",
            "schema_version": 1,
            "units": "SI",
        },
        "split": "validation",
    }
    root = tmp_path / "converted"
    case_dir = root / entry["case_id"]
    convert_case(raw, entry, case_dir)

    times = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    pressure = np.load(case_dir / "pressure.npy", allow_pickle=False)
    np.save(case_dir / "pressure.npy", (times * pressure[-1]).astype(np.float32))
    for name in ("U.npy", "S.npy", "S_cell.npy", "S_integration_point.npy"):
        original = np.load(case_dir / name, allow_pickle=False)
        expanded = times.reshape((-1,) + (1,) * (original.ndim - 1)) * original[-1]
        np.save(case_dir / name, expanded.astype(np.float32))
    displacement = np.load(case_dir / "U.npy", allow_pickle=False)
    velocity = np.gradient(displacement, times, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    np.save(case_dir / "times.npy", times)
    np.save(case_dir / "V.npy", velocity.astype(np.float32))
    np.save(case_dir / "A.npy", acceleration.astype(np.float32))
    force = np.zeros_like(displacement)
    force[:, 3, 2] = times
    np.save(case_dir / "solver_nodal_force.npy", force)
    np.save(
        case_dir / "integration_point_mask.npy",
        np.ones((101, 1, 1), dtype=bool),
    )

    metadata_path = case_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["num_frames"] = 101
    metadata["benchmark_provenance"] = entry["benchmark_provenance"]
    metadata["derived_solver_parameters"]["contact_formulation"] = (
        "surface_to_surface"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "splits.json").write_text(
        json.dumps({"validation": [entry["case_id"]]}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "case_count": 1,
        "split_counts": {"validation": 1},
        "cases": [metadata],
    }
    (root / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config = {
        "data": {
            "root": str(root),
            "split_file": str(root / "splits.json"),
            "requirements": {
                "required_metadata": [
                    "time_semantics",
                    "benchmark_provenance.generator_version",
                    "derived_solver_parameters.contact_formulation",
                ],
                "required_files": [
                    "nodes.npy",
                    "cells.npy",
                    "U.npy",
                    "V.npy",
                    "A.npy",
                    "S_cell.npy",
                    "S_integration_point.npy",
                    "integration_point_mask.npy",
                    "solver_nodal_force.npy",
                ],
                "num_frames": 101,
                "explicit_contact_surface_mask": True,
                "prescribed_contact_surface": True,
                "zero_pressure_mask": True,
                "time_semantics": (
                    "quasi-static normalized step time, not physical transient time"
                ),
                "full_cell_stress_tensor": True,
                "full_integration_point_stress_tensor": True,
                "material_json": True,
                "material_features": True,
                "material_feature_names": True,
                "density": True,
                "stress_tensor_components": [
                    "S11",
                    "S22",
                    "S33",
                    "S12",
                    "S13",
                    "S23",
                ],
            },
        }
    }
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, case_dir, config_path


def test_audit_accepts_complete_dataset_and_records_provenance(tmp_path):
    root, _, config_path = _write_auditable_fixture(tmp_path)

    report = audit_hypercontact3d_dataset(root, config_path)

    assert report["status"] == "passed"
    assert report["summary"]["case_count"] == 1
    assert report["summary"]["minimum_j"] > 0.0
    assert report["coverage"]["exact"]
    assert all(report["checks"].values())
    assert report["provenance"]["generator_versions"] == ["1.9-test"]
    assert len(report["provenance"]["requirements_sha256"]) == 64
    assert report["cases"][0]["fixed_state_bit_exact"]
    assert report["cases"][0]["prescribed_ramp_within_float32_tolerance"]


def test_audit_fails_closed_for_zero_force_and_nonpositive_j(tmp_path):
    root, case_dir, config_path = _write_auditable_fixture(tmp_path)
    force = np.load(case_dir / "solver_nodal_force.npy", allow_pickle=False)
    np.save(case_dir / "solver_nodal_force.npy", np.zeros_like(force))
    displacement = np.load(case_dir / "U.npy", allow_pickle=False)
    displacement[-1, 3, 2] = -2.0
    np.save(case_dir / "U.npy", displacement)

    report = audit_hypercontact3d_dataset(root, config_path)

    assert report["status"] == "failed"
    assert not report["checks"]["solver_force_nonzero_all_cases"]
    assert not report["checks"]["positive_j_all_frames_cells"]
    errors = "\n".join(report["cases"][0]["errors"])
    assert "identically zero" in errors
    assert "non-positive" in errors


def test_audit_rejects_case_assigned_to_multiple_splits(tmp_path):
    root, _, config_path = _write_auditable_fixture(tmp_path)
    (root / "splits.json").write_text(
        json.dumps(
            {
                "validation": ["hc3d-fixture"],
                "test_id": ["hc3d-fixture"],
            }
        ),
        encoding="utf-8",
    )

    report = audit_hypercontact3d_dataset(root, config_path)

    assert report["status"] == "failed"
    assert not report["checks"]["split_case_manifest_exact"]
    assert any("occurs in both" in error for error in report["dataset_errors"])
