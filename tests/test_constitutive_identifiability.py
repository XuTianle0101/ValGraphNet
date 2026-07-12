from __future__ import annotations

from copy import deepcopy
import json
import os
from types import SimpleNamespace

import pytest
import torch

import valgraphnet.constitutive_identifiability as identifiability
from valgraphnet.config import load_config
from valgraphnet.constitutive_identifiability import (
    DIAGNOSTIC_TYPE,
    SCIENTIFIC_DISCLAIMER,
    CellScalarStressMLP,
    StressErrorSums,
    TrainOnlyStatistics,
    development_case_dirs,
    even_indices,
    objective_invariant_features,
    predict_nodal_scalar_stress,
    teacher_frame_indices,
    validate_control_config,
    volume_weighted_nodal_projection,
)
from valgraphnet.mechanics import precompute_tetrahedra, project_cell_to_nodes


def _tetrahedra():
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 3], [4, 1, 3, 2]], dtype=torch.long)
    reference = precompute_tetrahedra(nodes, cells)
    return nodes, cells, reference


def test_objective_features_are_fp32_zero_at_identity_and_rotation_invariant():
    nodes, cells, reference = _tetrahedra()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        identity, _ = objective_invariant_features(nodes, cells, reference.dm_inv)
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated, _ = objective_invariant_features(
        nodes @ rotation.T, cells, reference.dm_inv
    )

    assert identity.dtype == torch.float32
    assert rotated.dtype == torch.float32
    assert torch.allclose(identity, torch.zeros_like(identity), atol=1.0e-6)
    assert torch.allclose(rotated, identity, atol=1.0e-6)


def test_batch_projection_is_exactly_the_chp_volume_weighted_formula_fp32():
    nodes, cells, reference = _tetrahedra()
    values = torch.tensor([[[2.0], [8.0]], [[3.0], [5.0]]], dtype=torch.bfloat16)
    projected = volume_weighted_nodal_projection(
        values, cells, reference.volume, len(nodes)
    )
    expected = torch.stack(
        [
            project_cell_to_nodes(
                frame.float(), cells, len(nodes), weights=reference.volume
            )
            for frame in values
        ]
    )

    assert projected.dtype == torch.float32
    assert torch.equal(projected, expected)


def test_direct_decoder_is_nonnegative_differentiable_and_explicitly_not_a_potential():
    nodes, cells, reference = _tetrahedra()
    features, _ = objective_invariant_features(nodes, cells, reference.dm_inv)
    statistics = TrainOnlyStatistics(
        feature_mean=torch.zeros(3),
        feature_std=torch.ones(3),
        stress_rms=10.0,
        stress_mean=5.0,
        cell_samples=2,
        nodal_samples=5,
        requested_frames=1,
        admissible_frames=1,
    )
    model = CellScalarStressMLP(hidden_dim=8, hidden_layers=2)
    cell, nodal = predict_nodal_scalar_stress(
        model, features, statistics, cells, reference.volume, len(nodes)
    )
    nodal.sum().backward()

    assert DIAGNOSTIC_TYPE == "direct_nonnegative_cell_scalar_stress_decoder"
    assert "not a scalar energy potential" in SCIENTIFIC_DISCLAIMER
    assert torch.all(cell >= 0.0)
    assert torch.all(nodal >= 0.0)
    assert cell.dtype == nodal.dtype == torch.float32
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert statistics.json_dict()["feature_mean"] == [0.0, 0.0, 0.0]


def test_pooled_and_per_frame_p95_metrics_use_physical_sums():
    sums = StressErrorSums()
    sums.update(
        torch.tensor([1.0, 1.0, 3.0, 5.0]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    metrics = sums.metrics()

    assert metrics["teacher_stress_pooled_relative_rmse"] == pytest.approx(
        (2.0 / 30.0) ** 0.5
    )
    assert metrics["teacher_stress_per_frame_p95_relative_rmse"] == pytest.approx(
        0.25
    )
    assert metrics["teacher_stress_rmse"] == pytest.approx((2.0 / 4.0) ** 0.5)
    assert metrics["teacher_stress_relative_rmse"] == metrics[
        "teacher_stress_pooled_relative_rmse"
    ]
    assert metrics["teacher_stress_p95_relative_rmse"] == metrics[
        "teacher_stress_per_frame_p95_relative_rmse"
    ]
    assert metrics["evaluated_nodes"] == 4.0


def test_even_case_and_teacher_frame_selection_are_endpoint_inclusive():
    assert even_indices(100, 20) == [
        0,
        5,
        10,
        16,
        21,
        26,
        31,
        36,
        42,
        47,
        52,
        57,
        63,
        68,
        73,
        78,
        83,
        89,
        94,
        99,
    ]
    frames = teacher_frame_indices(400, 16)
    assert len(frames) == 16
    assert frames[0] == 1
    assert frames[-1] == 399


def test_protocol_config_is_cuda_bf16_train_only_and_fixed_val20():
    cfg = load_config("configs/deforming_plate_constitutive_identifiability.yaml")
    validate_control_config(cfg)

    assert cfg["training"]["device"] == "cuda"
    assert cfg["training"]["amp_dtype"] == "bfloat16"
    assert cfg["training"]["checkpoint_selection"] == "fixed_final_epoch"
    assert cfg["validation"] == {
        "cases": 20,
        "frames": 16,
        "case_selection": "even",
    }

    cpu = deepcopy(cfg)
    cpu["training"]["device"] = "cpu"
    with pytest.raises(ValueError, match="CUDA-only"):
        validate_control_config(cpu)
    leaked = deepcopy(cfg)
    leaked["data"]["val_split"] = leaked["data"]["test_split"]
    with pytest.raises(ValueError, match="must be distinct"):
        validate_control_config(leaked)
    selected_on_val = deepcopy(cfg)
    selected_on_val["training"]["checkpoint_selection"] = "best_validation"
    with pytest.raises(ValueError, match="forbidden"):
        validate_control_config(selected_on_val)


def test_development_resolution_never_requires_or_returns_a_test_case(tmp_path):
    case_root = tmp_path / "cases"
    case_root.mkdir()
    train_ids = ["train_000", "train_001"]
    val_ids = [f"val_{index:03d}" for index in range(20)]
    test_ids = ["test_must_not_be_opened"]
    for case_id in (*train_ids, *val_ids):
        (case_root / case_id).mkdir()
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}),
        encoding="utf-8",
    )
    cfg = load_config("configs/deforming_plate_constitutive_identifiability.yaml")
    cfg["data"]["root"] = str(case_root)
    cfg["data"]["split_file"] = str(split_file)
    cfg["training"]["cases"] = 2

    train, val, audit = development_case_dirs(cfg)

    assert [path.name for path in train] == train_ids
    assert [path.name for path in val] == val_ids
    assert not (case_root / test_ids[0]).exists()
    assert audit["test_cases_loaded"] == 0
    assert "selected_test_case_ids" not in audit


def test_provenance_manifest_hashes_only_selected_development_arrays(tmp_path):
    train = tmp_path / "train_000"
    val = tmp_path / "val_000"
    test = tmp_path / "test_must_not_be_hashed"
    for root in (train, val, test):
        root.mkdir()
        for name in identifiability.CONTROL_INPUT_ARRAYS:
            (root / name).write_bytes(f"{root.name}:{name}".encode())

    manifest = identifiability._development_input_manifest([train], [val])
    identifiability._verify_manifest_files_unchanged(manifest)

    assert [case["case_id"] for case in manifest["cases"]] == [
        "train_000",
        "val_000",
    ]
    assert manifest["test_case_arrays_hashed"] == 0
    assert len(manifest["aggregate_sha256"]) == 64
    (train / "S.npy").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed during run"):
        identifiability._verify_manifest_files_unchanged(manifest)


def test_train_manifest_can_be_frozen_without_opening_validation_arrays(tmp_path):
    train = tmp_path / "train_000"
    validation = tmp_path / "val_must_not_be_opened"
    train.mkdir()
    validation.mkdir()
    for name in identifiability.CONTROL_INPUT_ARRAYS:
        (train / name).write_bytes(name.encode())

    manifest = identifiability._development_input_manifest([train], ())

    assert [case["case_id"] for case in manifest["cases"]] == ["train_000"]
    assert not any(validation.iterdir())


def test_frozen_statistics_round_trip_is_cpu_and_exact():
    original = TrainOnlyStatistics(
        feature_mean=torch.tensor([1.0, 2.0, 3.0], device="cpu"),
        feature_std=torch.tensor([0.1, 0.2, 0.3], device="cpu"),
        stress_rms=12.0,
        stress_mean=5.0,
        cell_samples=20,
        nodal_samples=10,
        requested_frames=8,
        admissible_frames=7,
    )
    restored = TrainOnlyStatistics.from_state_dict(original.state_dict())
    assert restored.json_dict() == original.json_dict()
    assert restored.feature_mean.device.type == "cpu"


def test_admissibility_matches_inclusive_chp_gate_boundaries():
    state = SimpleNamespace(
        j=torch.tensor([[0.01, 1.0], [0.009, 1.0]]),
        i2_bar=torch.tensor([[3.0, 1.0e5], [3.0, 4.0]]),
    )
    valid = identifiability._admissible_frames(
        state,
        {"training": {"minimum_j": 0.01, "maximum_i2_bar": 1.0e5}},
    )
    assert valid.tolist() == [True, False]


@pytest.mark.parametrize("unsafe", ["../escape", "nested/case", "C:\\escape"])
def test_split_rejects_unsafe_or_duplicate_case_ids(tmp_path, unsafe):
    root = tmp_path / "cases"
    root.mkdir()
    val_ids = [f"val_{index:03d}" for index in range(20)]
    for case_id in val_ids:
        (root / case_id).mkdir()
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps({"train": [unsafe], "val": val_ids, "test": ["test_000"]}),
        encoding="utf-8",
    )
    cfg = load_config("configs/deforming_plate_constitutive_identifiability.yaml")
    cfg["data"]["root"] = str(root)
    cfg["data"]["split_file"] = str(split)
    cfg["training"]["cases"] = 1
    with pytest.raises(ValueError, match="unsafe case ids"):
        development_case_dirs(cfg)

    split.write_text(
        json.dumps(
            {
                "train": ["train_000", "train_000"],
                "val": val_ids,
                "test": ["test_000"],
            }
        ),
        encoding="utf-8",
    )
    cfg["training"]["cases"] = 2
    with pytest.raises(ValueError, match="duplicate case ids"):
        development_case_dirs(cfg)


def test_source_manifest_freezes_config_and_split_protocol_files(tmp_path):
    config = tmp_path / "diagnostic.yaml"
    split = tmp_path / "splits.json"
    config.write_text("seed: 42\n", encoding="utf-8")
    split.write_text('{"train": [], "val": [], "test": []}\n', encoding="utf-8")
    manifest = identifiability._source_file_manifest((config, split))

    frozen_paths = {record["path"] for record in manifest["files"]}
    assert str(config.resolve()) in frozen_paths
    assert str(split.resolve()) in frozen_paths
    identifiability._verify_manifest_files_unchanged(manifest)
    original_stat = config.stat()
    config.write_text("seed: 43\n", encoding="utf-8")
    os.utime(
        config,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    with pytest.raises(RuntimeError, match="changed during run"):
        identifiability._verify_manifest_files_unchanged(manifest)
