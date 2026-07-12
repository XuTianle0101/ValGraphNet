from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import valgraphnet.force_identifiability as force_id
from valgraphnet.config import load_config
from valgraphnet.force_identifiability import (
    FORCE_DIAGNOSTIC_TYPE,
    ForceCase,
    ForceMetricSums,
    development_case_dirs,
    eligible_force_nodes,
    evaluate_force_case,
    positive_inverse_inertia,
    single_step_acceleration_target,
    transition_frame_indices,
    validate_force_config,
)
from valgraphnet.mechanics import precompute_tetrahedra


def test_positive_inverse_inertia_and_pooled_metrics_recover_exact_scale():
    q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    target = 0.25 * q
    alpha = positive_inverse_inertia(q, target)
    sums = ForceMetricSums()
    sums.update(q, target)
    metrics = sums.metrics(float(alpha))

    assert float(alpha) == pytest.approx(0.25)
    assert metrics["zero_baseline_relative_rmse"] == 1.0
    assert metrics["force_cosine"] == pytest.approx(1.0)
    assert metrics["positive_scale_relative_rmse"] == pytest.approx(0.0)
    assert metrics["positive_scale_prediction_to_target_rms_ratio"] == pytest.approx(1.0)


def test_negative_force_correlation_clamps_scale_to_zero_without_hiding_cosine():
    q = torch.tensor([[1.0, -2.0, 3.0]])
    target = -2.0 * q
    alpha = positive_inverse_inertia(q, target)
    sums = ForceMetricSums()
    sums.update(q, target)
    metrics = sums.metrics(float(alpha))

    assert float(alpha) == 0.0
    assert metrics["unconstrained_inverse_inertia_alpha"] == pytest.approx(-2.0)
    assert metrics["force_cosine"] == pytest.approx(-1.0)
    assert metrics["positive_scale_relative_rmse"] == 1.0


def test_single_step_target_and_fixed_transition_frames_match_dp_export():
    current = torch.tensor([[1.0, 2.0, 3.0]])
    following = torch.tensor([[1.5, 1.0, 4.0]])
    torch.testing.assert_close(
        single_step_acceleration_target(current, following),
        following - current,
    )
    with pytest.raises(ValueError, match="exactly dt=1"):
        single_step_acceleration_target(current, following, dt=0.5)
    assert transition_frame_indices(400, 8) == [
        0,
        57,
        114,
        171,
        227,
        284,
        341,
        398,
    ]
    assert transition_frame_indices(400, 16) == [
        0,
        27,
        53,
        80,
        106,
        133,
        159,
        186,
        212,
        239,
        265,
        292,
        318,
        345,
        371,
        398,
    ]


def test_eligible_mask_excludes_constraints_and_radius_contact_moving_nodes():
    position = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.031, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    fixed = torch.tensor([False, True, False, False, False])
    prescribed = torch.tensor([True, False, False, False, False])
    eligible, contact = eligible_force_nodes(position, fixed, prescribed, 0.03)

    assert eligible.tolist() == [False, False, False, True, True]
    assert contact.tolist() == [False, False, True, False, False]


class _ConstantInternalForce:
    def constitutive_fields(self, static, position, *, deformation=None):
        acceleration = torch.tensor(
            [1.0, -2.0, 0.5], device=position.device, dtype=position.dtype
        )
        internal = static.lumped_mass[:, None] * acceleration
        return SimpleNamespace(internal_force=internal)


def _cpu_force_fixture() -> ForceCase:
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    reference = precompute_tetrahedra(nodes, cells)
    target = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    velocity = np.stack(
        [np.zeros((4, 3), np.float32), np.tile(target, (4, 1)), np.tile(2 * target, (4, 1))]
    )
    return ForceCase(
        case_id="fixture",
        root=Path("fixture"),
        nodes=nodes.numpy(),
        cells=cells.numpy(),
        dm_inv=reference.dm_inv.numpy(),
        volume=reference.volume.numpy(),
        shape_gradients=reference.shape_gradients.numpy(),
        lumped_mass=reference.lumped_mass.numpy(),
        fixed_mask=np.zeros(4, dtype=bool),
        prescribed_mask=np.zeros(4, dtype=bool),
        times=np.arange(3, dtype=np.float32),
        pressure=np.zeros(3, dtype=np.float32),
        displacement=np.zeros((3, 4, 3), dtype=np.float32),
        velocity=velocity,
    )


def test_cpu_fixture_evaluates_mechanics_contract_without_formal_cpu_fallback():
    sums, coverage = evaluate_force_case(
        _ConstantInternalForce(),
        _cpu_force_fixture(),
        [0, 1],
        device=torch.device("cpu"),
        contact_radius=0.03,
        minimum_j=0.01,
        maximum_i1_bar=1.0e4,
        maximum_i2_bar=1.0e5,
    )
    metrics = sums.metrics(sums.positive_alpha)

    assert FORCE_DIAGNOSTIC_TYPE == "frozen_potential_positive_inverse_inertia"
    assert sums.positive_alpha == pytest.approx(1.0)
    assert metrics["positive_scale_relative_rmse"] == pytest.approx(0.0)
    assert coverage.evaluated_frames == 2
    assert coverage.eligible_nodes == 8


def test_force_case_rejects_non_unit_time_step_and_nonzero_pressure():
    fixture = _cpu_force_fixture()
    non_unit_dt = replace(
        fixture, times=np.asarray([0.0, 1.0, 2.5], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="strictly dt=1"):
        force_id._validate_force_case(non_unit_dt)

    pressured = replace(
        fixture, pressure=np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="finite zero pressure"):
        force_id._validate_force_case(pressured)


def test_formal_config_is_cuda_only_fixed_val20_and_single_step_semantics():
    cfg = load_config("configs/deforming_plate_force_identifiability.yaml")
    validate_force_config(cfg)
    assert cfg["diagnostic"]["device"] == "cuda"
    assert cfg["diagnostic"]["target_semantics"] == "single_global_semi_implicit_dt1"
    assert cfg["validation"]["cases"] == 20

    cpu = deepcopy(cfg)
    cpu["diagnostic"]["device"] = "cpu"
    with pytest.raises(ValueError, match="CUDA-only"):
        validate_force_config(cpu)
    two_substeps = deepcopy(cfg)
    two_substeps["diagnostic"]["target_semantics"] = "two_substeps"
    with pytest.raises(ValueError, match="target_semantics"):
        validate_force_config(two_substeps)
    selected = deepcopy(cfg)
    selected["validation"]["frames"] = selected["validation"]["frames"][:-1]
    with pytest.raises(ValueError, match="validation.frames"):
        validate_force_config(selected)


def test_development_resolution_reports_ids_but_never_opens_test_content(tmp_path):
    root = tmp_path / "cases"
    root.mkdir()
    train_ids = ["train_000", "train_001"]
    val_ids = [f"val_{index:03d}" for index in range(20)]
    test_ids = ["test_must_not_exist"]
    for case_id in (*train_ids, *val_ids):
        (root / case_id).mkdir()
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}),
        encoding="utf-8",
    )
    cfg = load_config("configs/deforming_plate_force_identifiability.yaml")
    cfg["data"]["root"] = str(root)
    cfg["data"]["split_file"] = str(split)
    cfg["fit"]["cases"] = 2

    train, validation, audit = development_case_dirs(cfg)

    assert [path.name for path in train] == train_ids
    assert [path.name for path in validation] == val_ids
    assert audit["test_case_arrays_loaded"] == 0
    assert audit["test_content_accessed"] is False
    assert not (root / test_ids[0]).exists()


def test_input_manifest_hashes_only_explicit_development_cases(tmp_path):
    train = tmp_path / "train_000"
    validation = tmp_path / "val_000"
    test = tmp_path / "test_must_not_be_hashed"
    for root in (train, validation, test):
        root.mkdir()
        for name in force_id.FORCE_INPUT_ARRAYS:
            (root / name).write_bytes(f"{root.name}:{name}".encode())

    train_manifest = force_id._input_manifest("train", [train])
    val_manifest = force_id._input_manifest("validation", [validation])
    force_id._verify_manifest_unchanged(train_manifest)
    force_id._verify_manifest_unchanged(val_manifest)

    assert [case["case_id"] for case in train_manifest["cases"]] == ["train_000"]
    assert [case["case_id"] for case in val_manifest["cases"]] == ["val_000"]
    assert train_manifest["test_case_arrays_hashed"] == 0
    assert val_manifest["test_case_arrays_hashed"] == 0
    assert "test" not in train_manifest["aggregate_sha256"]
