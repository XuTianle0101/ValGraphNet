import math

import pytest
import torch

from valgraphnet.fair_train import (
    _fair_validation_due,
    FAIR_CHECKPOINT_SCHEMA_VERSION,
    FAIR_MODEL_FAMILY,
    ROLLOUT_METRIC_KEYS,
    _checkpoint_payload,
    fair_one_step_loss,
    load_native_reference,
    minimax_checkpoint_score,
    one_start_per_trajectory,
)
from valgraphnet.normalization import Normalizers
from valgraphnet.stress_transform import AsinhStressTransform


class _Case:
    def __init__(self, steps):
        self.num_steps = steps


def test_fair_rollout_validation_cadence_includes_final_epoch():
    assert not _fair_validation_due(1, 10, 3)
    assert _fair_validation_due(3, 10, 3)
    assert _fair_validation_due(10, 10, 3)
    assert not _fair_validation_due(10, 10, 0)


def test_one_uniform_start_per_trajectory_is_deterministic_and_valid():
    cases = [_Case(400), _Case(20), _Case(2)]
    first = one_start_per_trajectory(cases, epoch=3, seed=42)
    second = one_start_per_trajectory(cases, epoch=3, seed=42)

    assert first == second
    assert sorted(case_index for case_index, _ in first) == [0, 1, 2]
    assert all(0 <= start < cases[case_index].num_steps - 1 for case_index, start in first)


def test_fair_loss_has_only_delta_and_asinh_stress_objectives():
    transform = AsinhStressTransform.fit([torch.tensor([[0.0], [10.0], [100.0]])])
    target_delta = torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    scale = torch.tensor([1.0, 2.0, 3.0])
    target_stress = torch.tensor([[10.0], [100.0]])
    moving = torch.tensor([True, False])
    prediction = {
        "delta_x": target_delta / scale,
        "stress_transformed": transform.transform(target_stress),
    }

    loss, metrics = fair_one_step_loss(
        prediction,
        target_delta,
        target_stress,
        moving,
        scale,
        transform,
        {"loss": {"peak_weight": 0.5}},
    )

    assert loss == 0
    assert metrics["delta_rmse"] == 0
    assert set(metrics) == {
        "total",
        "delta",
        "stress",
        "stress_base",
        "stress_peak",
        "delta_rmse",
    }


def test_four_metric_minimax_blocks_compensating_improvements():
    native = {key: 1.0 for key in ROLLOUT_METRIC_KEYS}
    balanced = {key: 0.8 for key in ROLLOUT_METRIC_KEYS}
    stress_regression = {**balanced, "stress_p95_relative_rmse": 1.1}

    assert minimax_checkpoint_score(balanced, native) == pytest.approx(0.8)
    assert minimax_checkpoint_score(stress_regression, native) == pytest.approx(1.1)
    assert math.isinf(
        minimax_checkpoint_score(
            balanced,
            {**native, "stress_p95_relative_rmse": 0.0},
        )
    )


def test_checkpoint_declares_physical_output_contract_and_gpu_precision():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW([parameter])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    normalizers = Normalizers(
        node_mean=torch.zeros(1),
        node_std=torch.ones(1),
        mesh_edge_mean=torch.zeros(1),
        mesh_edge_std=torch.ones(1),
        world_edge_mean=torch.zeros(1),
        world_edge_std=torch.ones(1),
        target_scale=torch.ones(10),
    )
    stress = AsinhStressTransform.fit([torch.tensor([[0.0], [1.0]])])
    reference = {key: 1.0 for key in ROLLOUT_METRIC_KEYS}
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        {},
        normalizers,
        stress,
        reference,
        reference,
        epoch=1,
        score=1.0,
        best_score=1.0,
    )

    assert payload["schema_version"] == FAIR_CHECKPOINT_SCHEMA_VERSION
    assert payload["model_family"] == FAIR_MODEL_FAMILY
    assert payload["training_device"] == "cuda"
    assert payload["training_precision"] == "bfloat16"
    assert payload["state_integration"].startswith("velocity_and_acceleration_derived")
    assert payload["checkpoint_metric"] == "four_metric_native_ratio_minimax"


def test_native_reference_accepts_standard_physical_evaluation_summary():
    values = {key: index + 1.0 for index, key in enumerate(ROLLOUT_METRIC_KEYS)}
    assert load_native_reference(
        {"validation": {"native_reference": {"summary": values}}}
    ) == values
