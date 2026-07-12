import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from valgraphnet.chp_model import CHPGNS, CHPState
from valgraphnet.chp_train import (
    ROLLOUT_METRIC_KEYS,
    _assert_no_gate_failure_artifact,
    _enforce_scientific_gates,
    _scientific_gate_status,
    acceleration_frame_scores,
    curriculum_horizon,
    fit_chp_normalizers,
    integration_consistent_targets,
    minimax_checkpoint_score,
    load_native_reference,
    select_rollout_start,
    stress_frame_scores,
    validate_chp_checkpoint_semantics,
)
from valgraphnet.physical_evaluation import (
    CELL_TENSOR_STRESS_SOURCE,
    NODAL_STRESS_FALLBACK_SOURCE,
)


def test_default_curriculum_matches_fixed_long_horizon_schedule():
    assert [curriculum_horizon(epoch) for epoch in range(1, 17)] == [
        1, 1, 1, 1,
        2, 2, 2,
        4, 4, 4,
        8, 8, 8,
        16, 16, 16,
    ]


def test_noise_corrected_targets_match_public_full_step_convention():
    clean_current = torch.tensor([[1.0, 0.0, 0.0]])
    noise = torch.tensor([[0.003, -0.002, 0.001]])
    input_state = CHPState(
        position=clean_current + noise,
        velocity=torch.tensor([[0.01, 0.0, 0.0]]),
    )
    exact_next = torch.tensor([[1.02, 0.0, 0.0]])
    target_velocity, target_acceleration = integration_consistent_targets(
        input_state, exact_next, 1.0
    )
    torch.testing.assert_close(input_state.position + target_velocity, exact_next)
    torch.testing.assert_close(
        input_state.velocity + target_acceleration,
        target_velocity,
    )
    torch.testing.assert_close(
        target_velocity,
        torch.tensor([[0.017, 0.002, -0.001]]),
    )


def test_noise_corrected_targets_match_two_symplectic_substeps():
    state = CHPState(
        position=torch.zeros(1, 3),
        velocity=torch.tensor([[0.1, 0.0, 0.0]]),
    )
    acceleration = torch.tensor([[2.0, -1.0, 0.5]])
    dt = 0.5
    exact_next = (
        state.position
        + dt * state.velocity
        + 0.75 * dt**2 * acceleration
    )

    target_velocity, target_acceleration = integration_consistent_targets(
        state, exact_next, dt, substeps=2
    )

    torch.testing.assert_close(target_acceleration, acceleration)
    torch.testing.assert_close(
        target_velocity, state.velocity + dt * acceleration
    )


def test_rollout_start_sampler_has_requested_mixture_and_stress_tail():
    rng = np.random.default_rng(123)
    scores = np.arange(400, dtype=np.float32)
    selected = [select_rollout_start(400, 16, scores, rng) for _ in range(20_000)]
    categories = [category for _, category in selected]
    fractions = {key: categories.count(key) / len(categories) for key in set(categories)}
    assert 0.48 < fractions["uniform"] < 0.52
    assert 0.23 < fractions["late"] < 0.27
    assert 0.23 < fractions["stress"] < 0.27
    stress_starts = [start for start, category in selected if category == "stress"]
    assert min(stress_starts) >= 345


def test_minimax_checkpoint_cannot_trade_stress_for_displacement():
    native = {key: 1.0 for key in ROLLOUT_METRIC_KEYS}
    balanced = {
        "moving_displacement_relative_rmse": 0.8,
        "final_displacement_relative_rmse": 0.8,
        "stress_relative_rmse": 0.8,
        "stress_p95_relative_rmse": 0.8,
    }
    stress_regression = {**balanced, "stress_relative_rmse": 1.1}
    assert minimax_checkpoint_score(balanced, native) == 0.8
    assert minimax_checkpoint_score(stress_regression, native) == 1.1


def test_minimax_rejects_nodal_native_denominator_for_cell_tensor_metrics():
    metrics = {
        key: 0.5 for key in ROLLOUT_METRIC_KEYS
    } | {"stress_metric_source": CELL_TENSOR_STRESS_SOURCE}
    nodal_native = {
        key: 1.0 for key in ROLLOUT_METRIC_KEYS
    } | {"stress_metric_source": NODAL_STRESS_FALLBACK_SOURCE}
    with pytest.raises(ValueError, match="different stress metrics"):
        minimax_checkpoint_score(metrics, nodal_native)
    with pytest.raises(ValueError, match="source-compatible"):
        minimax_checkpoint_score(
            metrics, {key: 1.0 for key in ROLLOUT_METRIC_KEYS}
        )
    tensor_native = {
        key: 1.0 for key in ROLLOUT_METRIC_KEYS
    } | {"stress_metric_source": CELL_TENSOR_STRESS_SOURCE}
    assert minimax_checkpoint_score(metrics, tensor_native) == 0.5
    legacy_nodal_metrics = {
        key: 0.75 for key in ROLLOUT_METRIC_KEYS
    } | {"stress_metric_source": NODAL_STRESS_FALLBACK_SOURCE}
    assert minimax_checkpoint_score(
        legacy_nodal_metrics,
        {key: 1.0 for key in ROLLOUT_METRIC_KEYS},
    ) == 0.75


def test_checkpoint_rejects_ambiguous_legacy_residual_semantics():
    legacy = {"schema_version": CHPGNS.checkpoint_schema_version}
    with pytest.raises(ValueError, match="dynamics semantics"):
        validate_chp_checkpoint_semantics(legacy)
    current = {
        "schema_version": CHPGNS.checkpoint_schema_version,
        "dynamics_schema_version": CHPGNS.dynamics_schema_version,
        "residual_parameterization": CHPGNS.residual_parameterization,
        "residual_gate": CHPGNS.residual_gate,
    }
    validate_chp_checkpoint_semantics(current)
    with pytest.raises(ValueError, match="did not pass"):
        validate_chp_checkpoint_semantics(
            current, require_scientific_gate=True
        )
    validate_chp_checkpoint_semantics(
        {**current, "scientific_gate_status": "passed"},
        require_scientific_gate=True,
    )


def test_failed_scientific_gate_blocks_resume_and_best_eligibility(tmp_path):
    stages = [{"horizon": 1, "epochs": 4}, {"horizon": 2, "epochs": 3}]
    cfg = {
        "validation": {
            "enforce_teacher_stress_gate": True,
            "teacher_stress_threshold": 0.5,
        }
    }
    assert _scientific_gate_status(3, stages, cfg) == "pending"
    assert _scientific_gate_status(4, stages, cfg) == "passed"
    assert _scientific_gate_status(
        1,
        stages,
        {
            "validation": {
                "enforce_teacher_stress_gate": False,
                "enforce_rollout_pilot_gate": False,
            }
        },
    ) == "not_required"

    with pytest.raises(RuntimeError, match="teacher-forced stress gate failed"):
        _enforce_scientific_gates(
            4,
            stages,
            {
                "teacher_stress_relative_rmse": 0.7,
                "teacher_stress_source": "nodal_scalar_vm_fallback",
            },
            {},
            cfg,
            tmp_path,
        )
    assert (tmp_path / "teacher_stress_gate_failure.json").is_file()
    with pytest.raises(RuntimeError, match="refusing training/resume"):
        _assert_no_gate_failure_artifact(tmp_path, context="training/resume")


def test_native_reference_loads_shared_evaluator_summary(tmp_path):
    values = {key: float(index + 1) for index, key in enumerate(ROLLOUT_METRIC_KEYS)}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"summary": values}), encoding="utf-8")
    assert load_native_reference(
        {"validation": {"native_reference_file": str(path)}}
    ) == values

    tensor_payload = {
        "summary": {
            **values,
            "stress_metric_source": CELL_TENSOR_STRESS_SOURCE,
        }
    }
    path.write_text(json.dumps(tensor_payload), encoding="utf-8")
    assert load_native_reference(
        {"validation": {"native_reference_file": str(path)}}
    ) == {
        **values,
        "stress_metric_source": CELL_TENSOR_STRESS_SOURCE,
    }


def test_absolute_validation_mode_rejects_native_reference_leakage():
    assert load_native_reference(
        {"validation": {"checkpoint_reference_mode": "absolute_validation"}}
    ) is None
    with pytest.raises(ValueError, match="forbids native references"):
        load_native_reference(
            {
                "validation": {
                    "checkpoint_reference_mode": "absolute_validation",
                    "native_reference": {"moving_relative_rmse": 1.0},
                }
            }
        )


def test_normalizers_are_finite_invertible_and_do_not_clip_stress():
    displacement = np.zeros((5, 4, 3), dtype=np.float32)
    for frame in range(5):
        displacement[frame, :, 0] = frame * 0.01
    velocity = np.zeros_like(displacement)
    velocity[..., 0] = 0.01
    stress = np.arange(20, dtype=np.float32).reshape(5, 4, 1) * 1.0e5
    case = SimpleNamespace(
        num_steps=5,
        num_nodes=4,
        stress_dim=1,
        displacement=displacement,
        velocity=velocity,
        stress=stress,
    )
    normalizers = fit_chp_normalizers(
        [case], max_cases=1, frames_per_case=4, nodes_per_frame=4
    )
    values = torch.tensor([[0.0], [1.9e6], [-2.5e7]])
    recovered = normalizers.stress.inverse(normalizers.stress.transform(values))
    torch.testing.assert_close(recovered, values, rtol=1.0e-5, atol=1.0e-3)
    assert torch.isfinite(normalizers.displacement_scale).all()
    assert torch.isfinite(normalizers.velocity_scale).all()
    assert torch.isfinite(normalizers.acceleration_scale).all()
    torch.testing.assert_close(
        normalizers.acceleration_scale,
        torch.tensor(1.0e-8),
    )
    assert stress_frame_scores(case).shape == (5,)


def test_acceleration_normalizer_uses_free_nodes_and_nonunit_dt():
    velocity = np.zeros((3, 3, 3), dtype=np.float32)
    velocity[1, 0] = np.asarray([2.0, 4.0, 6.0])
    velocity[2, 0] = np.asarray([14.0, 20.0, 26.0])
    velocity[:, 1:] = 100.0
    case = SimpleNamespace(
        num_steps=3,
        num_nodes=3,
        stress_dim=1,
        times=np.asarray([0.0, 2.0, 6.0], dtype=np.float32),
        displacement=np.zeros_like(velocity),
        velocity=velocity,
        stress=np.ones((3, 3, 1), dtype=np.float32),
        fixed_mask=np.asarray([False, True, False]),
        prescribed_mask=np.asarray([False, False, True]),
    )
    normalizers = fit_chp_normalizers(
        [case], max_cases=1, frames_per_case=2, nodes_per_frame=3
    )
    torch.testing.assert_close(
        normalizers.acceleration_scale,
        torch.sqrt(torch.tensor(64.0 / 6.0)),
    )
    np.testing.assert_allclose(
        acceleration_frame_scores(case),
        np.asarray(
            [np.sqrt(14.0 / 3.0), np.sqrt(50.0 / 3.0), 0.0],
            dtype=np.float32,
        ),
        rtol=1.0e-6,
    )

    state = normalizers.state_dict()
    restored = type(normalizers).from_state_dict(state)
    torch.testing.assert_close(
        restored.acceleration_scale, normalizers.acceleration_scale
    )
    state.pop("acceleration_scale")
    legacy = type(normalizers).from_state_dict(state)
    torch.testing.assert_close(legacy.acceleration_scale, legacy.velocity_scale)
