import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from valgraphnet.chp_model import CHPGNS
from valgraphnet.chp_train import (
    ROLLOUT_METRIC_KEYS,
    curriculum_horizon,
    fit_chp_normalizers,
    minimax_checkpoint_score,
    load_native_reference,
    select_rollout_start,
    stress_frame_scores,
    validate_chp_checkpoint_semantics,
)


def test_default_curriculum_matches_fixed_long_horizon_schedule():
    assert [curriculum_horizon(epoch) for epoch in range(1, 17)] == [
        1, 1, 1, 1,
        2, 2, 2,
        4, 4, 4,
        8, 8, 8,
        16, 16, 16,
    ]


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


def test_native_reference_loads_shared_evaluator_summary(tmp_path):
    values = {key: float(index + 1) for index, key in enumerate(ROLLOUT_METRIC_KEYS)}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"summary": values}), encoding="utf-8")
    assert load_native_reference(
        {"validation": {"native_reference_file": str(path)}}
    ) == values


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
    assert stress_frame_scores(case).shape == (5,)
