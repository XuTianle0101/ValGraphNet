import json
from pathlib import Path

import pytest
import torch
import yaml

from valgraphnet.train import (
    _load_rollout_native_reference,
    _resolve_resume_path,
    _rollout_metric_masks,
    _rollout_validation_due,
    _trajectory_stress_p95_sums,
    _validate_output_directory_for_resume,
)


def test_rollout_checkpoint_schedule_always_includes_final_epoch():
    cfg = {"training": {"rollout_validation_every": 3}}

    assert not _rollout_validation_due(1, 10, cfg)
    assert _rollout_validation_due(3, 10, cfg)
    assert _rollout_validation_due(10, 10, cfg)
    with pytest.raises(ValueError, match="must be positive"):
        _rollout_validation_due(
            1, 10, {"training": {"rollout_validation_every": 0}}
        )


def test_repository_checkpoint_reference_is_validation_only_and_four_metric(tmp_path):
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps({"val": ["a", "b", "c"], "test": ["held_out"]}),
        encoding="utf-8",
    )
    reference_file = tmp_path / "native.json"
    values = {
        "moving_displacement_relative_rmse": 1.0,
        "final_displacement_relative_rmse": 2.0,
        "stress_relative_rmse": 3.0,
        "stress_p95_relative_rmse": 4.0,
    }
    reference_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "evaluation": {"split": "val"},
                "per_case": [
                    {"case_id": "a", "evaluated_frames": 2},
                    {"case_id": "c", "evaluated_frames": 2},
                ],
                "rollout": values,
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "data": {"split_file": str(split_file), "val_split": "val"},
        "training": {"rollout_validation_cases": 2, "rollout_validation_steps": 1},
        "validation": {"native_reference_file": str(reference_file)},
    }

    assert _load_rollout_native_reference(cfg) == values


def test_full400_repository_baseline_is_cuda_rollout_selected():
    cfg = yaml.safe_load(
        Path("configs/deforming_plate_case.full400_repo.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert cfg["data"]["root"] == "data/deforming_plate_cases_400"
    assert cfg["data"]["test_split"] == "test"
    assert cfg["model"]["independent_stress_decoder"] is True
    assert cfg["training"]["device"] == "cuda"
    assert cfg["training"]["amp_dtype"] == "bfloat16"
    assert cfg["training"]["checkpoint_metric"] == "rollout"
    assert (
        cfg["training"]["rollout_checkpoint_score_mode"]
        == "four_metric_native_ratio_minimax"
    )
    assert cfg["training"]["rollout_validation_cases"] == 20
    assert cfg["training"]["rollout_validation_steps"] == 399
    assert cfg["validation"]["native_reference_file"].endswith(
        "native_val20/metrics.json"
    )
    assert cfg["provenance"]["checkpoint_policy"] == "strict_v2"
    assert cfg["provenance"]["expected_frames"] == 400
    assert cfg["evaluation"]["case_selection"] == "even"


def test_rollout_stress_mask_matches_shared_physical_evaluator():
    tensors = {
        "fixed": torch.tensor([False, True, False, True]),
        "prescribed": torch.tensor([False, False, True, True]),
    }

    moving, stress = _rollout_metric_masks(tensors)

    torch.testing.assert_close(
        moving, torch.tensor([True, False, False, False])
    )
    # Fixed/clamped nodes remain in stress evaluation; only prescribed nodes
    # are excluded, exactly as in physical_evaluation.py.
    torch.testing.assert_close(
        stress, torch.tensor([True, True, False, False])
    )


def test_rollout_p95_uses_one_threshold_for_the_complete_trajectory():
    truth = [torch.tensor([1.0, 2.0]), torch.tensor([100.0, 200.0])]
    residual = [torch.tensor([7.0, 8.0]), torch.tensor([9.0, 10.0])]

    error, reference = _trajectory_stress_p95_sums(truth, residual)

    # The trajectory-level 95th percentile is 185, so only truth=200 is in
    # the peak region.  Per-frame selection would incorrectly include 2.
    torch.testing.assert_close(error, torch.tensor(100.0))
    torch.testing.assert_close(reference, torch.tensor(40_000.0))


def test_strict_resume_rejects_missing_explicit_path_and_populated_fresh_output(
    tmp_path,
):
    output = tmp_path / "run"
    output.mkdir()
    explicit = tmp_path / "missing.pt"
    cfg = {
        "provenance": {"checkpoint_policy": "strict_v2"},
        "training": {"resume_from": str(explicit)},
    }
    with pytest.raises(FileNotFoundError, match="explicit resume"):
        _resolve_resume_path(cfg, output)

    (output / "history.json").write_text("[]", encoding="utf-8")
    cfg["training"]["resume_from"] = "auto"
    with pytest.raises(RuntimeError, match="populated output directory"):
        _validate_output_directory_for_resume(cfg, output, None)
