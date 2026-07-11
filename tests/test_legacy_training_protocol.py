import json
from pathlib import Path

import pytest
import yaml

from valgraphnet.train import _load_rollout_native_reference, _rollout_validation_due


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
