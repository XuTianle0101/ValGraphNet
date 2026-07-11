from pathlib import Path

import pytest
import yaml

from valgraphnet.train import _rollout_validation_due


def test_rollout_checkpoint_schedule_always_includes_final_epoch():
    cfg = {"training": {"rollout_validation_every": 3}}

    assert not _rollout_validation_due(1, 10, cfg)
    assert _rollout_validation_due(3, 10, cfg)
    assert _rollout_validation_due(10, 10, cfg)
    with pytest.raises(ValueError, match="must be positive"):
        _rollout_validation_due(
            1, 10, {"training": {"rollout_validation_every": 0}}
        )


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
    assert cfg["training"]["rollout_validation_cases"] == 20
    assert cfg["training"]["rollout_validation_steps"] == 399
