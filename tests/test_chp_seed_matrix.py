from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.run_chp_seed_matrix import (
    SEEDS,
    SeedMatrixError,
    checkpoint_structure_hash,
    derive_seed_config,
    experiment_config_hash,
    run_seed_matrix,
    validate_base_protocol,
)
from valgraphnet.chp_model import CHPGNS
from valgraphnet.physical_evaluation import PRIMARY_METRICS


def _base_config(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "data": {"val_split": "val", "test_split": "test"},
        "model": {"scalar_dim": 8, "vector_dim": 2, "cell_dim": 4},
        "contact": {"enabled": True},
        "training": {
            "device": "cuda",
            "amp": True,
            "amp_dtype": "bfloat16",
            "output_dir": str(tmp_path / "chp_seed42"),
            "resume_from": "auto",
        },
        "validation": {
            "cases": 20,
            "steps": 399,
            "native_reference_case_selection": "even",
            "enforce_teacher_stress_gate": True,
            "enforce_rollout_pilot_gate": True,
        },
    }


def _write_config(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _checkpoint(cfg: dict, *, shape: tuple[int, ...] = (2, 3)) -> dict:
    metrics = {key: 0.1 * (index + 1) for index, key in enumerate(PRIMARY_METRICS)}
    metrics["diverged_cases"] = 0.0
    metrics["teacher_stress_relative_rmse"] = 0.25
    return {
        "schema_version": CHPGNS.checkpoint_schema_version,
        "dynamics_schema_version": CHPGNS.dynamics_schema_version,
        "residual_parameterization": CHPGNS.residual_parameterization,
        "residual_gate": CHPGNS.residual_gate,
        "architecture": "CHP-GNS",
        "problem_type": "dynamic",
        "time_semantics": "dynamic",
        "scientific_gate_status": "passed",
        "epoch": 16,
        "rollout_metrics": metrics,
        "model": {"synthetic.weight": torch.zeros(shape)},
        "config": cfg,
    }


def test_seed_configs_change_only_seed_and_independent_output(tmp_path):
    base = _base_config(tmp_path)
    hashes = []
    outputs = []
    for seed in SEEDS:
        cfg = derive_seed_config(base, seed)
        hashes.append(experiment_config_hash(cfg))
        outputs.append(cfg["training"]["output_dir"])
        assert cfg["seed"] == seed
    assert len(set(hashes)) == 1
    assert len(set(outputs)) == 3
    assert base["seed"] == 42
    assert base["training"]["output_dir"].endswith("seed42")


def test_protocol_requires_cuda_bf16_val20_and_never_test(tmp_path):
    base = _base_config(tmp_path)
    validate_base_protocol(base)
    invalid = deepcopy(base)
    invalid["data"]["val_split"] = "test"
    with pytest.raises(SeedMatrixError, match="never test"):
        validate_base_protocol(invalid)
    invalid = deepcopy(base)
    invalid["training"]["device"] = "cpu"
    with pytest.raises(SeedMatrixError, match="CUDA"):
        validate_base_protocol(invalid)


def test_dry_run_does_not_train_and_keeps_later_seeds_conditional(tmp_path):
    source = tmp_path / "base.yaml"
    _write_config(source, _base_config(tmp_path))
    called = []

    result = run_seed_matrix(
        source,
        state_dir=tmp_path / "matrix",
        dry_run=True,
        trainer=lambda *_: called.append(True),
    )

    assert called == []
    assert result["validation"]["test_split_accessed"] is False
    assert result["seeds"][0]["action"] == "train"
    assert result["seeds"][1]["action"] == "conditional_on_seed42_gate"
    assert not (tmp_path / "matrix").exists()


def test_matrix_trains_strictly_in_seed_order_and_aggregates_val20(tmp_path):
    source = tmp_path / "base.yaml"
    base = _base_config(tmp_path)
    _write_config(source, base)
    calls = []

    def trainer(config_path: Path, seed: int, output_dir: Path) -> None:
        calls.append(seed)
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg["seed"] == seed
        assert Path(cfg["training"]["output_dir"]).resolve() == output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = _checkpoint(cfg)
        for index, name in enumerate(PRIMARY_METRICS):
            checkpoint["rollout_metrics"][name] += (seed - 42) * 0.01
        torch.save(checkpoint, output_dir / "best.pt")

    result = run_seed_matrix(
        source,
        state_dir=tmp_path / "matrix",
        trainer=trainer,
    )

    assert calls == [42, 43, 44]
    assert result["aggregate"]["all_scientific_gates_passed"]
    assert result["aggregate"]["gate_status"] == {
        "42": "passed",
        "43": "passed",
        "44": "passed",
    }
    moving = result["aggregate"]["metrics"][PRIMARY_METRICS[0]]
    assert moving["mean"] == pytest.approx(0.11)
    assert moving["std"] == pytest.approx(0.01)
    saved = json.loads(
        (tmp_path / "matrix" / "val20_seed_aggregate.json").read_text()
    )
    assert saved["validation"]["test_split_accessed"] is False


def test_failed_seed42_checkpoint_blocks_seed43_and_seed44(tmp_path):
    source = tmp_path / "base.yaml"
    base = _base_config(tmp_path)
    _write_config(source, base)
    seed42 = Path(base["training"]["output_dir"])
    seed42.mkdir(parents=True)
    failed = _checkpoint(base)
    failed["scientific_gate_status"] = "pending"
    torch.save(failed, seed42 / "best.pt")
    calls = []

    with pytest.raises(SeedMatrixError, match="did not pass"):
        run_seed_matrix(
            source,
            state_dir=tmp_path / "matrix",
            trainer=lambda *_: calls.append(True),
        )
    assert calls == []
    assert not Path(derive_seed_config(base, 43)["training"]["output_dir"]).exists()


@pytest.mark.parametrize("drift", ["config", "structure"])
def test_existing_seed_drift_fails_closed(tmp_path, drift):
    source = tmp_path / "base.yaml"
    base = _base_config(tmp_path)
    _write_config(source, base)
    seed42_cfg = derive_seed_config(base, 42)
    seed43_cfg = derive_seed_config(base, 43)
    for cfg in (seed42_cfg, seed43_cfg):
        Path(cfg["training"]["output_dir"]).mkdir(parents=True)
    torch.save(
        _checkpoint(seed42_cfg),
        Path(seed42_cfg["training"]["output_dir"]) / "best.pt",
    )
    shape = (2, 3)
    if drift == "config":
        seed43_cfg["model"]["scalar_dim"] = 99
    else:
        shape = (3, 3)
    torch.save(
        _checkpoint(seed43_cfg, shape=shape),
        Path(seed43_cfg["training"]["output_dir"]) / "best.pt",
    )

    with pytest.raises(SeedMatrixError, match=f"{drift}.*hash"):
        run_seed_matrix(source, state_dir=tmp_path / "matrix", dry_run=True)


def test_structure_hash_uses_parameter_shapes(tmp_path):
    cfg = _base_config(tmp_path)
    first = checkpoint_structure_hash(_checkpoint(cfg, shape=(2, 3)))
    second = checkpoint_structure_hash(_checkpoint(cfg, shape=(3, 2)))
    assert first != second
