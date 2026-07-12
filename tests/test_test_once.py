from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from valgraphnet.test_once import (
    PRIMARY_METRICS,
    TestOnceError,
    claim_test_run,
    freeze_experiment,
    protocol_paths,
    sha256_file,
    verify_frozen_experiment,
)


def _metrics(value: float = 0.5) -> dict[str, float]:
    return {key: value for key in PRIMARY_METRICS}


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _config(tmp_path: Path, family: str) -> dict:
    cfg = {
        "seed": 42,
        "data": {
            "root": "data/unused",
            "split_file": "data/unused/splits.json",
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
        },
        "training": {"device": "cuda"},
        "validation": {},
    }
    if family == "repo":
        cfg["training"].update(
            {
                "checkpoint_metric": "rollout",
                "rollout_checkpoint_score_mode": "four_metric_native_ratio_minimax",
            }
        )
    if family in {"fair", "multiscale", "repo"}:
        reference = tmp_path / f"{family}_native_val.json"
        _write_json(
            reference,
            {
                "schema_version": 2,
                "evaluation": {"split": "val"},
                "summary": _metrics(1.0),
            },
        )
        cfg["validation"]["native_reference_file"] = reference.name
    if family == "chp":
        cfg["validation"]["checkpoint_reference_mode"] = "absolute_validation"
    return cfg


def _checkpoint(family: str, cfg: dict, epoch: int = 3) -> dict:
    common = {"model": {}, "epoch": epoch, "score": 0.5}
    if family == "native":
        return {**common, "cfg": cfg, "scheduler": {}}
    if family == "repo":
        return {**common, "cfg": cfg, "normalizers": {}, "output_dim": 10}
    if family == "fair":
        return {
            **common,
            "cfg": cfg,
            "schema_version": 2,
            "model_family": "fair_deforming_plate_mgn",
            "checkpoint_metric": "four_metric_native_ratio_minimax",
            "rollout_metrics": _metrics(),
            "native_reference": _metrics(1.0),
            "best_score": 0.5,
        }
    if family == "multiscale":
        return {
            **common,
            "cfg": cfg,
            "schema_version": 2,
            "model_family": "two_level_bistride_deforming_plate_mgn",
            "checkpoint_metric": "four_metric_native_ratio_minimax",
            "checkpoint_split": "val",
            "rollout_metrics": _metrics(),
            "native_reference": _metrics(1.0),
            "best_score": 0.5,
        }
    return {
        **common,
        "config": cfg,
        "schema_version": 2,
        "architecture": "CHP-GNS",
        "scientific_gate_status": "passed",
        "rollout_metrics": _metrics(),
        "best_score": 0.5,
    }


def _model_files(tmp_path: Path, family: str) -> dict:
    cfg = _config(tmp_path, family)
    config_path = tmp_path / f"{family}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    checkpoint_path = tmp_path / f"{family}_best.pt"
    torch.save(_checkpoint(family, cfg), checkpoint_path)
    model = {
        "name": family,
        "family": family,
        "config": config_path.name,
        "checkpoint": checkpoint_path.name,
    }
    if family == "native":
        evidence = tmp_path / "native_val_metrics.json"
        _write_json(
            evidence,
            {
                "schema_version": 2,
                "evaluation": {"split": "val"},
                "summary": {"loss": 0.5},
            },
        )
        model["selection_evidence"] = evidence.name
    if family == "repo":
        evidence = tmp_path / "repo_history.json"
        _write_json(
            evidence,
            [
                {
                    "epoch": 3,
                    "rollout_val": {**_metrics(), "score": 0.5},
                    "score": 0.5,
                }
            ],
        )
        model["selection_evidence"] = evidence.name
    return model


def _write_spec(tmp_path: Path, models: list[dict], experiment_id: str = "paper-seed42") -> Path:
    entrypoint = tmp_path / "frozen_eval.py"
    entrypoint.write_text("raise RuntimeError('unit tests must not run test evaluation')\n")
    spec = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "models": models,
        "evaluation_entrypoints": [entrypoint.name],
        "test_commands": [
            {
                "name": "frozen_matrix",
                "models": [model["name"] for model in models],
                "argv": ["python", entrypoint.name],
            }
        ],
    }
    path = tmp_path / "test_once_spec.json"
    _write_json(path, spec)
    return path


@pytest.mark.parametrize("family", ["native", "fair", "multiscale", "repo", "chp"])
def test_freeze_supports_every_checkpoint_family_without_test_access(tmp_path, family):
    model = _model_files(tmp_path, family)
    spec = _write_spec(tmp_path, [model], experiment_id=f"matrix-{family}")
    registry = tmp_path / "registry"

    manifest_path = freeze_experiment(spec, registry, workspace_root=tmp_path)
    manifest = verify_frozen_experiment(registry, f"matrix-{family}")

    assert manifest_path.exists()
    assert manifest["test_data_accessed"] is False
    assert manifest["models"][0]["family"] == family
    assert len(manifest["models"][0]["config"]["sha256"]) == 64
    assert len(manifest["models"][0]["checkpoint"]["sha256"]) == 64
    assert manifest["models"][0]["selection"]["split"] == "val"


def test_claim_is_atomic_and_same_experiment_cannot_run_twice(tmp_path):
    spec = _write_spec(tmp_path, [_model_files(tmp_path, "chp")])
    registry = tmp_path / "registry"
    freeze_experiment(spec, registry, workspace_root=tmp_path)

    claim = claim_test_run(registry, "paper-seed42")

    assert claim.exists()
    assert json.loads(claim.read_text(encoding="utf-8"))["retry_allowed"] is False
    with pytest.raises(TestOnceError, match="already claimed"):
        claim_test_run(registry, "paper-seed42")


def test_drift_blocks_claim_without_consuming_attempt(tmp_path):
    model = _model_files(tmp_path, "native")
    spec = _write_spec(tmp_path, [model])
    registry = tmp_path / "registry"
    freeze_experiment(spec, registry, workspace_root=tmp_path)
    config_path = tmp_path / model["config"]
    config_path.write_text(config_path.read_text() + "training:\n  device: cpu\n")

    with pytest.raises(TestOnceError, match="drift detected"):
        claim_test_run(registry, "paper-seed42")

    _, claim_path, _ = protocol_paths(registry, "paper-seed42")
    assert not claim_path.exists()


def test_validation_and_test_split_must_be_distinct(tmp_path):
    model = _model_files(tmp_path, "chp")
    config_path = tmp_path / model["config"]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["data"]["val_split"] = "test"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    torch.save(_checkpoint("chp", cfg), tmp_path / model["checkpoint"])
    spec = _write_spec(tmp_path, [model])

    with pytest.raises(TestOnceError, match="present and distinct"):
        freeze_experiment(spec, tmp_path / "registry", workspace_root=tmp_path)


def test_failed_chp_scientific_gate_cannot_be_frozen(tmp_path):
    model = _model_files(tmp_path, "chp")
    checkpoint_path = tmp_path / model["checkpoint"]
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["scientific_gate_status"] = "failed"
    torch.save(checkpoint, checkpoint_path)
    spec = _write_spec(tmp_path, [model])

    with pytest.raises(TestOnceError, match="gates have not passed"):
        freeze_experiment(spec, tmp_path / "registry", workspace_root=tmp_path)


def test_manifest_binds_raw_config_checkpoint_and_spec_sha256(tmp_path):
    model = _model_files(tmp_path, "fair")
    spec = _write_spec(tmp_path, [model])
    registry = tmp_path / "registry"
    path = freeze_experiment(spec, registry, workspace_root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["specification"]["sha256"] == sha256_file(spec)
    assert payload["models"][0]["config"]["sha256"] == sha256_file(
        tmp_path / model["config"]
    )
    assert payload["models"][0]["checkpoint"]["sha256"] == sha256_file(
        tmp_path / model["checkpoint"]
    )
