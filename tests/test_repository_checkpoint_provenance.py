from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from valgraphnet.checkpoint_provenance import (
    ARTIFACT_TYPE,
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_FAMILY,
    atomic_torch_save,
    build_repo_data_contract,
    checkpoint_metadata,
    validate_repo_checkpoint,
)
from valgraphnet.legacy_rollout_export import (
    _evaluate_exported_predictions,
    _select_export_case_ids,
    export_legacy_rollouts,
)
from valgraphnet.train import save_checkpoint


def _write_case(root: Path, case_id: str, frames: int) -> None:
    case = root / case_id
    case.mkdir(parents=True)
    nodes = 3
    (case / "metadata.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "schema_version": 2,
                "source": "unit-test",
            }
        ),
        encoding="utf-8",
    )
    np.save(case / "nodes.npy", np.zeros((nodes, 3), dtype=np.float32))
    np.save(case / "times.npy", np.arange(frames, dtype=np.float32))
    np.save(case / "U.npy", np.zeros((frames, nodes, 3), dtype=np.float32))
    np.save(case / "S.npy", np.zeros((frames, nodes, 1), dtype=np.float32))


def _strict_fixture(tmp_path: Path, frames: int = 4) -> tuple[dict, Path]:
    root = tmp_path / "data"
    _write_case(root, "train_0", frames)
    _write_case(root, "val_0", frames)
    # The held-out id deliberately has no directory.  Contract construction
    # must never open test arrays.
    split_file = root / "splits.json"
    split_file.write_text(
        json.dumps(
            {"train": ["train_0"], "val": ["val_0"], "test": ["test_0"]}
        ),
        encoding="utf-8",
    )
    cfg = {
        "seed": 42,
        "provenance": {
            "checkpoint_policy": "strict_v2",
            "expected_frames": frames,
            "expected_split_counts": {"train": 1, "val": 1, "test": 1},
            "expected_case_schema_version": 2,
        },
        "data": {
            "root": str(root),
            "split_file": str(split_file),
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
        },
        "model": {"type": "hybrid"},
        "training": {"resume_from": "auto"},
        "evaluation": {"case_selection": "even"},
        "loss": {"stress": 0.5},
    }
    return cfg, root


def _checkpoint(cfg: dict, contract: dict, *, role: str = "best") -> dict:
    model = {"layer.weight": torch.zeros((2, 3), dtype=torch.float32)}
    return {
        **checkpoint_metadata(cfg, model, contract, 10, artifact_role=role),
        "cfg": deepcopy(cfg),
        "model": model,
        "output_dim": 10,
        "normalizers": None,
    }


def test_strict_v2_checkpoint_round_trip_and_test_isolation(tmp_path):
    cfg, _ = _strict_fixture(tmp_path)
    contract = build_repo_data_contract(cfg)
    checkpoint = _checkpoint(cfg, contract)

    assert contract["test_content_accessed"] is False
    assert contract["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert all(
        len(array["sha256"]) == 64
        for role in ("train", "val")
        for case in contract["content_records"][role]
        for array in case["arrays"]
    )
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["artifact_type"] == ARTIFACT_TYPE
    assert checkpoint["model_family"] == MODEL_FAMILY
    validate_repo_checkpoint(
        checkpoint, cfg, contract, purpose="resume", source="fixture.pt"
    )
    validate_repo_checkpoint(
        checkpoint, cfg, contract, purpose="export", source="fixture.pt"
    )
    with pytest.raises(ValueError, match="require a data contract"):
        checkpoint_metadata(cfg, checkpoint["model"], None, 10, artifact_role="best")


def test_strict_v2_rejects_legacy_and_tampered_config_but_legacy_policy_accepts(
    tmp_path,
):
    cfg, _ = _strict_fixture(tmp_path)
    contract = build_repo_data_contract(cfg)
    with pytest.raises(ValueError, match="schema"):
        validate_repo_checkpoint(
            {"model": {}}, cfg, contract, purpose="resume"
        )

    checkpoint = _checkpoint(cfg, contract)
    changed = deepcopy(cfg)
    changed["loss"]["stress"] = 0.25
    with pytest.raises(ValueError, match="config does not match"):
        validate_repo_checkpoint(
            checkpoint, changed, contract, purpose="resume"
        )
    checkpoint["cfg"]["loss"]["stress"] = 9.0
    with pytest.raises(ValueError, match="embedded config fingerprint"):
        validate_repo_checkpoint(
            checkpoint, cfg, contract, purpose="resume"
        )

    # Historical 200-frame configs opt out by omission and keep their original
    # schema-less checkpoint compatibility.
    validate_repo_checkpoint(
        {"model": {}}, {"training": {}}, None, purpose="resume"
    )


def test_data_contract_changes_when_frame_count_changes(tmp_path):
    cfg, root = _strict_fixture(tmp_path, frames=4)
    contract_4 = build_repo_data_contract(cfg)
    for case_id in ("train_0", "val_0"):
        case = root / case_id
        np.save(case / "times.npy", np.arange(2, dtype=np.float32))
        np.save(case / "U.npy", np.zeros((2, 3, 3), dtype=np.float32))
        np.save(case / "S.npy", np.zeros((2, 3, 1), dtype=np.float32))
    cfg["provenance"].pop("expected_frames")
    contract_2 = build_repo_data_contract(cfg)

    assert contract_4["fingerprint_sha256"] != contract_2["fingerprint_sha256"]
    with pytest.raises(ValueError, match="data fingerprint"):
        validate_repo_checkpoint(
            _checkpoint(cfg, contract_4),
            cfg,
            contract_2,
            purpose="resume",
        )


def test_data_contract_detects_same_shape_content_tampering(tmp_path):
    cfg, root = _strict_fixture(tmp_path, frames=4)
    original = build_repo_data_contract(cfg)
    path = root / "train_0" / "U.npy"
    changed = np.load(path, allow_pickle=False).copy()
    changed[0, 0, 0] = 123.0
    np.save(path, changed)

    tampered = build_repo_data_contract(cfg)

    assert original["fingerprint_sha256"] != tampered["fingerprint_sha256"]
    with pytest.raises(ValueError, match="data fingerprint"):
        validate_repo_checkpoint(
            _checkpoint(cfg, original),
            cfg,
            tampered,
            purpose="resume",
        )


def test_resume_path_is_excluded_but_other_training_config_is_bound(tmp_path):
    cfg, _ = _strict_fixture(tmp_path)
    contract = build_repo_data_contract(cfg)
    checkpoint = _checkpoint(cfg, contract)
    relocated = deepcopy(cfg)
    relocated["training"]["resume_from"] = str(tmp_path / "copied.pt")
    validate_repo_checkpoint(
        checkpoint, relocated, contract, purpose="resume"
    )
    relocated["training"]["epochs"] = 99
    with pytest.raises(ValueError, match="config does not match"):
        validate_repo_checkpoint(
            checkpoint, relocated, contract, purpose="resume"
        )


def test_formal_export_requires_best_role(tmp_path):
    cfg, _ = _strict_fixture(tmp_path)
    contract = build_repo_data_contract(cfg)
    with pytest.raises(ValueError, match="best checkpoint"):
        validate_repo_checkpoint(
            _checkpoint(cfg, contract, role="latest"),
            cfg,
            contract,
            purpose="export",
        )


def test_atomic_save_replaces_checkpoint(tmp_path):
    path = tmp_path / "latest.pt"
    atomic_torch_save({"epoch": 1}, path)
    atomic_torch_save({"epoch": 2}, path)
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 2
    assert not list(tmp_path.glob(".latest.pt.*.tmp"))


def test_training_save_checkpoint_emits_strict_v2_best_artifact(tmp_path):
    cfg, _ = _strict_fixture(tmp_path)
    contract = build_repo_data_contract(cfg)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "best.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        cfg,
        None,
        10,
        3,
        0.5,
        contract,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["artifact_role"] == "best"
    validate_repo_checkpoint(
        checkpoint, cfg, contract, purpose="resume", source=path
    )


def test_export_even_selection_and_evaluator_protocol_are_identical(
    tmp_path, monkeypatch
):
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps({"val": [f"v{i}" for i in range(5)]}), encoding="utf-8"
    )
    assert _select_export_case_ids(split_file, "val", 3, "even") == [
        "v0",
        "v2",
        "v4",
    ]

    observed = {}

    def fake_evaluator(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"summary": {}}

    monkeypatch.setattr(
        "valgraphnet.legacy_rollout_export.evaluate_prediction_directory",
        fake_evaluator,
    )
    _evaluate_exported_predictions(
        tmp_path,
        split_file,
        "val",
        tmp_path / "predictions",
        output_path=tmp_path / "metrics.json",
        max_cases=3,
        case_selection="even",
    )
    assert observed["kwargs"]["max_cases"] == 3
    assert observed["kwargs"]["case_selection"] == "even"


def test_strict_export_reports_provenance_and_split_errors_before_cuda(tmp_path):
    cfg, _ = _strict_fixture(tmp_path)
    legacy_path = tmp_path / "legacy.pt"
    torch.save({"model": {}}, legacy_path)
    with pytest.raises(ValueError, match="schema"):
        export_legacy_rollouts(
            cfg,
            legacy_path,
            tmp_path / "legacy-out",
            split="val",
            max_cases=1,
            case_selection="even",
        )

    contract = build_repo_data_contract(cfg)
    valid_path = tmp_path / "best.pt"
    torch.save(_checkpoint(cfg, contract), valid_path)
    with pytest.raises(ValueError, match="explicit --split"):
        export_legacy_rollouts(
            cfg,
            valid_path,
            tmp_path / "strict-out",
            max_cases=1,
            case_selection="even",
        )
    with pytest.raises(ValueError, match="differs from evaluation"):
        export_legacy_rollouts(
            cfg,
            valid_path,
            tmp_path / "strict-out",
            split="val",
            max_cases=1,
            case_selection="head",
        )
