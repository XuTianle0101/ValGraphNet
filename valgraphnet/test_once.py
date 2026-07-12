"""Fail-closed test-once locking for frozen experiment matrices.

The protocol deliberately does not inspect a test split or any trajectory data.
It binds development-time evidence (configuration, validation-selected checkpoint,
and optional validation artifacts) to a unique experiment id.  A guarded test
plan can then be claimed exactly once after every bound file is re-verified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from valgraphnet.checkpoint_provenance import (
    ARTIFACT_TYPE as REPOSITORY_ARTIFACT_TYPE,
    CHECKPOINT_SCHEMA_VERSION as REPOSITORY_CHECKPOINT_SCHEMA_VERSION,
    MODEL_FAMILY as REPOSITORY_MODEL_FAMILY,
    config_sha256,
    strict_checkpoint_provenance,
)


LOCK_SCHEMA_VERSION = 1
CLAIM_SCHEMA_VERSION = 1
PROTOCOL_NAME = "valgraphnet-test-once"
SUPPORTED_FAMILIES = frozenset({"native", "fair", "multiscale", "repo", "chp"})
PRIMARY_METRICS = (
    "moving_displacement_relative_rmse",
    "final_displacement_relative_rmse",
    "stress_relative_rmse",
    "stress_p95_relative_rmse",
)


class TestOnceError(RuntimeError):
    """Raised when a frozen test protocol cannot be trusted."""

    __test__ = False


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_experiment(
    spec_path: str | Path,
    registry_dir: str | Path = "outputs/test_once_registry",
    *,
    workspace_root: str | Path = ".",
) -> Path:
    """Validate and atomically freeze one complete test matrix.

    Paths in the JSON/YAML specification are relative to ``workspace_root``.
    The function never opens the configured split file or case directory.
    """

    workspace = Path(workspace_root).resolve()
    spec_file = Path(spec_path).resolve()
    spec = _read_mapping(spec_file, "test-once specification")
    if int(spec.get("schema_version", 0)) != LOCK_SCHEMA_VERSION:
        raise TestOnceError(
            f"specification schema_version must be {LOCK_SCHEMA_VERSION}"
        )
    experiment_id = _validate_experiment_id(spec.get("experiment_id"))
    registry = Path(registry_dir).resolve()
    manifest_path, claim_path, _ = protocol_paths(registry, experiment_id)
    if claim_path.exists():
        raise TestOnceError(
            f"experiment {experiment_id!r} has already consumed its test run"
        )
    if manifest_path.exists():
        raise TestOnceError(
            f"experiment {experiment_id!r} is already frozen; use a new id for "
            "a new protocol and never overwrite the existing lock"
        )

    raw_models = spec.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise TestOnceError("specification models must be a non-empty list")
    names: set[str] = set()
    models: list[dict[str, Any]] = []
    for raw_model in raw_models:
        if not isinstance(raw_model, Mapping):
            raise TestOnceError("every model specification must be an object")
        name = str(raw_model.get("name", "")).strip()
        if not name or name in names:
            raise TestOnceError("model names must be non-empty and unique")
        names.add(name)
        models.append(_audit_model(dict(raw_model), workspace))

    test_commands = _validate_test_commands(spec.get("test_commands"), names)
    entrypoints = _fingerprint_paths(
        spec.get("evaluation_entrypoints", []), workspace, "evaluation entrypoint"
    )
    frozen_inputs = _fingerprint_paths(
        spec.get("frozen_inputs", []), workspace, "frozen input"
    )
    manifest: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "experiment_id": experiment_id,
        "created_at_utc": _utc_now(),
        "workspace_root": str(workspace),
        "specification": _fingerprint(spec_file),
        "models": models,
        "test_commands": test_commands,
        "evaluation_entrypoints": entrypoints,
        "frozen_inputs": frozen_inputs,
        "test_data_accessed": False,
        "state": "frozen_not_claimed",
    }
    manifest["frozen_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at_utc", "frozen_payload_sha256"}
        }
    )
    _atomic_create_json(manifest_path, manifest)
    return manifest_path


def verify_frozen_experiment(
    registry_dir: str | Path,
    experiment_id: str,
) -> dict[str, Any]:
    """Re-audit every frozen artifact without claiming or reading test data."""

    experiment_id = _validate_experiment_id(experiment_id)
    manifest_path, _, _ = protocol_paths(registry_dir, experiment_id)
    manifest = _read_mapping(manifest_path, "test-once lock manifest")
    _validate_manifest_identity(manifest, experiment_id)
    workspace = Path(str(manifest.get("workspace_root", ""))).resolve()
    if not workspace.is_dir():
        raise TestOnceError(f"frozen workspace no longer exists: {workspace}")

    _verify_fingerprint(manifest.get("specification"), "specification")
    for collection, label in (
        (manifest.get("evaluation_entrypoints", []), "evaluation entrypoint"),
        (manifest.get("frozen_inputs", []), "frozen input"),
    ):
        if not isinstance(collection, list):
            raise TestOnceError(f"manifest {label} collection is invalid")
        for record in collection:
            _verify_fingerprint(record, label)

    frozen_models = manifest.get("models")
    if not isinstance(frozen_models, list) or not frozen_models:
        raise TestOnceError("lock manifest has no frozen models")
    refreshed: list[dict[str, Any]] = []
    for frozen in frozen_models:
        if not isinstance(frozen, Mapping):
            raise TestOnceError("lock manifest contains an invalid model record")
        for key in ("config", "checkpoint"):
            _verify_fingerprint(frozen.get(key), f"{frozen.get('name')} {key}")
        evidence = frozen.get("selection_evidence")
        if evidence is not None:
            _verify_fingerprint(evidence, f"{frozen.get('name')} validation evidence")
        for dependency in frozen.get("selection_dependencies", []):
            _verify_fingerprint(dependency, f"{frozen.get('name')} validation dependency")
        refreshed_model = _audit_model(
            {
                "name": frozen.get("name"),
                "family": frozen.get("family"),
                "config": frozen.get("config", {}).get("path"),
                "checkpoint": frozen.get("checkpoint", {}).get("path"),
                "selection_evidence": (
                    evidence.get("path") if isinstance(evidence, Mapping) else None
                ),
            },
            workspace,
            paths_are_absolute=True,
        )
        refreshed.append(refreshed_model)
    if _canonical_sha256(refreshed) != _canonical_sha256(frozen_models):
        raise TestOnceError(
            "validation provenance changed semantically after the experiment was frozen"
        )
    return manifest


def claim_test_run(
    registry_dir: str | Path,
    experiment_id: str,
) -> Path:
    """Atomically consume the only allowed test attempt after a fresh audit."""

    experiment_id = _validate_experiment_id(experiment_id)
    manifest_path, claim_path, _ = protocol_paths(registry_dir, experiment_id)
    if claim_path.exists():
        raise TestOnceError(
            f"experiment {experiment_id!r} already claimed its one allowed test run"
        )
    manifest = verify_frozen_experiment(registry_dir, experiment_id)
    verified_manifest_sha = sha256_file(manifest_path)
    return _claim_verified_test_run(
        registry_dir,
        experiment_id,
        manifest,
        manifest_path,
        claim_path,
        verified_manifest_sha,
    )


def _claim_verified_test_run(
    registry_dir: str | Path,
    experiment_id: str,
    manifest: Mapping[str, Any],
    manifest_path: Path | None = None,
    claim_path: Path | None = None,
    verified_manifest_sha: str | None = None,
) -> Path:
    if manifest_path is None or claim_path is None:
        manifest_path, claim_path, _ = protocol_paths(registry_dir, experiment_id)
    current_manifest_sha = sha256_file(manifest_path)
    if (
        verified_manifest_sha is not None
        and current_manifest_sha != verified_manifest_sha
    ):
        raise TestOnceError("test-once lock changed between verification and claim")
    claim = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "experiment_id": experiment_id,
        "claimed_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": current_manifest_sha,
        "frozen_payload_sha256": manifest["frozen_payload_sha256"],
        "state": "test_claimed",
        "retry_allowed": False,
    }
    try:
        _atomic_create_json(claim_path, claim)
    except FileExistsError as error:
        raise TestOnceError(
            f"experiment {experiment_id!r} was claimed concurrently; no second run is allowed"
        ) from error
    return claim_path


def run_locked_test_plan(
    registry_dir: str | Path,
    experiment_id: str,
) -> Path:
    """Claim once and execute the exact, shell-free command plan in the lock.

    A failed process still consumes the test attempt.  This is intentional: a
    failure cannot become a channel for adapting hyperparameters to test data.
    """

    manifest = verify_frozen_experiment(registry_dir, experiment_id)
    manifest_path, claim_path, result_path = protocol_paths(registry_dir, experiment_id)
    if claim_path.exists():
        raise TestOnceError(
            f"experiment {experiment_id!r} already claimed its one allowed test run"
        )
    if result_path.exists():
        raise TestOnceError(
            f"experiment {experiment_id!r} has a result without a claim; "
            "the registry is inconsistent"
        )
    claim_path = _claim_verified_test_run(
        registry_dir,
        experiment_id,
        manifest,
        manifest_path,
        claim_path,
        sha256_file(manifest_path),
    )
    workspace = Path(manifest["workspace_root"])
    environment = os.environ.copy()
    environment["VALGRAPHNET_TEST_ONCE_CLAIM"] = str(claim_path)
    environment["VALGRAPHNET_TEST_ONCE_EXPERIMENT_ID"] = str(experiment_id)
    outcomes: list[dict[str, Any]] = []
    overall_status = "completed"
    for command in manifest["test_commands"]:
        started = _utc_now()
        try:
            completed = subprocess.run(
                command["argv"],
                cwd=workspace,
                env=environment,
                check=False,
            )
            returncode: int | None = int(completed.returncode)
            launch_error = None
        except OSError as error:
            returncode = None
            launch_error = f"{type(error).__name__}: {error}"
        outcome = {
            "name": command["name"],
            "models": command["models"],
            "argv_sha256": command["argv_sha256"],
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "returncode": returncode,
        }
        if launch_error is not None:
            outcome["launch_error"] = launch_error
        outcomes.append(outcome)
        if returncode != 0:
            overall_status = "failed_no_retry"
            break
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "experiment_id": experiment_id,
        "claim_sha256": sha256_file(claim_path),
        "status": overall_status,
        "retry_allowed": False,
        "commands": outcomes,
    }
    _atomic_create_json(result_path, result)
    if overall_status != "completed":
        raise TestOnceError(
            "the frozen test plan failed and the one allowed attempt remains consumed"
        )
    return result_path


def protocol_paths(
    registry_dir: str | Path, experiment_id: str
) -> tuple[Path, Path, Path]:
    """Return deterministic manifest, claim, and result paths."""

    experiment_id = _validate_experiment_id(experiment_id)
    registry = Path(registry_dir).resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_id).strip("._-")[:80]
    suffix = hashlib.sha256(experiment_id.encode("utf-8")).hexdigest()[:12]
    stem = f"{slug}-{suffix}"
    return (
        registry / f"{stem}.lock.json",
        registry / f"{stem}.claim.json",
        registry / f"{stem}.result.json",
    )


def _audit_model(
    raw: dict[str, Any],
    workspace: Path,
    *,
    paths_are_absolute: bool = False,
) -> dict[str, Any]:
    name = str(raw.get("name", "")).strip()
    family = str(raw.get("family", "")).strip().lower()
    if not name:
        raise TestOnceError("model name cannot be empty")
    if family not in SUPPORTED_FAMILIES:
        raise TestOnceError(
            f"{name}: family must be one of {sorted(SUPPORTED_FAMILIES)}"
        )
    config_path = _resolve_file(raw.get("config"), workspace, paths_are_absolute)
    checkpoint_path = _resolve_file(
        raw.get("checkpoint"), workspace, paths_are_absolute
    )
    config = _read_mapping(config_path, f"{name} configuration")
    checkpoint = _torch_load(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TestOnceError(f"{name}: checkpoint root must be a mapping")
    embedded = checkpoint.get("config" if family == "chp" else "cfg")
    if not isinstance(embedded, Mapping):
        raise TestOnceError(f"{name}: checkpoint has no embedded configuration")
    config_semantic_sha = _canonical_sha256(config)
    embedded_semantic_sha = _canonical_sha256(embedded)
    if config_semantic_sha != embedded_semantic_sha:
        raise TestOnceError(
            f"{name}: external configuration differs from the checkpoint configuration"
        )

    val_split, test_split = _development_splits(config, family)
    if not val_split or not test_split or val_split == test_split:
        raise TestOnceError(
            f"{name}: validation and test split names must be present and distinct"
        )
    epoch = _nonnegative_int(checkpoint.get("epoch"), f"{name} checkpoint epoch")
    score = _finite(checkpoint.get("score"), f"{name} checkpoint score")

    evidence_path: Path | None = None
    evidence: Any = None
    if raw.get("selection_evidence"):
        evidence_path = _resolve_file(
            raw["selection_evidence"], workspace, paths_are_absolute
        )
        evidence = _read_json(evidence_path, f"{name} validation evidence")
    selection = _validate_family_selection(
        name=name,
        family=family,
        config=config,
        checkpoint=checkpoint,
        val_split=val_split,
        epoch=epoch,
        score=score,
        evidence=evidence,
        workspace=workspace,
    )
    dependencies = _selection_dependencies(config, family, workspace, val_split)
    return {
        "name": name,
        "family": family,
        "config": {
            **_fingerprint(config_path),
            "semantic_sha256": config_semantic_sha,
        },
        "checkpoint": _fingerprint(checkpoint_path),
        "selection_evidence": (
            _fingerprint(evidence_path) if evidence_path is not None else None
        ),
        "selection_dependencies": dependencies,
        "selection": {
            "split": val_split,
            "test_split_name": test_split,
            "epoch": epoch,
            "score": score,
            **selection,
        },
    }


def _validate_family_selection(
    *,
    name: str,
    family: str,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    val_split: str,
    epoch: int,
    score: float,
    evidence: Any,
    workspace: Path,
) -> dict[str, Any]:
    if family == "native":
        if checkpoint.get("model_family") is not None:
            raise TestOnceError(f"{name}: native checkpoint has an unexpected model_family")
        if "scheduler" not in checkpoint:
            raise TestOnceError(f"{name}: checkpoint is not a native PhysicsNeMo artifact")
        if evidence is None:
            raise TestOnceError(
                f"{name}: native checkpoints require a validation metrics artifact"
            )
        _validate_metrics_evidence(evidence, val_split, name)
        return {"criterion": "validation_loss", "evidence": "external_val_metrics"}

    if family in {"fair", "multiscale"}:
        expected_family = (
            "fair_deforming_plate_mgn"
            if family == "fair"
            else "two_level_bistride_deforming_plate_mgn"
        )
        if checkpoint.get("model_family") != expected_family:
            raise TestOnceError(f"{name}: checkpoint model_family is inconsistent")
        if checkpoint.get("checkpoint_metric") != "four_metric_native_ratio_minimax":
            raise TestOnceError(f"{name}: checkpoint was not selected by minimax rollout")
        if family == "multiscale" and str(checkpoint.get("checkpoint_split")) != val_split:
            raise TestOnceError(f"{name}: checkpoint_split is not the validation split")
        metrics = _primary_metrics(checkpoint.get("rollout_metrics"), name)
        reference = _primary_metrics(checkpoint.get("native_reference"), name)
        _require_positive_reference(reference, name)
        expected_score = max(metrics[key] / reference[key] for key in PRIMARY_METRICS)
        _require_same_score(score, expected_score, name)
        _require_best_payload(checkpoint, score, name)
        return {"criterion": "four_metric_native_ratio_minimax", "metrics": metrics}

    if family == "repo":
        checkpoint_family = checkpoint.get("model_family")
        if checkpoint_family not in {None, REPOSITORY_MODEL_FAMILY}:
            raise TestOnceError(f"{name}: repository checkpoint family is ambiguous")
        if strict_checkpoint_provenance(config):
            if checkpoint_family != REPOSITORY_MODEL_FAMILY:
                raise TestOnceError(
                    f"{name}: strict repository checkpoint model_family is missing"
                )
            if (
                int(checkpoint.get("schema_version", 0))
                != REPOSITORY_CHECKPOINT_SCHEMA_VERSION
                or checkpoint.get("artifact_type") != REPOSITORY_ARTIFACT_TYPE
                or checkpoint.get("artifact_role") != "best"
            ):
                raise TestOnceError(
                    f"{name}: strict repository checkpoint schema is inconsistent"
                )
            provenance = checkpoint.get("provenance")
            if not isinstance(provenance, Mapping) or provenance.get(
                "config_sha256"
            ) != config_sha256(checkpoint.get("cfg", {})):
                raise TestOnceError(
                    f"{name}: strict repository checkpoint provenance is inconsistent"
                )
            data_contract = provenance.get("data_contract")
            if (
                not isinstance(data_contract, Mapping)
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(data_contract.get("fingerprint_sha256", "")),
                )
                or data_contract.get("test_content_accessed") is not False
            ):
                raise TestOnceError(
                    f"{name}: strict repository data provenance is inconsistent"
                )
        if "normalizers" not in checkpoint or "output_dim" not in checkpoint:
            raise TestOnceError(f"{name}: checkpoint is not a repository-model artifact")
        training = config.get("training", {})
        if not isinstance(training, Mapping):
            raise TestOnceError(f"{name}: training configuration is invalid")
        if str(training.get("checkpoint_metric", "")).lower() != "rollout":
            raise TestOnceError(f"{name}: repository checkpoint is not rollout-selected")
        if (
            str(training.get("rollout_checkpoint_score_mode", "")).lower()
            != "four_metric_native_ratio_minimax"
        ):
            raise TestOnceError(f"{name}: repository checkpoint is not minimax-selected")
        if not isinstance(evidence, list):
            raise TestOnceError(f"{name}: repository checkpoint requires history.json evidence")
        rows = [row for row in evidence if isinstance(row, Mapping) and row.get("epoch") == epoch]
        if len(rows) != 1:
            raise TestOnceError(f"{name}: history must contain exactly one selected epoch")
        rollout = rows[0].get("rollout_val")
        metrics = _primary_metrics(rollout, name)
        history_score = _finite(
            rollout.get("score") if isinstance(rollout, Mapping) else None,
            f"{name} history rollout score",
        )
        _require_same_score(score, history_score, name)
        return {"criterion": "four_metric_native_ratio_minimax", "metrics": metrics}

    if checkpoint.get("architecture") != "CHP-GNS":
        raise TestOnceError(f"{name}: checkpoint architecture is not CHP-GNS")
    if str(checkpoint.get("scientific_gate_status")) not in {"passed", "not_required"}:
        raise TestOnceError(f"{name}: CHP scientific gates have not passed")
    metrics = _primary_metrics(checkpoint.get("rollout_metrics"), name)
    reference_mode = str(
        _nested(config, "validation.checkpoint_reference_mode", "auto")
    ).lower()
    if reference_mode == "absolute_validation":
        expected_score = max(metrics.values())
    else:
        reference_path = _nested(config, "validation.native_reference_file", None)
        if not reference_path:
            raise TestOnceError(f"{name}: CHP minimax selection has no native reference")
        reference = _primary_metrics(
            _load_reference_payload(reference_path, workspace), name
        )
        _require_positive_reference(reference, name)
        expected_score = max(metrics[key] / reference[key] for key in PRIMARY_METRICS)
    _require_same_score(score, expected_score, name)
    _require_best_payload(checkpoint, score, name)
    return {
        "criterion": (
            "absolute_validation_minimax"
            if reference_mode == "absolute_validation"
            else "four_metric_native_ratio_minimax"
        ),
        "metrics": metrics,
        "scientific_gate_status": str(checkpoint["scientific_gate_status"]),
    }


def _selection_dependencies(
    config: Mapping[str, Any], family: str, workspace: Path, val_split: str
) -> list[dict[str, Any]]:
    value = _nested(config, "validation.native_reference_file", None)
    if not value:
        if family in {"fair", "multiscale", "repo"}:
            raise TestOnceError(
                f"{family}: minimax checkpoint has no provenance-rich native "
                "validation reference file"
            )
        return []
    path = _resolve_file(value, workspace, False)
    payload = _read_json(path, "native validation reference")
    if isinstance(payload, Mapping) and isinstance(payload.get("evaluation"), Mapping):
        actual = str(payload["evaluation"].get("split", ""))
        if actual != val_split:
            raise TestOnceError(
                f"native reference split mismatch: expected {val_split!r}, got {actual!r}"
            )
    elif family in {"fair", "multiscale", "repo", "chp"}:
        raise TestOnceError("native reference is missing validation split provenance")
    return [{**_fingerprint(path), "role": "native_validation_reference"}]


def _load_reference_payload(path_value: Any, workspace: Path) -> Mapping[str, Any]:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = workspace / path
    payload = _read_json(path, "native validation reference")
    if not isinstance(payload, Mapping):
        raise TestOnceError("native validation reference must be an object")
    for key in ("summary", "aggregate", "rollout"):
        child = payload.get(key)
        if isinstance(child, Mapping):
            return child
    return payload


def _validate_metrics_evidence(payload: Any, val_split: str, name: str) -> None:
    if not isinstance(payload, Mapping):
        raise TestOnceError(f"{name}: validation metrics evidence must be an object")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise TestOnceError(f"{name}: validation evidence has no split provenance")
    if str(evaluation.get("split", "")) != val_split:
        raise TestOnceError(f"{name}: validation evidence was not evaluated on {val_split!r}")
    if str(evaluation.get("split", "")).lower().startswith("test"):
        raise TestOnceError(f"{name}: test-derived checkpoint evidence is forbidden")


def _primary_metrics(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TestOnceError(f"{name}: selected checkpoint has no rollout metrics")
    result: dict[str, float] = {}
    for key in PRIMARY_METRICS:
        metric = _finite(value.get(key), f"{name} {key}")
        if metric < 0.0:
            raise TestOnceError(f"{name}: {key} must be non-negative")
        result[key] = metric
    return result


def _development_splits(
    config: Mapping[str, Any], family: str
) -> tuple[str, str]:
    data = config.get("data", {})
    if not isinstance(data, Mapping):
        raise TestOnceError("configuration data section must be an object")
    case_backed_native = family == "native" and data.get("case_dir") is not None
    val_key = "val_case_split" if case_backed_native else "val_split"
    test_key = "test_case_split" if case_backed_native else "test_split"
    return str(data.get(val_key, "")).strip(), str(data.get(test_key, "")).strip()


def _validate_test_commands(value: Any, model_names: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TestOnceError("test_commands must freeze at least one guarded command")
    result: list[dict[str, Any]] = []
    covered: set[str] = set()
    command_names: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise TestOnceError("every test command must be an object")
        name = str(item.get("name", "")).strip()
        if not name or name in command_names:
            raise TestOnceError("test command names must be non-empty and unique")
        command_names.add(name)
        models = item.get("models")
        if not isinstance(models, list) or not models:
            raise TestOnceError(f"test command {name!r} must name its models")
        normalized_models = [str(model) for model in models]
        unknown = set(normalized_models) - model_names
        if unknown:
            raise TestOnceError(f"test command {name!r} references unknown models: {unknown}")
        argv = item.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(part, str) or not part for part in argv)
        ):
            raise TestOnceError(f"test command {name!r} argv must be non-empty strings")
        covered.update(normalized_models)
        result.append(
            {
                "name": name,
                "models": normalized_models,
                "argv": list(argv),
                "argv_sha256": _canonical_sha256(argv),
            }
        )
    missing = model_names - covered
    if missing:
        raise TestOnceError(f"frozen test commands do not cover models: {sorted(missing)}")
    return result


def _fingerprint_paths(
    values: Any, workspace: Path, label: str
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise TestOnceError(f"{label} list is invalid")
    return [_fingerprint(_resolve_file(value, workspace, False)) for value in values]


def _fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise TestOnceError(f"frozen file does not exist: {path}")
    stat = path.stat()
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": stat.st_size}


def _verify_fingerprint(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise TestOnceError(f"frozen {label} fingerprint is invalid")
    path = Path(str(value.get("path", "")))
    if not path.is_file():
        raise TestOnceError(f"frozen {label} disappeared: {path}")
    actual_size = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_size != int(value.get("size_bytes", -1)) or actual_sha != value.get("sha256"):
        raise TestOnceError(f"frozen {label} drift detected: {path}")


def _resolve_file(value: Any, workspace: Path, already_absolute: bool) -> Path:
    if value is None or not str(value).strip():
        raise TestOnceError("required frozen file path is missing")
    path = Path(str(value))
    if not path.is_absolute() and not already_absolute:
        path = workspace / path
    path = path.resolve()
    if not path.is_file():
        raise TestOnceError(f"required frozen file does not exist: {path}")
    return path


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise TestOnceError(f"{label} does not exist: {path}")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise TestOnceError(f"cannot parse {label}: {path}") from error
    if not isinstance(value, dict):
        raise TestOnceError(f"{label} must contain an object")
    return value


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TestOnceError(f"cannot parse {label}: {path}") from error


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as error:
        raise TestOnceError(f"cannot load checkpoint metadata: {path}") from error


def _nested(value: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = value
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TestOnceError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise TestOnceError(f"{label} must be finite")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TestOnceError(f"{label} must be a non-negative integer") from error
    if result < 0:
        raise TestOnceError(f"{label} must be a non-negative integer")
    return result


def _require_same_score(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-9):
        raise TestOnceError(
            f"{name}: checkpoint score {actual} does not match validation evidence {expected}"
        )


def _require_positive_reference(reference: Mapping[str, float], name: str) -> None:
    nonpositive = [key for key, value in reference.items() if value <= 0.0]
    if nonpositive:
        raise TestOnceError(
            f"{name}: native minimax references must be positive: {nonpositive}"
        )


def _require_best_payload(checkpoint: Mapping[str, Any], score: float, name: str) -> None:
    best = _finite(checkpoint.get("best_score"), f"{name} best checkpoint score")
    _require_same_score(score, best, name)


def _validate_experiment_id(value: Any) -> str:
    experiment_id = str(value or "").strip()
    invalid = (
        not experiment_id
        or len(experiment_id) > 160
        or any(ord(char) < 32 for char in experiment_id)
    )
    if invalid:
        raise TestOnceError("experiment_id must be a non-empty printable string <=160 chars")
    return experiment_id


def _validate_manifest_identity(manifest: Mapping[str, Any], experiment_id: str) -> None:
    if int(manifest.get("schema_version", 0)) != LOCK_SCHEMA_VERSION:
        raise TestOnceError("unsupported test-once lock schema")
    if manifest.get("protocol") != PROTOCOL_NAME:
        raise TestOnceError("file is not a ValGraphNet test-once lock")
    if str(manifest.get("experiment_id")) != experiment_id:
        raise TestOnceError("test-once lock experiment id mismatch")
    expected = _canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at_utc", "frozen_payload_sha256"}
        }
    )
    if manifest.get("frozen_payload_sha256") != expected:
        raise TestOnceError("test-once lock manifest was modified")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
