"""Fail-closed provenance for repository ValGraphNet checkpoints.

The historical 200-frame experiments predate checkpoint schemas.  They remain
loadable when a configuration does not opt into ``strict_v2``.  Formal
full-trajectory experiments opt in explicitly and bind every resume/export to
the same configuration and train/validation data contract.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from valgraphnet.config import get_cfg
from valgraphnet.data.case import read_split_file


CHECKPOINT_SCHEMA_VERSION = 2
DATA_CONTRACT_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
ARTIFACT_TYPE = "valgraphnet.training_checkpoint"
MODEL_FAMILY = "repository_valgraphnet"
STRICT_POLICY = "strict_v2"
LEGACY_POLICY = "legacy_compatible"


def checkpoint_policy(cfg: Mapping[str, Any]) -> str:
    """Return the explicit checkpoint policy, defaulting to legacy behavior."""

    policy = str(
        get_cfg(dict(cfg), "provenance.checkpoint_policy", LEGACY_POLICY)
    ).lower()
    if policy not in {STRICT_POLICY, LEGACY_POLICY}:
        raise ValueError(
            "provenance.checkpoint_policy must be strict_v2 or legacy_compatible"
        )
    return policy


def strict_checkpoint_provenance(cfg: Mapping[str, Any]) -> bool:
    return checkpoint_policy(cfg) == STRICT_POLICY


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_sha256(cfg: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(cfg))


def resume_config_sha256(cfg: Mapping[str, Any]) -> str:
    """Hash semantic run configuration while allowing resume path relocation."""

    normalized = deepcopy(dict(cfg))
    training = normalized.get("training")
    if isinstance(training, dict):
        training.pop("resume_from", None)
    return canonical_sha256(normalized)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_structure_sha256(state_dict: Mapping[str, Any]) -> str:
    structure = []
    for name, value in sorted(state_dict.items()):
        if torch.is_tensor(value):
            structure.append(
                {
                    "name": str(name),
                    "shape": [int(item) for item in value.shape],
                    "dtype": str(value.dtype),
                }
            )
        else:
            structure.append({"name": str(name), "type": type(value).__name__})
    return canonical_sha256(structure)


def build_repo_data_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint train/validation inputs without opening held-out test arrays."""

    root_value = get_cfg(dict(cfg), "data.root", get_cfg(dict(cfg), "data.case_dir"))
    split_value = get_cfg(
        dict(cfg), "data.split_file", get_cfg(dict(cfg), "data.case_split_file")
    )
    if root_value is None or split_value is None:
        raise ValueError("strict_v2 requires data.root and data.split_file")
    root = Path(root_value)
    split_file = Path(split_value)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"split file does not exist: {split_file}")
    with split_file.open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)
    if not isinstance(split_payload, dict):
        raise ValueError("split file must contain a JSON object")

    split_names = {
        "train": str(get_cfg(dict(cfg), "data.train_split", "train")),
        "val": str(get_cfg(dict(cfg), "data.val_split", "val")),
        "test": str(get_cfg(dict(cfg), "data.test_split", "test")),
    }
    split_ids: dict[str, list[str]] = {}
    for role, split_name in split_names.items():
        split_ids[role] = read_split_file(split_file, split_name)
    _validate_disjoint_splits(split_ids)

    expected_counts = get_cfg(dict(cfg), "provenance.expected_split_counts", {})
    if expected_counts is not None and not isinstance(expected_counts, dict):
        raise ValueError("provenance.expected_split_counts must be a mapping")
    for role, expected in (expected_counts or {}).items():
        if role in split_ids and len(split_ids[role]) != int(expected):
            raise ValueError(
                f"{role} split contains {len(split_ids[role])} cases; expected {expected}"
            )

    expected_frames = get_cfg(dict(cfg), "provenance.expected_frames", None)
    expected_case_schema = get_cfg(
        dict(cfg), "provenance.expected_case_schema_version", None
    )
    content_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for role in ("train", "val"):
        for case_id in split_ids[role]:
            record = _case_contract(root / case_id, case_id)
            if expected_frames is not None and record["num_frames"] != int(
                expected_frames
            ):
                raise ValueError(
                    f"{case_id} has {record['num_frames']} frames; "
                    f"expected {int(expected_frames)}"
                )
            if expected_case_schema is not None and record[
                "case_schema_version"
            ] != int(expected_case_schema):
                raise ValueError(
                    f"{case_id} case schema is {record['case_schema_version']}; "
                    f"expected {int(expected_case_schema)}"
                )
            content_records[role].append(record)

    identity = {
        "schema_version": DATA_CONTRACT_SCHEMA_VERSION,
        "split_file_sha256": sha256_file(split_file),
        "split_payload_sha256": canonical_sha256(split_payload),
        "split_names": split_names,
        "split_case_ids": split_ids,
        "split_case_ids_sha256": {
            role: canonical_sha256(ids) for role, ids in split_ids.items()
        },
        "split_counts": {role: len(ids) for role, ids in split_ids.items()},
        "content_records": content_records,
        "test_content_accessed": False,
    }
    return {
        **identity,
        # The root is useful for an audit but intentionally excluded from the
        # identity hash so an unchanged dataset can be moved as a unit.
        "data_root": str(root.resolve()),
        "split_file": str(split_file.resolve()),
        "fingerprint_sha256": canonical_sha256(identity),
    }


def compact_data_contract(contract: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if contract is None:
        return None
    return {
        "schema_version": int(contract["schema_version"]),
        "fingerprint_sha256": str(contract["fingerprint_sha256"]),
        "split_file_sha256": str(contract["split_file_sha256"]),
        "split_payload_sha256": str(contract["split_payload_sha256"]),
        "split_counts": dict(contract["split_counts"]),
        "split_case_ids_sha256": dict(contract["split_case_ids_sha256"]),
        "test_content_accessed": bool(contract["test_content_accessed"]),
    }


def checkpoint_metadata(
    cfg: Mapping[str, Any],
    model_state: Mapping[str, Any],
    data_contract: Mapping[str, Any] | None,
    output_dim: int,
    *,
    artifact_role: str,
) -> dict[str, Any]:
    if strict_checkpoint_provenance(cfg) and data_contract is None:
        raise ValueError("strict_v2 checkpoints require a data contract")
    stress_dim = max(int(output_dim) - 9, 0)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_role": str(artifact_role),
        "model_family": MODEL_FAMILY,
        "architecture": str(get_cfg(dict(cfg), "model.type", "hybrid")),
        "output_contract": {
            "schema_version": 1,
            "fields": {
                "delta_u": 3,
                "delta_v": 3,
                "accel": 3,
                "stress": stress_dim,
            },
            "output_dim": int(output_dim),
            "state_update": "independent_delta_u_delta_v",
        },
        "provenance": {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "checkpoint_policy": checkpoint_policy(cfg),
            "config_sha256": config_sha256(cfg),
            "resume_config_sha256": resume_config_sha256(cfg),
            "model_structure_sha256": model_structure_sha256(model_state),
            "data_contract": compact_data_contract(data_contract),
        },
    }


def validate_repo_checkpoint(
    checkpoint: Mapping[str, Any],
    cfg: Mapping[str, Any],
    data_contract: Mapping[str, Any] | None,
    *,
    purpose: str,
    source: str | Path | None = None,
) -> None:
    """Validate a checkpoint when the current configuration opts into strict_v2."""

    if not strict_checkpoint_provenance(cfg):
        return
    label = str(source) if source is not None else "checkpoint"
    if int(checkpoint.get("schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{label}: unsupported repository checkpoint schema")
    if checkpoint.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"{label}: checkpoint artifact type mismatch")
    if checkpoint.get("model_family") != MODEL_FAMILY:
        raise ValueError(f"{label}: checkpoint model family mismatch")
    if purpose == "export" and checkpoint.get("artifact_role") != "best":
        raise ValueError(f"{label}: formal export requires a best checkpoint")

    embedded_cfg = checkpoint.get("cfg")
    if not isinstance(embedded_cfg, dict):
        raise ValueError(f"{label}: checkpoint is missing its embedded config")
    if checkpoint.get("architecture") != str(
        get_cfg(embedded_cfg, "model.type", "hybrid")
    ):
        raise ValueError(f"{label}: checkpoint architecture metadata mismatch")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict) or int(
        provenance.get("schema_version", 0)
    ) != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"{label}: checkpoint provenance is missing or unsupported")
    if provenance.get("config_sha256") != config_sha256(embedded_cfg):
        raise ValueError(f"{label}: embedded config fingerprint is inconsistent")
    if provenance.get("resume_config_sha256") != resume_config_sha256(embedded_cfg):
        raise ValueError(f"{label}: embedded resume config fingerprint is inconsistent")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError(f"{label}: checkpoint is missing model state")
    if provenance.get("model_structure_sha256") != model_structure_sha256(model_state):
        raise ValueError(f"{label}: model structure fingerprint is inconsistent")
    output_contract = checkpoint.get("output_contract")
    if not isinstance(output_contract, dict) or int(
        output_contract.get("schema_version", 0)
    ) != 1:
        raise ValueError(f"{label}: checkpoint output contract is unsupported")
    if int(output_contract.get("output_dim", -1)) != int(
        checkpoint.get("output_dim", -2)
    ):
        raise ValueError(f"{label}: checkpoint output dimension is inconsistent")

    if purpose == "warm_start":
        policy = str(
            get_cfg(dict(cfg), "training.initial_checkpoint_policy", "")
        ).lower()
        if policy != "compatible_weights_only":
            raise ValueError(
                "strict_v2 warm-start requires "
                "training.initial_checkpoint_policy=compatible_weights_only"
            )
        return
    if purpose not in {"resume", "export"}:
        raise ValueError(f"unsupported checkpoint validation purpose: {purpose}")
    if provenance.get("resume_config_sha256") != resume_config_sha256(cfg):
        raise ValueError(f"{label}: checkpoint config does not match the current run")
    if data_contract is None:
        raise ValueError("strict_v2 validation requires a current data contract")
    stored_contract = provenance.get("data_contract")
    if not isinstance(stored_contract, dict):
        raise ValueError(f"{label}: checkpoint is missing its data contract")
    if stored_contract.get("test_content_accessed") is not False:
        raise ValueError(f"{label}: training data contract accessed held-out test content")
    if stored_contract.get("fingerprint_sha256") != data_contract.get(
        "fingerprint_sha256"
    ):
        raise ValueError(f"{label}: checkpoint data fingerprint does not match")


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically replace a checkpoint without corrupting the previous file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _case_contract(case_dir: Path, expected_case_id: str) -> dict[str, Any]:
    if not case_dir.is_dir():
        raise FileNotFoundError(f"missing case directory: {case_dir}")
    metadata_path = case_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata_sha256 = sha256_file(metadata_path)
    else:
        metadata = {}
        metadata_sha256 = None
    case_id = str(metadata.get("case_id", case_dir.name))
    if case_id != str(expected_case_id):
        raise ValueError(
            f"case id mismatch: split has {expected_case_id}, metadata has {case_id}"
        )

    arrays = []
    for path in sorted(case_dir.glob("*.npy")):
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        arrays.append(
            {
                "name": path.name,
                "shape": [int(item) for item in array.shape],
                "dtype": str(array.dtype),
                "size_bytes": int(path.stat().st_size),
                # Formal strict_v2 binds content, not merely shape.  Hash every
                # train/validation array so same-size label or geometry swaps
                # cannot pass resume/export provenance checks.
                "sha256": sha256_file(path),
            }
        )
        del array
    json_files = [
        {
            "name": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(case_dir.glob("*.json"))
    ]
    by_name = {item["name"]: item for item in arrays}
    for required in ("nodes.npy", "times.npy", "U.npy", "S.npy"):
        if required not in by_name:
            raise FileNotFoundError(f"{case_dir}: missing {required}")
    u_shape = by_name["U.npy"]["shape"]
    s_shape = by_name["S.npy"]["shape"]
    nodes_shape = by_name["nodes.npy"]["shape"]
    times_shape = by_name["times.npy"]["shape"]
    if (
        len(u_shape) != 3
        or len(s_shape) != 3
        or len(nodes_shape) != 2
        or len(times_shape) != 1
    ):
        raise ValueError(f"{case_dir}: invalid U/S/nodes array rank")
    if (
        int(u_shape[0]) != int(s_shape[0])
        or int(u_shape[0]) != int(times_shape[0])
        or int(u_shape[1]) != int(nodes_shape[0])
        or int(s_shape[1]) != int(nodes_shape[0])
    ):
        raise ValueError(f"{case_dir}: inconsistent trajectory array shapes")
    return {
        "case_id": case_id,
        "case_schema_version": int(metadata.get("schema_version", 0)),
        "source": metadata.get("source"),
        "num_frames": int(u_shape[0]),
        "num_nodes": int(nodes_shape[0]),
        "stress_dim": int(s_shape[-1]),
        "num_cells": int(by_name.get("cells.npy", {"shape": [0]})["shape"][0]),
        "metadata_sha256": metadata_sha256,
        "arrays": arrays,
        "json_files": json_files,
    }


def _validate_disjoint_splits(split_ids: Mapping[str, list[str]]) -> None:
    for role, ids in split_ids.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"{role} split contains duplicate case ids")
    roles = list(split_ids)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = set(split_ids[left]).intersection(split_ids[right])
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(
                    f"{left}/{right} splits overlap (for example {example})"
                )
