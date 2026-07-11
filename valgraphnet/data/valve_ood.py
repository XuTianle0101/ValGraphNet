"""Strict, auditable geometry/material OOD splits for exported valve cases.

The splitter intentionally uses explicit metadata identifiers.  It never infers
geometry or material identity from directory names, meshes, or floating-point
arrays because those heuristics can silently leak replicated simulations across
train and held-out partitions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from valgraphnet.config import get_cfg
from valgraphnet.data.case import ValveCase


SPLIT_SCHEMA_VERSION = 1
SPLIT_STRATEGY = "geometry_material_combination_disjoint"
CANONICAL_STRESS_COMPONENTS = ("S11", "S22", "S33", "S12", "S13", "S23")
CANONICAL_STRAIN_COMPONENTS = ("LE11", "LE22", "LE33", "LE12", "LE13", "LE23")


def build_strict_ood_split(
    data_root: str | Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    geometry_key: str = "geometry_id",
    material_key: str = "material_id",
) -> dict[str, Any]:
    """Build a deterministic split with disjoint geometry/material pairs.

    Every case carrying the same ``(geometry_id, material_id)`` pair is assigned
    to exactly one partition.  Individual geometry or material identifiers may
    still occur in multiple partitions; this is a *combinatorial* OOD split, and
    the returned audit explicitly reports factor overlap with the training set.
    """

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Valve case root does not exist: {root}")
    _validate_fractions(validation_fraction, test_fraction)
    geometry_key = _validate_metadata_key(geometry_key, "geometry_key")
    material_key = _validate_metadata_key(material_key, "material_key")

    records = _read_case_records(root, geometry_key, material_key)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_key = (
            _typed_metadata_token(record["geometry_id"]),
            _typed_metadata_token(record["material_id"]),
        )
        groups[group_key].append(record)
    if len(groups) < 3:
        raise ValueError(
            "A strict train/val/test OOD split needs at least three distinct "
            f"geometry/material combinations; found {len(groups)}"
        )

    ordered_groups = sorted(
        groups,
        key=lambda group: (
            _assignment_digest(seed, group),
            group[0],
            group[1],
        ),
    )
    num_val, num_test = _held_out_group_counts(
        len(ordered_groups), validation_fraction, test_fraction
    )
    assignment: dict[str, list[tuple[str, str]]] = {
        "test": ordered_groups[:num_test],
        "val": ordered_groups[num_test : num_test + num_val],
        "train": ordered_groups[num_test + num_val :],
    }

    split_case_ids: dict[str, list[str]] = {}
    group_audit: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        case_ids: list[str] = []
        audit_rows: list[dict[str, Any]] = []
        for group in assignment[split]:
            members = sorted(groups[group], key=lambda value: value["case_id"])
            member_ids = [str(value["case_id"]) for value in members]
            case_ids.extend(member_ids)
            audit_rows.append(
                {
                    "geometry_id": members[0]["geometry_id"],
                    "material_id": members[0]["material_id"],
                    "combination_sha256": _combination_digest(group),
                    "case_ids": member_ids,
                }
            )
        split_case_ids[split] = sorted(case_ids)
        group_audit[split] = sorted(
            audit_rows, key=lambda value: value["combination_sha256"]
        )

    _assert_combination_disjoint(group_audit)
    factor_audit = _factor_overlap_audit(group_audit)
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "strategy": SPLIT_STRATEGY,
        "seed": int(seed),
        "metadata_keys": {
            "geometry": geometry_key,
            "material": material_key,
        },
        "requested_fractions": {
            "train": float(1.0 - validation_fraction - test_fraction),
            "val": float(validation_fraction),
            "test": float(test_fraction),
        },
        "train": split_case_ids["train"],
        "val": split_case_ids["val"],
        "test": split_case_ids["test"],
        "groups": group_audit,
        "audit": {
            "combination_disjoint": True,
            "num_cases": len(records),
            "num_combinations": len(groups),
            "case_counts": {
                split: len(split_case_ids[split]) for split in ("train", "val", "test")
            },
            "combination_counts": {
                split: len(group_audit[split]) for split in ("train", "val", "test")
            },
            "factor_overlap_with_train": factor_audit,
        },
    }


def write_strict_ood_split(
    data_root: str | Path,
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and write a canonical JSON split, returning the same payload."""

    payload = build_strict_ood_split(data_root, **kwargs)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_case_requirements(case: ValveCase, cfg: Mapping[str, Any]) -> None:
    """Enforce opt-in CHP Valve data requirements declared by a config."""

    requirements = get_cfg(dict(cfg), "data.requirements", None)
    if requirements is None:
        return
    if not isinstance(requirements, Mapping):
        raise ValueError("data.requirements must be a mapping")

    errors: list[str] = []
    required_files = requirements.get("required_files", [])
    if not isinstance(required_files, list) or not all(
        isinstance(value, str) and value for value in required_files
    ):
        raise ValueError("data.requirements.required_files must be a list of file names")
    for name in required_files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe required case file path: {name}")
        if not (case.root / relative).is_file():
            errors.append(f"missing required file {name}")

    metadata_keys = requirements.get("required_metadata", [])
    if not isinstance(metadata_keys, list) or not all(
        isinstance(value, str) and value for value in metadata_keys
    ):
        raise ValueError("data.requirements.required_metadata must be a list of dotted keys")
    for key in metadata_keys:
        try:
            value = _lookup_metadata(case.metadata, key)
            _normalize_metadata_id(value, key)
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))

    if bool(requirements.get("material_feature_names", False)):
        try:
            names = _normalize_ordered_names(
                _lookup_metadata(case.metadata, "material_feature_names"),
                "material_feature_names",
            )
            if len(names) != case.material_features.shape[1]:
                errors.append(
                    "metadata material_feature_names length must equal "
                    f"material_features.npy width ({case.material_features.shape[1]})"
                )
            material_names = _normalize_ordered_names(
                case.material.get("material_feature_names"),
                "material.json material_feature_names",
            )
            if material_names != names:
                errors.append(
                    "material.json and metadata.json must declare the same ordered "
                    "material_feature_names"
                )
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))

    _validate_component_requirement(
        case.metadata,
        requirements,
        "stress_tensor_components",
        CANONICAL_STRESS_COMPONENTS,
        errors,
    )
    _validate_component_requirement(
        case.metadata,
        requirements,
        "strain_tensor_components",
        CANONICAL_STRAIN_COMPONENTS,
        errors,
    )

    if bool(requirements.get("full_cell_stress_tensor", False)):
        expected = (case.num_steps, case.num_cells, 6)
        if case.cell_stress.shape != expected:
            errors.append(
                f"S_cell.npy must contain complete symmetric tensors with shape {expected}; "
                f"found {case.cell_stress.shape}"
            )
        elif not _array_is_finite(case.cell_stress):
            errors.append("S_cell.npy contains non-finite values")

    if bool(requirements.get("full_integration_point_stress_tensor", False)):
        stress = case.integration_point_stress
        mask = case.integration_point_mask
        valid_shape = (
            stress.ndim == 4
            and stress.shape[:2] == (case.num_steps, case.num_cells)
            and stress.shape[2] > 0
            and stress.shape[3] == 6
            and mask.shape == stress.shape[:3]
        )
        if not valid_shape:
            errors.append(
                "S_integration_point.npy and integration_point_mask.npy must have "
                "shapes [T,M,I,6] and [T,M,I] with I>0"
            )
        else:
            coverage = np.asarray(mask).any(axis=2)
            if not bool(coverage.all()):
                errors.append("integration-point stress is missing for at least one frame/cell")
            elif not _masked_tensor_is_finite(stress, mask):
                errors.append("valid integration-point stress entries contain non-finite values")

    if bool(requirements.get("full_cell_strain_tensor", False)):
        expected = (case.num_steps, case.num_cells, 6)
        if case.cell_strain.shape != expected:
            errors.append(
                f"LE_cell.npy must contain complete symmetric tensors with shape {expected}; "
                f"found {case.cell_strain.shape}"
            )
        elif not _array_is_finite(case.cell_strain):
            errors.append("LE_cell.npy contains non-finite values")

    if bool(requirements.get("material_json", False)) and not case.material:
        errors.append("material.json must contain a non-empty object")
    if bool(requirements.get("material_features", False)):
        if (
            case.material_features.shape[0] != case.num_cells
            or case.material_features.shape[1] == 0
        ):
            errors.append("material_features.npy must have shape [M,P] with P>0")
        elif not _array_is_finite(case.material_features):
            errors.append("material_features.npy contains non-finite values")
    if bool(requirements.get("density", False)):
        if case.density.shape != (case.num_cells, 1):
            errors.append(f"density.npy must have shape ({case.num_cells}, 1)")
        elif not _array_is_finite(case.density) or bool(np.any(case.density <= 0.0)):
            errors.append("density.npy must contain finite, strictly positive cell densities")
    if bool(requirements.get("fiber_direction", False)):
        if case.fiber_direction.shape != (case.num_cells, 3):
            errors.append(f"fiber_direction.npy must have shape ({case.num_cells}, 3)")
        elif not _array_is_finite(case.fiber_direction):
            errors.append("fiber_direction.npy contains non-finite values")
        elif bool(np.any(np.linalg.norm(case.fiber_direction, axis=1) <= 1.0e-8)):
            errors.append("fiber_direction.npy must define a non-zero direction for every cell")

    expected_frames = requirements.get("num_frames")
    if expected_frames is not None and case.num_steps != int(expected_frames):
        errors.append(
            f"case must contain exactly {int(expected_frames)} frames; found {case.num_steps}"
        )
    if bool(requirements.get("explicit_contact_surface_mask", False)):
        mask = np.asarray(case.contact_surface_mask, dtype=bool)
        if mask.shape != (case.num_nodes,) or not bool(mask.any()):
            errors.append(
                "contact_surface_mask.npy must explicitly select at least one full-node surface"
            )
    if bool(requirements.get("prescribed_contact_surface", False)):
        contact = np.asarray(case.contact_surface_mask, dtype=bool)
        prescribed = np.asarray(case.prescribed_mask, dtype=bool)
        if contact.shape != (case.num_nodes,) or not bool(prescribed.any()):
            errors.append("a prescribed contact body is required")
        elif not bool(np.all(contact[prescribed])):
            errors.append("every prescribed indenter node must be in contact_surface_mask.npy")
        elif not bool(np.any(contact & ~prescribed)):
            errors.append("contact_surface_mask.npy must also select deformable contact nodes")
    if bool(requirements.get("zero_pressure_mask", False)) and bool(
        np.asarray(case.pressure_mask, dtype=bool).any()
    ):
        errors.append("pressure_mask.npy must be all false for displacement-driven HyperContact")
    required_time_semantics = requirements.get("time_semantics")
    if required_time_semantics is not None and case.metadata.get("time_semantics") != str(
        required_time_semantics
    ):
        errors.append(
            "metadata time_semantics must equal "
            f"{str(required_time_semantics)!r}"
        )

    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(f"{case.root}: CHP Valve data requirements failed:\n  - {details}")


def validate_case_collection_requirements(
    cases: list[ValveCase], cfg: Mapping[str, Any]
) -> None:
    """Reject cross-case material feature schema drift before training."""

    requirements = get_cfg(dict(cfg), "data.requirements", None)
    if not isinstance(requirements, Mapping) or not bool(
        requirements.get("material_feature_names", False)
    ):
        return
    expected: list[str] | None = None
    expected_case = ""
    for case in cases:
        names = _normalize_ordered_names(
            _lookup_metadata(case.metadata, "material_feature_names"),
            "material_feature_names",
        )
        if expected is None:
            expected = names
            expected_case = case.case_id
        elif names != expected:
            raise ValueError(
                f"{case.root}: material_feature_names order differs from case "
                f"{expected_case!r}; expected {expected}, found {names}"
            )


def _read_case_records(
    root: Path,
    geometry_key: str,
    material_key: str,
) -> list[dict[str, Any]]:
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and ((path / "nodes.npy").exists() or (path / "metadata.json").exists())
    )
    if not candidates:
        raise FileNotFoundError(f"No exported Valve cases found under {root}")

    records: list[dict[str, Any]] = []
    for case_dir in candidates:
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing case metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_path}: expected a JSON object")
        declared_case_id = metadata.get("case_id")
        if declared_case_id != case_dir.name:
            raise ValueError(
                f"{metadata_path}: case_id must equal its directory name "
                f"({case_dir.name!r}); found {declared_case_id!r}"
            )
        geometry = _normalize_metadata_id(
            _lookup_metadata(metadata, geometry_key), geometry_key
        )
        material = _normalize_metadata_id(
            _lookup_metadata(metadata, material_key), material_key
        )
        records.append(
            {
                "case_id": case_dir.name,
                "geometry_id": geometry,
                "material_id": material,
            }
        )
    return records


def _validate_fractions(validation_fraction: float, test_fraction: float) -> None:
    values = (float(validation_fraction), float(test_fraction))
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("validation_fraction and test_fraction must be finite and positive")
    if sum(values) >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than one")


def _validate_metadata_key(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(not part for part in value.split("."))
    ):
        raise ValueError(f"{name} must be a non-empty dotted metadata key")
    return value


def _normalize_ordered_names(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an explicit ordered list")
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(
                f"{name}[{index}] must be a non-empty string without outer whitespace"
            )
        names.append(item)
    if len(set(names)) != len(names):
        raise ValueError(f"{name} entries must be unique")
    return names


def _validate_component_requirement(
    metadata: Mapping[str, Any],
    requirements: Mapping[str, Any],
    key: str,
    canonical: tuple[str, ...],
    errors: list[str],
) -> None:
    requested = requirements.get(key)
    if requested is None:
        return
    try:
        expected = _normalize_ordered_names(requested, f"data.requirements.{key}")
        if tuple(expected) != canonical:
            raise ValueError(
                f"data.requirements.{key} must use canonical order {list(canonical)}"
            )
        declared = _normalize_ordered_names(
            _lookup_metadata(metadata, key), f"metadata {key}"
        )
        if declared != expected:
            raise ValueError(
                f"metadata {key} must equal {expected}; found {declared}"
            )
    except (KeyError, ValueError) as exc:
        errors.append(str(exc))


def _lookup_metadata(metadata: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = metadata
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"metadata key {dotted_key!r} is required")
        current = current[part]
    return current


def _normalize_metadata_id(value: Any, key: str) -> str | int | float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"metadata key {key!r} must be a string or finite number")
    if isinstance(value, str):
        if not value or value != value.strip():
            raise ValueError(f"metadata key {key!r} must be non-empty without outer whitespace")
        return value
    if not math.isfinite(float(value)):
        raise ValueError(f"metadata key {key!r} must be finite")
    return value


def _typed_metadata_token(value: str | int | float) -> str:
    value_type = "str" if isinstance(value, str) else "int" if isinstance(value, int) else "float"
    return json.dumps([value_type, value], ensure_ascii=False, separators=(",", ":"))


def _assignment_digest(seed: int, group: tuple[str, str]) -> str:
    message = json.dumps([int(seed), group[0], group[1]], separators=(",", ":"))
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _combination_digest(group: tuple[str, str]) -> str:
    message = json.dumps(group, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _held_out_group_counts(
    num_groups: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    targets = [num_groups * validation_fraction, num_groups * test_fraction]
    counts = [max(1, int(math.floor(target + 0.5))) for target in targets]
    while sum(counts) > num_groups - 1:
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError("Not enough combinations to keep train, val, and test non-empty")
        index = max(
            candidates,
            key=lambda item: (counts[item] - targets[item], counts[item], -item),
        )
        counts[index] -= 1
    return counts[0], counts[1]


def _assert_combination_disjoint(groups: Mapping[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for record in groups[split]:
            digest = str(record["combination_sha256"])
            previous = seen.setdefault(digest, split)
            if previous != split:
                raise AssertionError(
                    f"geometry/material combination leaked between {previous} and {split}"
                )


def _factor_overlap_audit(
    groups: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    train_geometry = {_typed_metadata_token(row["geometry_id"]) for row in groups["train"]}
    train_material = {_typed_metadata_token(row["material_id"]) for row in groups["train"]}
    audit: dict[str, dict[str, int]] = {}
    for split in ("val", "test"):
        geometry = {_typed_metadata_token(row["geometry_id"]) for row in groups[split]}
        material = {_typed_metadata_token(row["material_id"]) for row in groups[split]}
        audit[split] = {
            "geometry_seen_in_train": len(geometry & train_geometry),
            "geometry_unseen_in_train": len(geometry - train_geometry),
            "material_seen_in_train": len(material & train_material),
            "material_unseen_in_train": len(material - train_material),
        }
    return audit


def _array_is_finite(value: np.ndarray, chunk_size: int = 1_000_000) -> bool:
    array = np.asarray(value)
    flat = array.reshape(-1)
    for start in range(0, flat.size, chunk_size):
        if not bool(np.isfinite(flat[start : start + chunk_size]).all()):
            return False
    return True


def _masked_tensor_is_finite(stress: np.ndarray, mask: np.ndarray) -> bool:
    for frame in range(stress.shape[0]):
        valid = np.asarray(mask[frame])
        values = np.asarray(stress[frame])
        if not bool(np.isfinite(values[valid]).all()):
            return False
    return True
