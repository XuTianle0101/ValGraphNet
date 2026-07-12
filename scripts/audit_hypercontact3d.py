#!/usr/bin/env python3
"""Audit a converted HyperContact-3D dataset and write provenance-rich JSON.

The audit is deliberately independent from training.  It verifies that every
declared split member is present exactly once, applies the requirements from a
CHP configuration, and recomputes deformation determinants directly from the
converted reference geometry and displacement history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from valgraphnet.data.case import discover_case_dirs, load_case
from valgraphnet.data.valve_ood import validate_case_requirements


AUDIT_SCHEMA_VERSION = 2
REACTION_EQUILIBRIUM_TOLERANCE = 1.0e-3


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {label} must contain a JSON object")
    return value


def _normalized_splits(value: Mapping[str, Any]) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for raw_name, raw_ids in value.items():
        name = str(raw_name)
        if not name or not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"split {name!r} must contain a non-empty JSON list")
        ids = [str(case_id) for case_id in raw_ids]
        if any(not case_id for case_id in ids):
            raise ValueError(f"split {name!r} contains an empty case id")
        if len(ids) != len(set(ids)):
            raise ValueError(f"split {name!r} contains duplicate case ids")
        splits[name] = ids
    if not splits:
        raise ValueError("split file contains no splits")
    return splits


def _coverage_audit(
    root: Path,
    splits: Mapping[str, list[str]],
    conversion_manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    errors: list[str] = []
    owners: dict[str, str] = {}
    for split, case_ids in splits.items():
        for case_id in case_ids:
            if case_id in owners:
                errors.append(
                    f"case {case_id!r} occurs in both {owners[case_id]!r} and {split!r}"
                )
            else:
                owners[case_id] = split

    discovered = {path.name for path in discover_case_dirs(root)}
    declared = set(owners)
    missing_directories = sorted(declared - discovered)
    unassigned_directories = sorted(discovered - declared)
    if missing_directories:
        errors.append(f"split cases missing directories: {missing_directories[:10]}")
    if unassigned_directories:
        errors.append(f"case directories absent from splits: {unassigned_directories[:10]}")

    manifest_cases = conversion_manifest.get("cases")
    if not isinstance(manifest_cases, list):
        errors.append("conversion manifest cases must be a list")
        manifest_cases = []
    manifest_ids = [
        str(entry.get("case_id", ""))
        for entry in manifest_cases
        if isinstance(entry, Mapping)
    ]
    if len(manifest_ids) != len(manifest_cases) or any(not value for value in manifest_ids):
        errors.append("conversion manifest contains malformed case entries")
    if len(manifest_ids) != len(set(manifest_ids)):
        errors.append("conversion manifest contains duplicate case ids")
    missing_manifest = sorted(declared - set(manifest_ids))
    extra_manifest = sorted(set(manifest_ids) - declared)
    if missing_manifest:
        errors.append(f"split cases missing from conversion manifest: {missing_manifest[:10]}")
    if extra_manifest:
        errors.append(f"conversion manifest cases absent from splits: {extra_manifest[:10]}")
    if conversion_manifest.get("case_count") != len(declared):
        errors.append(
            "conversion manifest case_count does not equal unique split coverage "
            f"({conversion_manifest.get('case_count')!r} != {len(declared)})"
        )
    expected_split_counts = {name: len(ids) for name, ids in sorted(splits.items())}
    if conversion_manifest.get("split_counts") != expected_split_counts:
        errors.append("conversion manifest split_counts do not match splits.json")

    manifest_by_id = {
        str(entry["case_id"]): entry
        for entry in manifest_cases
        if isinstance(entry, Mapping) and entry.get("case_id")
    }
    coverage = {
        "unique_case_count": len(declared),
        "discovered_case_count": len(discovered),
        "manifest_case_count": len(manifest_ids),
        "split_counts": expected_split_counts,
        "missing_directories": missing_directories,
        "unassigned_directories": unassigned_directories,
        "missing_manifest_entries": missing_manifest,
        "extra_manifest_entries": extra_manifest,
        "exact": not errors,
    }
    return errors, coverage, owners


def _finite(name: str, value: np.ndarray, errors: list[str]) -> bool:
    if not bool(np.isfinite(np.asarray(value)).all()):
        errors.append(f"{name} contains non-finite values")
        return False
    return True


def _minimum_j(case: Any, chunk_size: int) -> tuple[float, float]:
    nodes = np.asarray(case.nodes, dtype=np.float64)
    cells = np.asarray(case.cells, dtype=np.int64)
    dm_inv = np.asarray(case.dm_inv, dtype=np.float64)
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, case.num_steps, chunk_size):
        stop = min(start + chunk_size, case.num_steps)
        current = nodes[None] + np.asarray(case.displacement[start:stop], dtype=np.float64)
        vertices = current[:, cells]
        ds = np.stack(
            (
                vertices[:, :, 1] - vertices[:, :, 0],
                vertices[:, :, 2] - vertices[:, :, 0],
                vertices[:, :, 3] - vertices[:, :, 0],
            ),
            axis=-1,
        )
        determinant = np.linalg.det(ds @ dm_inv[None])
        if not bool(np.isfinite(determinant).all()):
            return float("nan"), float("nan")
        minimum = min(minimum, float(determinant.min()))
        maximum = max(maximum, float(determinant.max()))
    return minimum, maximum


def _neo_hookean_internal_force(case: Any, frame: int) -> np.ndarray:
    """Reassemble the generator's Neo-Hookean force in float64."""

    position = np.asarray(case.nodes, dtype=np.float64) + np.asarray(
        case.displacement[int(frame)], dtype=np.float64
    )
    cells = np.asarray(case.cells, dtype=np.int64)
    vertices = position[cells]
    ds = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=-1,
    )
    deformation = ds @ np.asarray(case.dm_inv, dtype=np.float64)
    determinant = np.linalg.det(deformation)
    if not bool(np.isfinite(determinant).all()) or bool(np.any(determinant <= 0.0)):
        raise ValueError("Neo-Hookean reaction check requires positive finite J")
    right_cauchy_green = np.swapaxes(deformation, 1, 2) @ deformation
    first_invariant = np.trace(right_cauchy_green, axis1=1, axis2=2)
    cofactor = np.stack(
        (
            np.cross(deformation[:, :, 1], deformation[:, :, 2]),
            np.cross(deformation[:, :, 2], deformation[:, :, 0]),
            np.cross(deformation[:, :, 0], deformation[:, :, 1]),
        ),
        axis=-1,
    )
    derivative_i1_bar = (
        2.0 * determinant[:, None, None] ** (-2.0 / 3.0) * deformation
        - (2.0 / 3.0)
        * first_invariant[:, None, None]
        * determinant[:, None, None] ** (-5.0 / 3.0)
        * cofactor
    )
    material = np.asarray(case.material_features, dtype=np.float64)
    if material.shape[0] != case.num_cells or material.shape[1] < 1:
        raise ValueError("material_features must provide per-cell C10")
    c10 = material[:, 0]
    derived = case.metadata.get("derived_solver_parameters", {})
    d1 = float(derived.get("d1_pa_inverse", 0.0))
    if bool(np.any(c10 <= 0.0)) or not np.isfinite(d1) or d1 <= 0.0:
        raise ValueError("Neo-Hookean reaction check requires positive C10 and D1")
    first_piola = (
        c10[:, None, None] * derivative_i1_bar
        + (2.0 / d1)
        * (determinant - 1.0)[:, None, None]
        * cofactor
    )
    contribution = -np.einsum(
        "mij,mnj->mni",
        first_piola,
        np.asarray(case.shape_gradients, dtype=np.float64),
    )
    contribution *= np.asarray(
        case.reference_volume[:, 0], dtype=np.float64
    )[:, None, None]
    internal = np.zeros((case.num_nodes, 3), dtype=np.float64)
    for local_index in range(4):
        np.add.at(internal, cells[:, local_index], contribution[:, local_index])
    return internal


def _fixed_reaction_relative_rmse(
    case: Any, solver_force: np.ndarray, frame: int = -1
) -> float:
    fixed = np.asarray(case.fixed_mask, dtype=bool)
    if not bool(fixed.any()):
        raise ValueError("fixed reaction check requires fixed nodes")
    target = np.asarray(solver_force[int(frame), fixed], dtype=np.float64)
    reference = float(np.square(target).sum())
    if not np.isfinite(reference) or reference <= 1.0e-30:
        raise ValueError("fixed solver reaction is zero or non-finite")
    internal = _neo_hookean_internal_force(case, int(frame))[fixed]
    return float(np.sqrt(np.square(internal + target).sum() / reference))


def _case_audit(
    case_dir: Path,
    split: str,
    config: Mapping[str, Any],
    manifest_entry: Mapping[str, Any] | None,
    *,
    j_chunk_size: int,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        case = load_case(case_dir)
    except Exception as exc:  # keep auditing the remaining 167 cases
        return {
            "case_id": case_dir.name,
            "split": split,
            "status": "failed",
            "errors": [f"case load failed: {type(exc).__name__}: {exc}"],
        }

    if case.case_id != case_dir.name:
        errors.append(f"metadata case_id {case.case_id!r} differs from directory name")
    if case.metadata.get("split") != split:
        errors.append(
            f"metadata split {case.metadata.get('split')!r} differs from split file {split!r}"
        )
    if manifest_entry is None:
        errors.append("missing conversion manifest entry")
    else:
        for key in ("case_id", "split", "deck_sha256"):
            if manifest_entry.get(key) != case.metadata.get(key):
                errors.append(f"metadata {key} differs from conversion manifest")

    try:
        validate_case_requirements(case, config)
    except Exception as exc:
        errors.append(f"configuration requirements failed: {exc}")

    histories = {
        "U.npy": case.displacement,
        "V.npy": case.velocity,
        "A.npy": case.acceleration,
        "S.npy": case.stress,
    }
    finite = all(_finite(name, value, errors) for name, value in histories.items())
    cell_stress_finite = _finite("S_cell.npy", case.cell_stress, errors)
    ip_stress_finite = _finite(
        "S_integration_point.npy", case.integration_point_stress, errors
    )
    ip_mask = np.asarray(case.integration_point_mask, dtype=bool)
    full_ip_coverage = bool(
        ip_mask.shape == case.integration_point_stress.shape[:3] and ip_mask.all()
    )
    if not full_ip_coverage:
        errors.append("integration-point mask does not provide full frame/cell coverage")
    complete_finite_tensors = bool(
        case.cell_stress.shape == (case.num_steps, case.num_cells, 6)
        and cell_stress_finite
        and case.integration_point_stress.ndim == 4
        and case.integration_point_stress.shape[:2]
        == (case.num_steps, case.num_cells)
        and case.integration_point_stress.shape[2] > 0
        and case.integration_point_stress.shape[3] == 6
        and ip_stress_finite
        and full_ip_coverage
    )

    force_path = case.root / "solver_nodal_force.npy"
    if force_path.is_file():
        solver_force = np.load(force_path, allow_pickle=False, mmap_mode="r")
        if solver_force.shape != (case.num_steps, case.num_nodes, 3):
            errors.append(
                "solver_nodal_force.npy must have shape "
                f"({case.num_steps}, {case.num_nodes}, 3); found {solver_force.shape}"
            )
        force_finite = _finite("solver_nodal_force.npy", solver_force, errors)
    else:
        solver_force = np.zeros((0,), dtype=np.float32)
        force_finite = False
        errors.append("missing solver_nodal_force.npy")

    cell_stress_abs_max = (
        float(np.abs(np.asarray(case.cell_stress[1:])).max())
        if case.num_steps > 1 and case.cell_stress.size
        else 0.0
    )
    ip_after_reference = np.asarray(case.integration_point_stress[1:])[
        ip_mask[1:]
    ]
    ip_stress_abs_max = (
        float(np.abs(ip_after_reference).max())
        if ip_after_reference.size
        else 0.0
    )
    solver_force_abs_max = (
        float(np.abs(np.asarray(solver_force[1:])).max())
        if force_finite and solver_force.shape[0] > 1
        else 0.0
    )
    if cell_stress_abs_max <= 0.0:
        errors.append("cell stress is identically zero after the reference frame")
    if ip_stress_abs_max <= 0.0:
        errors.append("integration-point stress is identically zero after the reference frame")
    if solver_force_abs_max <= 0.0:
        errors.append("solver nodal force is identically zero after the reference frame")

    minimum_j, maximum_j = (
        _minimum_j(case, j_chunk_size) if finite else (float("nan"),) * 2
    )
    if not np.isfinite(minimum_j) or minimum_j <= 0.0:
        errors.append(f"non-positive or non-finite deformation determinant (min J={minimum_j})")

    fixed = np.asarray(case.fixed_mask, dtype=bool)
    prescribed = np.asarray(case.prescribed_mask, dtype=bool)
    if not bool(fixed.any()):
        errors.append("fixed_mask.npy selects no nodes")
    if not bool(prescribed.any()):
        errors.append("prescribed_mask.npy selects no nodes")
    if bool(np.any(fixed & prescribed)):
        errors.append("fixed and prescribed masks overlap")
    reaction_relative_rmse = float("nan")
    if force_finite and solver_force.shape == (case.num_steps, case.num_nodes, 3):
        try:
            reaction_relative_rmse = _fixed_reaction_relative_rmse(
                case, solver_force, frame=-1
            )
            if reaction_relative_rmse > REACTION_EQUILIBRIUM_TOLERANCE:
                errors.append(
                    "Neo-Hookean fixed-reaction equilibrium rRMSE exceeds "
                    f"{REACTION_EQUILIBRIUM_TOLERANCE}: {reaction_relative_rmse}"
                )
        except (ValueError, FloatingPointError) as exc:
            errors.append(f"fixed-reaction equilibrium check failed: {exc}")
    fixed_exact = bool(
        fixed.any()
        and all(
            np.array_equal(
                np.asarray(history)[:, fixed],
                np.zeros_like(np.asarray(history)[:, fixed]),
            )
            for history in (case.displacement, case.velocity, case.acceleration)
        )
    )
    if not fixed_exact:
        errors.append("fixed-node U/V/A histories are not bit-exact zero")

    prescribed_uniform = False
    prescribed_ramp = False
    ramp_abs_error = float("nan")
    if prescribed.any():
        prescribed_uniform = all(
            np.array_equal(
                np.asarray(history)[:, prescribed],
                np.broadcast_to(
                    np.asarray(history)[:, prescribed][:, :1],
                    np.asarray(history)[:, prescribed].shape,
                ),
            )
            for history in (case.displacement, case.velocity, case.acceleration)
        )
        if not prescribed_uniform:
            errors.append("prescribed rigid-body U/V/A are not bit-exact uniform per frame")
        derived = case.metadata.get("derived_solver_parameters", {})
        imposed = derived.get("imposed_indenter_displacement_m")
        duration = derived.get("step_duration")
        if (
            isinstance(imposed, (int, float))
            and isinstance(duration, (int, float))
            and duration > 0
        ):
            times = np.asarray(case.times, dtype=np.float64)
            expected_z = times / float(duration) * float(imposed)
            prescribed_u = np.asarray(case.displacement[:, prescribed], dtype=np.float64)
            ramp_abs_error = float(
                max(
                    np.abs(prescribed_u[..., :2]).max(initial=0.0),
                    np.abs(prescribed_u[..., 2] - expected_z[:, None]).max(initial=0.0),
                )
            )
            tolerance = max(
                16.0 * np.finfo(np.float32).eps * max(abs(float(imposed)), 1.0e-12),
                1.0e-12,
            )
            prescribed_ramp = ramp_abs_error <= tolerance
            if not prescribed_ramp:
                errors.append(
                    f"prescribed displacement differs from analytic ramp by {ramp_abs_error:.9g}"
                )
        else:
            errors.append("missing valid imposed displacement or step duration provenance")

    provenance = case.metadata.get("benchmark_provenance", {})
    return {
        "case_id": case.case_id,
        "split": split,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "num_frames": case.num_steps,
        "num_nodes": case.num_nodes,
        "num_cells": case.num_cells,
        "minimum_j": minimum_j,
        "maximum_j": maximum_j,
        "cell_stress_abs_max_pa": cell_stress_abs_max,
        "integration_point_stress_abs_max_pa": ip_stress_abs_max,
        "solver_force_abs_max_n": solver_force_abs_max,
        "neo_hookean_fixed_reaction_relative_rmse": reaction_relative_rmse,
        "complete_finite_cell_and_ip_tensors": complete_finite_tensors,
        "fixed_state_bit_exact": fixed_exact,
        "prescribed_state_spatially_bit_exact": prescribed_uniform,
        "prescribed_ramp_abs_error_m": ramp_abs_error,
        "prescribed_ramp_within_float32_tolerance": prescribed_ramp,
        "deck_sha256": case.metadata.get("deck_sha256"),
        "generator_version": provenance.get("generator_version"),
        "generator_config_sha256": provenance.get("config_sha256"),
        "stress_source": case.metadata.get("stress_source"),
        "time_semantics": case.metadata.get("time_semantics"),
    }


def audit_hypercontact3d_dataset(
    data_root: str | Path,
    config_path: str | Path,
    *,
    split_file: str | Path | None = None,
    conversion_manifest: str | Path | None = None,
    j_chunk_size: int = 8,
) -> dict[str, Any]:
    """Run all converted-dataset checks and return a JSON-serializable report."""

    root = Path(data_root).resolve()
    config_file = Path(config_path).resolve()
    if j_chunk_size < 1:
        raise ValueError("j_chunk_size must be positive")
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_file}: configuration must be a mapping")
    requirements = config.get("data", {}).get("requirements", {})

    configured_split = config.get("data", {}).get("split_file")
    split_path = Path(split_file or configured_split or root / "splits.json").resolve()
    manifest_path = Path(conversion_manifest or root / "conversion_manifest.json").resolve()
    split_value = _read_json_object(split_path, "split file")
    splits = _normalized_splits(split_value)
    manifest = _read_json_object(manifest_path, "conversion manifest")
    coverage_errors, coverage, owners = _coverage_audit(root, splits, manifest)
    manifest_by_id = {
        str(entry["case_id"]): entry
        for entry in manifest.get("cases", [])
        if isinstance(entry, Mapping) and entry.get("case_id")
    }

    cases = [
        _case_audit(
            root / case_id,
            owners[case_id],
            config,
            manifest_by_id.get(case_id),
            j_chunk_size=j_chunk_size,
        )
        for case_id in sorted(owners)
        if (root / case_id).is_dir()
    ]
    failed_cases = [row["case_id"] for row in cases if row["status"] != "passed"]
    numeric_cases = [row for row in cases if "minimum_j" in row]
    generator_versions = sorted(
        {str(row["generator_version"]) for row in numeric_cases if row.get("generator_version")}
    )
    generator_hashes = sorted(
        {
            str(row["generator_config_sha256"])
            for row in numeric_cases
            if row.get("generator_config_sha256")
        }
    )
    dataset_errors = list(coverage_errors)
    if len(generator_versions) != 1:
        dataset_errors.append(f"expected one generator version; found {generator_versions}")
    if len(generator_hashes) != 1:
        dataset_errors.append(f"expected one generator config hash; found {generator_hashes}")
    if any(row.get("num_frames") != 101 for row in cases):
        dataset_errors.append("every HyperContact-3D case must contain exactly 101 frames")

    minima = [float(row["minimum_j"]) for row in numeric_cases]
    maxima_stress = [float(row["cell_stress_abs_max_pa"]) for row in numeric_cases]
    maxima_force = [float(row["solver_force_abs_max_n"]) for row in numeric_cases]
    reaction_relative = [
        float(row.get("neo_hookean_fixed_reaction_relative_rmse", np.nan))
        for row in numeric_cases
    ]
    status = "passed" if not dataset_errors and not failed_cases else "failed"
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "HyperContact-3D converted dataset",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "dataset_root": str(root),
            "config": str(config_file),
            "config_sha256": _sha256(config_file),
            "requirements_sha256": _canonical_sha256(requirements),
            "split_file": str(split_path),
            "split_file_sha256": _sha256(split_path),
            "conversion_manifest": str(manifest_path),
            "conversion_manifest_sha256": _sha256(manifest_path),
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "generator_versions": generator_versions,
            "generator_config_sha256": generator_hashes,
        },
        "coverage": coverage,
        "checks": {
            "split_case_manifest_exact": not coverage_errors,
            "config_requirements_all_cases": all(
                not any("configuration requirements failed" in error for error in row["errors"])
                for row in cases
            ),
            "all_cases_101_frames": all(row.get("num_frames") == 101 for row in cases),
            "full_finite_cell_and_ip_stress": all(
                bool(row.get("complete_finite_cell_and_ip_tensors", False))
                for row in cases
            ),
            "stress_nonzero_all_cases": all(
                row.get("cell_stress_abs_max_pa", 0.0) > 0.0
                and row.get("integration_point_stress_abs_max_pa", 0.0) > 0.0
                for row in cases
            ),
            "solver_force_nonzero_all_cases": all(
                row.get("solver_force_abs_max_n", 0.0) > 0.0 for row in cases
            ),
            "neo_hookean_fixed_reaction_equilibrium_all_cases": all(
                np.isfinite(value)
                and value <= REACTION_EQUILIBRIUM_TOLERANCE
                for value in reaction_relative
            ),
            "positive_j_all_frames_cells": all(
                np.isfinite(row.get("minimum_j", np.nan)) and row.get("minimum_j", 0.0) > 0.0
                for row in cases
            ),
            "fixed_state_bit_exact_all_cases": all(
                bool(row.get("fixed_state_bit_exact", False)) for row in cases
            ),
            "prescribed_state_exact_all_cases": all(
                bool(row.get("prescribed_state_spatially_bit_exact", False))
                and bool(row.get("prescribed_ramp_within_float32_tolerance", False))
                for row in cases
            ),
        },
        "summary": {
            "case_count": len(cases),
            "failed_case_count": len(failed_cases),
            "failed_cases": failed_cases,
            "minimum_j": min(minima, default=None),
            "maximum_cell_stress_abs_pa": max(maxima_stress, default=None),
            "maximum_solver_force_abs_n": max(maxima_force, default=None),
            "maximum_neo_hookean_fixed_reaction_relative_rmse": max(
                reaction_relative, default=None
            ),
        },
        "dataset_errors": dataset_errors,
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--conversion-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--j-chunk-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured_root = config.get("data", {}).get("root") if isinstance(config, dict) else None
    data_root = args.data_root or configured_root
    if data_root is None:
        raise SystemExit("--data-root is required when data.root is absent from the config")
    report = audit_hypercontact3d_dataset(
        data_root,
        args.config,
        split_file=args.split_file,
        conversion_manifest=args.conversion_manifest,
        j_chunk_size=args.j_chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": report["summary"]["case_count"],
                "failed_case_count": report["summary"]["failed_case_count"],
                "minimum_j": report["summary"]["minimum_j"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
