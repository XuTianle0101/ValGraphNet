"""Materialize and evaluate a fail-closed, validation-only CHP ablation matrix.

The runner intentionally has no training or test mode.  It creates tagged
training configs for future runs and can evaluate already-trained, provenance-
matched checkpoints on the matrix's fixed validation subset only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import torch
import yaml

from valgraphnet.chp_train import validate_chp_checkpoint_semantics
from valgraphnet.config import deep_update, get_cfg, load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPOSITORY_ROOT / "configs/deforming_plate_chp_ablation.validation.yaml"
RUNNABLE = "runnable"
BLOCKED = "blocked"
DIAGNOSTIC_NAME = "cell_memory_diagnostic"


class AblationProtocolError(ValueError):
    """Raised when an ablation label would overstate the implemented experiment."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(cfg: Mapping[str, Any]) -> str:
    value = deepcopy(dict(cfg))
    tag = value.get("ablation")
    if isinstance(tag, dict):
        tag.pop("config_fingerprint", None)
    return _sha256_bytes(_json_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationProtocolError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AblationProtocolError(f"JSON artifact must contain an object: {path}")
    return value


def _resolve_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _forbidden_split(value: Any, forbidden: Iterable[str]) -> bool:
    split = str(value).strip().lower()
    return any(token.strip().lower() in split for token in forbidden)


def _leaf_paths(value: Mapping[str, Any], prefix: str = "") -> set[str]:
    leaves: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            leaves.update(_leaf_paths(item, path))
        else:
            leaves.add(path)
    return leaves


def load_ablation_matrix(path: str | Path) -> dict[str, Any]:
    matrix_path = Path(path)
    try:
        value = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise AblationProtocolError(f"cannot read ablation matrix {matrix_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AblationProtocolError("ablation matrix must be a YAML mapping")
    return value


def _validate_variant_override(variant_id: str, spec: Mapping[str, Any]) -> None:
    status = str(spec.get("status", ""))
    axis = str(spec.get("axis", ""))
    overrides = spec.get("overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise AblationProtocolError(f"{variant_id}.overrides must be a mapping")
    if status == BLOCKED:
        if overrides:
            raise AblationProtocolError(
                f"blocked variant {variant_id!r} cannot carry executable overrides"
            )
        if not spec.get("missing_capability") or not spec.get("blocked_reason"):
            raise AblationProtocolError(
                f"blocked variant {variant_id!r} must name its missing capability and reason"
            )
        return
    if status != RUNNABLE:
        raise AblationProtocolError(
            f"variant {variant_id!r} has unsupported status {status!r}"
        )

    leaves = _leaf_paths(overrides)
    allowed = {
        "reference": {"training.output_dir"},
        "rollout_training": {
            "training.epochs",
            "training.curriculum",
            "training.checkpoint_min_horizon",
            "training.output_dir",
        },
        "work_energy": {"loss.work_energy", "training.output_dir"},
        "hierarchy": {
            "model.use_topology_hierarchy",
            "training.output_dir",
        },
    }.get(axis)
    if allowed is None:
        raise AblationProtocolError(
            f"runnable variant {variant_id!r} uses an unaudited axis {axis!r}"
        )
    if not leaves <= allowed:
        raise AblationProtocolError(
            f"variant {variant_id!r} changes unaudited keys: {sorted(leaves - allowed)}"
        )
    if axis == "rollout_training":
        curriculum = get_cfg(dict(overrides), "training.curriculum")
        if not isinstance(curriculum, list) or not curriculum:
            raise AblationProtocolError("one-step ablation requires an explicit curriculum")
        if any(int(stage.get("horizon", 0)) != 1 for stage in curriculum):
            raise AblationProtocolError("one-step ablation may contain only K=1 stages")
        stage_epochs = sum(int(stage.get("epochs", 0)) for stage in curriculum)
        if stage_epochs != int(get_cfg(dict(overrides), "training.epochs", -1)):
            raise AblationProtocolError(
                "one-step curriculum epochs must equal training.epochs"
            )
        if int(get_cfg(dict(overrides), "training.checkpoint_min_horizon", -1)) != 1:
            raise AblationProtocolError(
                "one-step ablation must permit validation checkpointing at K=1"
            )
    if axis == "work_energy" and float(
        get_cfg(dict(overrides), "loss.work_energy", float("nan"))
    ) != 0.0:
        raise AblationProtocolError("no-work ablation must set loss.work_energy=0")
    if axis == "hierarchy" and get_cfg(
        dict(overrides), "model.use_topology_hierarchy", None
    ) is not False:
        raise AblationProtocolError(
            "flat hierarchy ablation must set model.use_topology_hierarchy=false"
        )


def validate_ablation_matrix(
    matrix: Mapping[str, Any],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    verify_dataset: bool = False,
) -> dict[str, Any]:
    """Validate the matrix and return its pinned cell-memory decision."""

    root = Path(repository_root)
    if int(matrix.get("schema_version", 0)) != 1:
        raise AblationProtocolError("unsupported ablation matrix schema")
    protocol = matrix.get("protocol")
    variants = matrix.get("variants")
    comparisons = matrix.get("comparisons")
    seeds = matrix.get("seeds")
    if not isinstance(protocol, Mapping):
        raise AblationProtocolError("matrix.protocol must be a mapping")
    if not isinstance(variants, Mapping) or not variants:
        raise AblationProtocolError("matrix.variants must be a non-empty mapping")
    if not isinstance(comparisons, Mapping) or not comparisons:
        raise AblationProtocolError("matrix.comparisons must be a non-empty mapping")
    if not isinstance(seeds, list) or not seeds or len(set(map(int, seeds))) != len(seeds):
        raise AblationProtocolError("matrix.seeds must contain unique integer seeds")
    if not bool(protocol.get("development_only", False)):
        raise AblationProtocolError("ablation protocol must be development-only")
    if not all(
        bool(protocol.get(key, False))
        for key in ("require_cuda", "require_bf16", "require_scientific_gates")
    ):
        raise AblationProtocolError("CUDA, BF16, and scientific gates are mandatory")
    if int(protocol.get("cases", 0)) != 20 or int(protocol.get("steps", 0)) != 399:
        raise AblationProtocolError("deforming_plate ablations require val20 x 399 steps")
    if str(protocol.get("case_selection")) != "even":
        raise AblationProtocolError("validation case selection must be even")
    forbidden = protocol.get("forbidden_split_tokens", ["test"])
    if not isinstance(forbidden, list) or not forbidden:
        raise AblationProtocolError("forbidden_split_tokens must be non-empty")
    evaluation_split = str(protocol.get("evaluation_split", ""))
    if not evaluation_split or _forbidden_split(evaluation_split, forbidden):
        raise AblationProtocolError(
            f"unsafe ablation evaluation split: {evaluation_split!r}"
        )

    for variant_id, spec in variants.items():
        if not isinstance(spec, Mapping):
            raise AblationProtocolError(f"variant {variant_id!r} must be a mapping")
        _validate_variant_override(str(variant_id), spec)
    for comparison_id, spec in comparisons.items():
        if not isinstance(spec, Mapping):
            raise AblationProtocolError(f"comparison {comparison_id!r} must be a mapping")
        pair = spec.get("variants")
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]:
            raise AblationProtocolError(
                f"comparison {comparison_id!r} must contain two distinct variants"
            )
        missing = [variant for variant in pair if variant not in variants]
        if missing:
            raise AblationProtocolError(
                f"comparison {comparison_id!r} references missing variants: {missing}"
            )

    base_path = _resolve_path(str(matrix.get("base_config", "")), root)
    if not base_path.is_file():
        raise AblationProtocolError(f"base config does not exist: {base_path}")
    _validate_resolved_config(load_config(base_path), protocol)

    diagnostic_spec = matrix.get(DIAGNOSTIC_NAME)
    if not isinstance(diagnostic_spec, Mapping):
        raise AblationProtocolError("cell_memory_diagnostic must be configured")
    diagnostic_path = _resolve_path(str(diagnostic_spec.get("result_file", "")), root)
    if not diagnostic_path.is_file():
        raise AblationProtocolError(
            f"cell-memory diagnostic is required before ablation planning: {diagnostic_path}"
        )
    diagnostic = _read_json(diagnostic_path)
    ratio = float(diagnostic.get("conditional_to_global_variance", float("nan")))
    threshold = float(diagnostic_spec.get("threshold", float("nan")))
    recorded_threshold = float(diagnostic.get("trigger_threshold", float("nan")))
    if not math.isfinite(ratio) or not math.isfinite(threshold):
        raise AblationProtocolError("cell-memory diagnostic must contain finite values")
    if threshold != recorded_threshold:
        raise AblationProtocolError("matrix and diagnostic cell-memory thresholds differ")
    expected_trigger = ratio > threshold
    if bool(diagnostic.get("enable_cell_memory")) != expected_trigger:
        raise AblationProtocolError("cell-memory diagnostic decision is inconsistent")
    diagnostic_split = str(diagnostic.get("split", ""))
    if diagnostic_split != str(diagnostic_spec.get("split", "")):
        raise AblationProtocolError("cell-memory diagnostic split provenance differs")
    if _forbidden_split(diagnostic_split, forbidden):
        raise AblationProtocolError("cell-memory diagnosis may not read a test split")

    split_file = _resolve_path(str(diagnostic.get("split_file", "")), root)
    expected_hash = str(diagnostic.get("split_file_sha256", "")).lower()
    if verify_dataset and not split_file.is_file():
        raise AblationProtocolError(f"diagnostic split file is missing: {split_file}")
    if split_file.is_file() and expected_hash:
        actual_hash = _sha256_file(split_file)
        if actual_hash != expected_hash:
            raise AblationProtocolError(
                "cell-memory diagnostic split hash does not match the active dataset"
            )
    return diagnostic


def _validate_resolved_config(
    cfg: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    forbidden = protocol.get("forbidden_split_tokens", ["test"])
    split = str(protocol.get("evaluation_split", ""))
    if _forbidden_split(split, forbidden):
        raise AblationProtocolError(f"test-like split is forbidden: {split!r}")
    if str(get_cfg(dict(cfg), "data.val_split", "")) != split:
        raise AblationProtocolError("base data.val_split differs from the protocol")
    if str(get_cfg(dict(cfg), "validation.native_reference_split", "")) != split:
        raise AblationProtocolError("native checkpoint reference must use validation")
    if str(get_cfg(dict(cfg), "validation.native_reference_case_selection", "")) != str(
        protocol.get("case_selection")
    ):
        raise AblationProtocolError("native reference case selection differs")
    if int(get_cfg(dict(cfg), "validation.cases", 0)) != int(protocol.get("cases", 0)):
        raise AblationProtocolError("validation case count differs from the matrix")
    if int(get_cfg(dict(cfg), "validation.steps", 0)) != int(protocol.get("steps", 0)):
        raise AblationProtocolError("validation frame count differs from the matrix")
    if str(get_cfg(dict(cfg), "training.device", "")).lower() != "cuda":
        raise AblationProtocolError("CHP ablation training must use CUDA")
    if not bool(get_cfg(dict(cfg), "training.amp", False)):
        raise AblationProtocolError("CHP ablation training must enable AMP")
    if str(get_cfg(dict(cfg), "training.amp_dtype", "")).lower() not in {
        "bf16",
        "bfloat16",
    }:
        raise AblationProtocolError("CHP ablation neural blocks must use BF16")
    if not bool(get_cfg(dict(cfg), "validation.enforce_teacher_stress_gate", False)):
        raise AblationProtocolError("teacher stress gate must remain enabled")
    if not bool(get_cfg(dict(cfg), "validation.enforce_rollout_pilot_gate", False)):
        raise AblationProtocolError("rollout pilot gate must remain enabled")


def _comparison_statuses(
    matrix: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    variants = matrix["variants"]
    triggered = bool(diagnostic["enable_cell_memory"])
    result: dict[str, dict[str, Any]] = {}
    for comparison_id, spec in matrix["comparisons"].items():
        conditional = spec.get("conditional_on") == DIAGNOSTIC_NAME
        if conditional and not triggered:
            status = "not_triggered"
            blockers: list[str] = []
        else:
            blockers = [
                variant
                for variant in spec["variants"]
                if variants[variant]["status"] != RUNNABLE
            ]
            status = RUNNABLE if not blockers else BLOCKED
        result[str(comparison_id)] = {
            "status": status,
            "variants": list(spec["variants"]),
            "required": bool(spec.get("required", False)),
            "blockers": blockers,
        }
    return result


def _resolved_variant_config(
    base: Mapping[str, Any],
    matrix: Mapping[str, Any],
    matrix_sha256: str,
    variant_id: str,
    seed: int,
) -> dict[str, Any]:
    spec = matrix["variants"][variant_id]
    cfg = deep_update(dict(base), deepcopy(spec.get("overrides", {})))
    cfg["seed"] = int(seed)
    protocol = matrix["protocol"]
    cfg["evaluation"] = deep_update(
        cfg.get("evaluation", {}),
        {
            "steps": int(protocol["steps"]),
            "max_cases": int(protocol["cases"]),
            "case_selection": str(protocol["case_selection"]),
        },
    )
    tag = {
        "schema_version": 1,
        "matrix": str(matrix["name"]),
        "matrix_sha256": matrix_sha256,
        "variant_id": str(variant_id),
        "axis": str(spec["axis"]),
        "seed": int(seed),
        "development_only": True,
        "evaluation_split": str(protocol["evaluation_split"]),
    }
    cfg["ablation"] = tag
    cfg["ablation"]["config_fingerprint"] = _config_fingerprint(cfg)
    _validate_resolved_config(cfg, protocol)
    return cfg


def materialize_ablation_plan(
    matrix_path: str | Path,
    resolved_dir: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Write tagged runnable configs and a machine-readable protocol manifest."""

    root = Path(repository_root)
    matrix_file = Path(matrix_path)
    matrix = load_ablation_matrix(matrix_file)
    diagnostic = validate_ablation_matrix(matrix, repository_root=root)
    statuses = _comparison_statuses(matrix, diagnostic)
    incomplete = [
        comparison_id
        for comparison_id, status in statuses.items()
        if status["required"] and status["status"] == BLOCKED
    ]
    if require_complete and incomplete:
        raise AblationProtocolError(
            "required CHP ablations are not implemented: " + ", ".join(incomplete)
        )

    matrix_sha256 = _sha256_file(matrix_file)
    base_path = _resolve_path(str(matrix["base_config"]), root)
    base = load_config(base_path)
    output = Path(resolved_dir)
    output.mkdir(parents=True, exist_ok=True)
    runnable: dict[str, Any] = {}
    for variant_id, spec in matrix["variants"].items():
        if spec["status"] != RUNNABLE:
            continue
        for seed_value in matrix["seeds"]:
            seed = int(seed_value)
            cfg = _resolved_variant_config(
                base, matrix, matrix_sha256, str(variant_id), seed
            )
            filename = f"{variant_id}.seed{seed}.yaml"
            config_path = output / filename
            config_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
            )
            key = f"{variant_id}:seed{seed}"
            runnable[key] = {
                "variant_id": str(variant_id),
                "seed": seed,
                "axis": str(spec["axis"]),
                "config": str(config_path.resolve()),
                "config_fingerprint": cfg["ablation"]["config_fingerprint"],
                "training_command": [
                    sys.executable,
                    str((root / "scripts/train_chp.py").resolve()),
                    "--config",
                    str(config_path.resolve()),
                ],
            }

    manifest = {
        "schema_version": 1,
        "matrix": str(matrix["name"]),
        "matrix_file": str(matrix_file.resolve()),
        "matrix_sha256": matrix_sha256,
        "development_only": True,
        "evaluation_split": str(matrix["protocol"]["evaluation_split"]),
        "case_selection": str(matrix["protocol"]["case_selection"]),
        "cases": int(matrix["protocol"]["cases"]),
        "steps": int(matrix["protocol"]["steps"]),
        "runner_capabilities": ["materialize", "evaluate_validation"],
        "training_execution_supported": False,
        "test_execution_supported": False,
        "cell_memory_diagnostic": {
            "conditional_to_global_variance": float(
                diagnostic["conditional_to_global_variance"]
            ),
            "trigger_threshold": float(diagnostic["trigger_threshold"]),
            "enable_cell_memory": bool(diagnostic["enable_cell_memory"]),
        },
        "comparisons": statuses,
        "incomplete_required_comparisons": incomplete,
        "runnable": runnable,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _checkpoint_for(
    mapping: Mapping[str, Any], variant_id: str, seed: int
) -> Path:
    if int(mapping.get("schema_version", 0)) != 1:
        raise AblationProtocolError("unsupported checkpoint-map schema")
    checkpoints = mapping.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise AblationProtocolError("checkpoint map must contain a checkpoints object")
    variant = checkpoints.get(variant_id)
    if not isinstance(variant, Mapping) or str(seed) not in variant:
        raise AblationProtocolError(
            f"checkpoint map is missing {variant_id}:seed{seed}"
        )
    return Path(str(variant[str(seed)]))


def _validate_checkpoint_tag(checkpoint_path: Path, expected: Mapping[str, Any]) -> None:
    if not checkpoint_path.is_file():
        raise AblationProtocolError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise AblationProtocolError(f"checkpoint is not a mapping: {checkpoint_path}")
    validate_chp_checkpoint_semantics(
        checkpoint, source=checkpoint_path, require_scientific_gate=True
    )
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, Mapping):
        raise AblationProtocolError("checkpoint does not preserve its training config")
    tag = checkpoint_cfg.get("ablation")
    if not isinstance(tag, Mapping):
        raise AblationProtocolError(
            "checkpoint lacks an ablation provenance tag; it cannot be relabeled"
        )
    expected_tag = load_config(expected["config"])["ablation"]
    if tag.get("config_fingerprint") != _config_fingerprint(checkpoint_cfg):
        raise AblationProtocolError(
            "checkpoint config fingerprint is internally inconsistent"
        )
    for key in (
        "matrix_sha256",
        "variant_id",
        "seed",
        "config_fingerprint",
        "evaluation_split",
    ):
        if tag.get(key) != expected_tag.get(key):
            raise AblationProtocolError(
                f"checkpoint ablation provenance mismatch for {key}: {checkpoint_path}"
            )


def evaluate_validation_only(
    manifest: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    output_dir: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    selected_variants: Iterable[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run existing, tagged checkpoints only on the pinned validation split."""

    if not bool(manifest.get("development_only")):
        raise AblationProtocolError("refusing a non-development ablation manifest")
    split = str(manifest.get("evaluation_split", ""))
    if "test" in split.lower():
        raise AblationProtocolError(f"test evaluation is forbidden: {split!r}")
    chosen = set(selected_variants or ())
    known_variants = {
        value["variant_id"] for value in manifest.get("runnable", {}).values()
    }
    unknown = chosen - known_variants
    if unknown:
        raise AblationProtocolError(
            "requested variants are blocked or unknown: " + ", ".join(sorted(unknown))
        )
    root = Path(repository_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for key, expected in manifest.get("runnable", {}).items():
        variant_id = str(expected["variant_id"])
        if chosen and variant_id not in chosen:
            continue
        seed = int(expected["seed"])
        checkpoint = _checkpoint_for(checkpoints, variant_id, seed)
        command = [
            sys.executable,
            str((root / "scripts/evaluate_chp.py").resolve()),
            "--config",
            str(expected["config"]),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--out",
            str((output / variant_id / f"seed{seed}").resolve()),
            "--split",
            split,
            "--max-cases",
            str(int(manifest["cases"])),
        ]
        split_arguments = [
            command[index + 1]
            for index, part in enumerate(command[:-1])
            if part == "--split"
        ]
        if split_arguments != [split] or any(
            "test" in value.lower() for value in split_arguments
        ):
            raise AblationProtocolError("generated command contains a forbidden test split")
        record = {
            "key": key,
            "variant_id": variant_id,
            "seed": seed,
            "split": split,
            "command": command,
            "status": "planned" if dry_run else "pending",
        }
        if not dry_run:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for CHP ablation inference")
            _validate_checkpoint_tag(checkpoint, expected)
            completed = subprocess.run(command, cwd=root, check=False)
            if completed.returncode != 0:
                record["status"] = "failed"
                record["returncode"] = int(completed.returncode)
                records.append(record)
                raise RuntimeError(
                    f"validation ablation failed for {key} with exit {completed.returncode}"
                )
            record["status"] = "completed"
            record["returncode"] = 0
        records.append(record)
    result = {
        "schema_version": 1,
        "development_only": True,
        "evaluation_split": split,
        "dry_run": bool(dry_run),
        "records": records,
    }
    (output / "evaluation_plan.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize or evaluate the validation-only CHP-GNS ablation matrix. "
            "Training and test execution are intentionally unsupported."
        )
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--resolved-dir",
        type=Path,
        default=Path("outputs/deforming_plate_chp_ablation/protocol"),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--require-complete", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoints", type=Path, required=True)
    evaluate.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/deforming_plate_chp_ablation/validation"),
    )
    evaluate.add_argument("--variant", action="append", default=[])
    evaluate.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = materialize_ablation_plan(
        args.matrix,
        args.resolved_dir,
        require_complete=bool(getattr(args, "require_complete", False)),
    )
    if args.action == "materialize":
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return
    checkpoints = _read_json(args.checkpoints)
    result = evaluate_validation_only(
        manifest,
        checkpoints,
        args.out,
        selected_variants=args.variant,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
