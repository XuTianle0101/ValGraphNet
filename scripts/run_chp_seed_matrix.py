"""Run the frozen CHP-GNS three-seed validation matrix on one GPU.

This orchestrator deliberately does not import or call the rollout exporter: the
four metrics are read from each scientifically eligible ``best.pt`` checkpoint,
where they were produced on the fixed validation subset during training.  The
test split is therefore never opened by this script.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping

import torch
import yaml

from valgraphnet.chp_model import CHPGNS
from valgraphnet.chp_train import validate_chp_checkpoint_semantics
from valgraphnet.config import get_cfg, load_config
from valgraphnet.physical_evaluation import PRIMARY_METRICS


SEEDS = (42, 43, 44)
MATRIX_SCHEMA_VERSION = 1
FAILURE_MARKERS = (
    "teacher_stress_gate_failure.json",
    "rollout_pilot_gate_failure.json",
)


class SeedMatrixError(RuntimeError):
    """Raised when a matrix prerequisite or immutable identity check fails."""


Trainer = Callable[[Path, int, Path], None]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_experiment_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the two seed-specific fields from a complete config."""

    normalized = deepcopy(dict(cfg))
    normalized["seed"] = "<MATRIX_SEED>"
    training = normalized.get("training")
    if not isinstance(training, dict):
        raise SeedMatrixError("CHP matrix config requires a training mapping")
    training["output_dir"] = "<MATRIX_OUTPUT_DIR>"
    return normalized


def experiment_config_hash(cfg: Mapping[str, Any]) -> str:
    """Hash the full frozen experiment, ignoring only seed and output path."""

    return _sha256_json(normalized_experiment_config(cfg))


def exact_config_hash(cfg: Mapping[str, Any]) -> str:
    return _sha256_json(dict(cfg))


def checkpoint_structure_hash(checkpoint: Mapping[str, Any]) -> str:
    """Hash model tensor structure plus physical-dynamics compatibility tags."""

    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise SeedMatrixError("checkpoint is missing a non-empty model state")
    tensors = []
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise SeedMatrixError(f"model state {name!r} is not a tensor")
        tensors.append(
            {
                "name": str(name),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
    identity = {
        "architecture": checkpoint.get("architecture"),
        "schema_version": checkpoint.get("schema_version"),
        "dynamics_schema_version": checkpoint.get("dynamics_schema_version"),
        "residual_parameterization": checkpoint.get("residual_parameterization"),
        "residual_gate": checkpoint.get("residual_gate"),
        "problem_type": checkpoint.get("problem_type", "dynamic"),
        "time_semantics": checkpoint.get("time_semantics", "dynamic"),
        "model_state": tensors,
    }
    return _sha256_json(identity)


def _seed_output_dir(base_output: str | Path, seed: int) -> Path:
    base = Path(base_output)
    replaced, count = re.subn(
        r"seed42(?=$|[^0-9])",
        f"seed{int(seed)}",
        base.name,
        count=1,
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise SeedMatrixError(
            "training.output_dir must contain a distinct 'seed42' token so "
            "matrix runs cannot share a directory"
        )
    return base.with_name(replaced)


def derive_seed_config(base_cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    if int(seed) not in SEEDS:
        raise SeedMatrixError(f"unsupported matrix seed: {seed}")
    cfg = deepcopy(dict(base_cfg))
    cfg["seed"] = int(seed)
    training = cfg.get("training")
    if not isinstance(training, dict):
        raise SeedMatrixError("CHP matrix config requires a training mapping")
    output = training.get("output_dir")
    if output is None:
        raise SeedMatrixError("training.output_dir is required")
    training["output_dir"] = str(_seed_output_dir(output, seed))
    return cfg


def validate_base_protocol(cfg: Mapping[str, Any]) -> None:
    """Require the frozen GPU/val20 protocol and forbid test-as-validation."""

    if int(cfg.get("seed", -1)) != 42:
        raise SeedMatrixError("the immutable base config must use pilot seed 42")
    if str(get_cfg(dict(cfg), "training.device", "")).lower() != "cuda":
        raise SeedMatrixError("CHP seed-matrix training must use CUDA")
    if not bool(get_cfg(dict(cfg), "training.amp", False)):
        raise SeedMatrixError("CHP seed-matrix training must enable mixed precision")
    if str(get_cfg(dict(cfg), "training.amp_dtype", "")).lower() not in {
        "bf16",
        "bfloat16",
    }:
        raise SeedMatrixError("CHP seed-matrix AMP dtype must be BF16")
    if int(get_cfg(dict(cfg), "validation.cases", -1)) != 20:
        raise SeedMatrixError("seed-matrix aggregation requires exactly val20")
    if int(get_cfg(dict(cfg), "validation.steps", -1)) != 399:
        raise SeedMatrixError("seed-matrix aggregation requires all 399 transitions")
    if str(
        get_cfg(dict(cfg), "validation.native_reference_case_selection", "")
    ).lower() != "even":
        raise SeedMatrixError("val20 must use deterministic even case selection")
    val_split = str(get_cfg(dict(cfg), "data.val_split", "val"))
    test_split = str(get_cfg(dict(cfg), "data.test_split", "test"))
    if val_split != "val" or val_split == test_split:
        raise SeedMatrixError(
            "matrix development metrics must come from the val split, never test"
        )
    if not bool(get_cfg(dict(cfg), "validation.enforce_teacher_stress_gate", False)):
        raise SeedMatrixError("teacher-stress scientific gate must be enforced")
    if not bool(get_cfg(dict(cfg), "validation.enforce_rollout_pilot_gate", False)):
        raise SeedMatrixError("rollout pilot scientific gate must be enforced")
    # Also proves all derived paths are distinct before a subprocess can start.
    outputs = {
        str(_seed_output_dir(get_cfg(dict(cfg), "training.output_dir"), seed).resolve())
        for seed in SEEDS
    }
    if len(outputs) != len(SEEDS):
        raise SeedMatrixError("each matrix seed must have an independent output_dir")


def _torch_load_cpu(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise SeedMatrixError(f"checkpoint payload is not a mapping: {path}")
    return value


def _assert_no_failure_marker(output_dir: Path) -> None:
    found = [name for name in FAILURE_MARKERS if (output_dir / name).is_file()]
    if found:
        raise SeedMatrixError(
            f"scientific gate failure marker blocks {output_dir}: {', '.join(found)}"
        )


def inspect_seed_checkpoint(
    checkpoint_path: Path,
    expected_cfg: Mapping[str, Any],
    *,
    expected_structure_hash: str | None = None,
) -> dict[str, Any]:
    """Validate a seed checkpoint without running inference or opening test data."""

    if not checkpoint_path.is_file():
        raise SeedMatrixError(f"missing seed checkpoint: {checkpoint_path}")
    _assert_no_failure_marker(checkpoint_path.parent)
    checkpoint = _torch_load_cpu(checkpoint_path)
    try:
        validate_chp_checkpoint_semantics(
            dict(checkpoint),
            source=checkpoint_path,
            require_scientific_gate=True,
        )
    except (ValueError, RuntimeError) as error:
        raise SeedMatrixError(str(error)) from error
    if str(checkpoint.get("scientific_gate_status")) != "passed":
        raise SeedMatrixError(
            f"pilot/scientific gates are not passed: {checkpoint_path}"
        )
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, Mapping):
        raise SeedMatrixError("checkpoint is missing its complete training config")
    expected_logical_hash = experiment_config_hash(expected_cfg)
    actual_logical_hash = experiment_config_hash(checkpoint_cfg)
    if actual_logical_hash != expected_logical_hash:
        raise SeedMatrixError(
            "checkpoint config hash differs from the frozen base experiment: "
            f"{checkpoint_path}"
        )
    expected_seed = int(expected_cfg["seed"])
    if int(checkpoint_cfg.get("seed", -1)) != expected_seed:
        raise SeedMatrixError(
            f"checkpoint seed mismatch: expected {expected_seed}, "
            f"got {checkpoint_cfg.get('seed')!r}"
        )
    expected_output = Path(str(get_cfg(dict(expected_cfg), "training.output_dir")))
    actual_output = Path(str(get_cfg(dict(checkpoint_cfg), "training.output_dir", "")))
    if actual_output.resolve() != expected_output.resolve():
        raise SeedMatrixError("checkpoint output_dir does not match its isolated seed run")

    metrics = checkpoint.get("rollout_metrics")
    if not isinstance(metrics, Mapping):
        raise SeedMatrixError("checkpoint has no validation rollout metrics")
    selected_metrics: dict[str, float] = {}
    for name in PRIMARY_METRICS:
        value = float(metrics.get(name, float("nan")))
        if not math.isfinite(value) or value < 0.0:
            raise SeedMatrixError(f"invalid val20 metric {name!r}: {value!r}")
        selected_metrics[name] = value
    if float(metrics.get("diverged_cases", float("inf"))) != 0.0:
        raise SeedMatrixError("scientifically eligible val20 checkpoint must not diverge")
    teacher_stress = float(
        metrics.get("teacher_stress_relative_rmse", float("nan"))
    )
    teacher_threshold = float(
        get_cfg(dict(expected_cfg), "validation.teacher_stress_threshold", 0.50)
    )
    if not math.isfinite(teacher_stress) or teacher_stress >= teacher_threshold:
        raise SeedMatrixError(
            "checkpoint gate status contradicts its teacher-stress validation metric"
        )
    moving_threshold = float(
        get_cfg(
            dict(expected_cfg),
            "validation.pilot_moving_relative_rmse_threshold",
            0.80,
        )
    )
    stress_threshold = float(
        get_cfg(
            dict(expected_cfg),
            "validation.pilot_stress_relative_rmse_threshold",
            0.65,
        )
    )
    if (
        selected_metrics["moving_displacement_relative_rmse"] >= moving_threshold
        or selected_metrics["stress_relative_rmse"] >= stress_threshold
    ):
        raise SeedMatrixError(
            "checkpoint gate status contradicts its rollout-pilot validation metrics"
        )

    structure_hash = checkpoint_structure_hash(checkpoint)
    if expected_structure_hash is not None and structure_hash != expected_structure_hash:
        raise SeedMatrixError(
            "checkpoint model structure hash differs across seeds: "
            f"{checkpoint_path}"
        )
    return {
        "seed": expected_seed,
        "output_dir": str(expected_output.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "gate_status": "passed",
        "epoch": int(checkpoint.get("epoch", -1)),
        "logical_config_hash": actual_logical_hash,
        "exact_config_hash": exact_config_hash(checkpoint_cfg),
        "structure_hash": structure_hash,
        "val20_metrics": selected_metrics,
    }


def aggregate_seed_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    gate_status = {str(record["seed"]): record["gate_status"] for record in records}
    aggregates: dict[str, dict[str, float]] = {}
    for name in PRIMARY_METRICS:
        values = [float(record["val20_metrics"][name]) for record in records]
        aggregates[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return {
        "seed_count": len(records),
        "gate_status": gate_status,
        "all_scientific_gates_passed": (
            len(records) == len(SEEDS)
            and all(value == "passed" for value in gate_status.values())
        ),
        "metrics": aggregates,
    }


def _write_yaml_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)


@contextmanager
def _exclusive_gpu_lock(path: Path) -> Iterator[None]:
    """Prevent two seed-matrix orchestrators from sharing the selected GPU."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise SeedMatrixError(
            f"another sequential GPU matrix owns lock {path}"
        ) from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def _subprocess_trainer(gpu: str) -> Trainer:
    def train(config_path: Path, seed: int, output_dir: Path) -> None:
        del seed, output_dir
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["VALGRAPHNET_SEED_MATRIX_SERIAL"] = "1"
        subprocess.run(
            [sys.executable, "-m", "scripts.train_chp", "--config", str(config_path)],
            check=True,
            env=environment,
        )

    return train


def _default_state_dir(base_cfg: Mapping[str, Any]) -> Path:
    seed42 = _seed_output_dir(get_cfg(dict(base_cfg), "training.output_dir"), 42)
    name = re.sub(r"seed42(?=$|[^0-9])", "seed_matrix", seed42.name, count=1)
    return seed42.with_name(name)


def run_seed_matrix(
    base_config_path: str | Path,
    *,
    state_dir: str | Path | None = None,
    dry_run: bool = False,
    trainer: Trainer | None = None,
    gpu: str = "0",
) -> dict[str, Any]:
    """Run or plan seeds 42/43/44 strictly serially on one visible GPU."""

    source = Path(base_config_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = _file_sha256(source)
    base_cfg = load_config(source)
    validate_base_protocol(base_cfg)
    logical_hash = experiment_config_hash(base_cfg)
    root = Path(state_dir) if state_dir is not None else _default_state_dir(base_cfg)
    root = root.resolve()
    derived = {seed: derive_seed_config(base_cfg, seed) for seed in SEEDS}
    for seed, cfg in derived.items():
        if experiment_config_hash(cfg) != logical_hash:
            raise SeedMatrixError(f"derived seed {seed} changed the frozen config")

    result: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "protocol": "CHP-GNS sequential GPU seeds; val20 only; test untouched",
        "base_config": str(source),
        "base_config_file_sha256": source_hash,
        "logical_config_hash": logical_hash,
        "gpu": str(gpu),
        "dry_run": bool(dry_run),
        "validation": {
            "split": "val",
            "cases": 20,
            "case_selection": "even",
            "transitions": 399,
            "test_split_accessed": False,
        },
        "seeds": [],
    }
    expected_structure: str | None = None
    records: list[dict[str, Any]] = []

    def assert_source_frozen() -> None:
        if _file_sha256(source) != source_hash:
            raise SeedMatrixError("base config file changed while the matrix was running")

    def process() -> None:
        nonlocal expected_structure
        for seed in SEEDS:
            cfg = derived[seed]
            output_dir = Path(str(get_cfg(cfg, "training.output_dir"))).resolve()
            checkpoint = output_dir / "best.pt"
            _assert_no_failure_marker(output_dir)
            if dry_run and seed > 42 and expected_structure is None:
                result["seeds"].append(
                    {
                        "seed": seed,
                        "output_dir": str(output_dir),
                        "checkpoint": str(checkpoint),
                        "gate_status": "not_run",
                        "action": "conditional_on_seed42_gate",
                    }
                )
                continue
            if checkpoint.is_file():
                record = inspect_seed_checkpoint(
                    checkpoint,
                    cfg,
                    expected_structure_hash=expected_structure,
                )
                record["action"] = "reuse"
                records.append(record)
                result["seeds"].append(record)
                if seed == 42:
                    expected_structure = str(record["structure_hash"])
                continue

            if dry_run:
                result["seeds"].append(
                    {
                        "seed": seed,
                        "output_dir": str(output_dir),
                        "checkpoint": str(checkpoint),
                        "gate_status": "not_run",
                        "action": "train",
                    }
                )
                continue

            if seed > 42 and expected_structure is None:
                raise SeedMatrixError(
                    "seed 42 pilot must exist and pass scientific gates before "
                    f"seed {seed} is allowed"
                )
            config_path = root / "configs" / f"seed{seed}.yaml"
            _write_yaml_atomic(config_path, cfg)
            assert_source_frozen()
            selected_trainer = trainer or _subprocess_trainer(gpu)
            selected_trainer(config_path, seed, output_dir)
            assert_source_frozen()
            record = inspect_seed_checkpoint(
                checkpoint,
                cfg,
                expected_structure_hash=expected_structure,
            )
            record["action"] = "trained"
            records.append(record)
            result["seeds"].append(record)
            if seed == 42:
                expected_structure = str(record["structure_hash"])

    if dry_run:
        process()
    else:
        with _exclusive_gpu_lock(root / ".gpu-seed-matrix.lock"):
            process()
        if len(records) != len(SEEDS):
            raise SeedMatrixError("three-seed matrix did not complete")
        result["structure_hash"] = expected_structure
        result["aggregate"] = aggregate_seed_records(records)
        if not result["aggregate"]["all_scientific_gates_passed"]:
            raise SeedMatrixError("one or more scientific gates did not pass")
        _write_json_atomic(root / "val20_seed_aggregate.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/deforming_plate_chp.full400.yaml",
        help="Immutable seed-42 base config.",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Generated seed configs and val20 aggregate (never test outputs).",
    )
    parser.add_argument("--gpu", default="0", help="Single CUDA device id.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate identities and print the serial plan without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_seed_matrix(
        args.config,
        state_dir=args.state_dir,
        dry_run=args.dry_run,
        gpu=args.gpu,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
