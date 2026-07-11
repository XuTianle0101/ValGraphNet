"""Shared physical-unit metrics for deforming-plate simulator comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from valgraphnet.data.case import ValveCase, load_case, read_split_file


PRIMARY_METRICS = (
    "moving_displacement_relative_rmse",
    "final_displacement_relative_rmse",
    "stress_relative_rmse",
    "stress_p95_relative_rmse",
)
METRIC_SCHEMA_VERSION = 2


def select_case_ids(
    case_ids: list[str],
    max_cases: int | None,
    selection: str = "head",
) -> list[str]:
    """Select a deterministic trajectory subset for protocol-aligned evaluation."""

    if max_cases is None or int(max_cases) >= len(case_ids):
        return list(case_ids)
    count = max(int(max_cases), 0)
    if count == 0:
        return []
    if selection == "head":
        return list(case_ids[:count])
    if selection == "even":
        indices = np.linspace(0, len(case_ids) - 1, num=count).round().astype(np.int64)
        return [case_ids[int(index)] for index in indices]
    raise ValueError(f"unsupported case selection: {selection}")


@dataclass
class ErrorSums:
    u_error: float = 0.0
    u_reference: float = 0.0
    u_count: int = 0
    final_error: float = 0.0
    final_reference: float = 0.0
    final_count: int = 0
    stress_error: float = 0.0
    stress_reference: float = 0.0
    stress_count: int = 0
    p95_error: float = 0.0
    p95_reference: float = 0.0
    p95_count: int = 0
    diverged: int = 0

    def __add__(self, other: "ErrorSums") -> "ErrorSums":
        return ErrorSums(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            }
        )

    def metrics(self) -> dict[str, float]:
        eps = 1.0e-30
        result = {
            "moving_displacement_rmse": math.sqrt(
                self.u_error / max(self.u_count, 1)
            ),
            "moving_displacement_relative_rmse": math.sqrt(
                self.u_error / max(self.u_reference, eps)
            ),
            "final_displacement_rmse": math.sqrt(
                self.final_error / max(self.final_count, 1)
            ),
            "final_displacement_relative_rmse": math.sqrt(
                self.final_error / max(self.final_reference, eps)
            ),
            "stress_rmse": math.sqrt(self.stress_error / max(self.stress_count, 1)),
            "stress_relative_rmse": math.sqrt(
                self.stress_error / max(self.stress_reference, eps)
            ),
            "stress_p95_rmse": math.sqrt(
                self.p95_error / max(self.p95_count, 1)
            ),
            "stress_p95_relative_rmse": math.sqrt(
                self.p95_error / max(self.p95_reference, eps)
            ),
            "diverged_cases": float(self.diverged),
        }
        if self.diverged:
            for key in PRIMARY_METRICS:
                result[key] = max(result[key], 1.0e6)
        return result


def evaluate_prediction(
    case: ValveCase,
    displacement_prediction: np.ndarray,
    stress_prediction: np.ndarray,
) -> tuple[ErrorSums, dict[str, float]]:
    """Evaluate one trajectory with moving and non-prescribed masks."""

    predicted_u = np.asarray(displacement_prediction, dtype=np.float64)
    predicted_s = np.asarray(stress_prediction, dtype=np.float64)
    truth_u = np.asarray(case.displacement, dtype=np.float64)
    if predicted_u.ndim != 3 or predicted_u.shape[1:] != truth_u.shape[1:]:
        raise ValueError(f"{case.case_id}: U_pred must have shape [T,N,3]")
    steps = min(predicted_u.shape[0], truth_u.shape[0])
    predicted_u = predicted_u[:steps]
    truth_u = truth_u[:steps]
    moving = ~(case.fixed_mask | case.prescribed_mask)
    if not moving.any():
        moving = ~case.fixed_mask
    stress_mask = ~case.prescribed_mask
    if not stress_mask.any():
        stress_mask = np.ones(case.num_nodes, dtype=bool)

    finite_u = bool(np.isfinite(predicted_u).all())
    if not finite_u:
        predicted_u = np.nan_to_num(predicted_u, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
    u_residual = predicted_u[:, moving] - truth_u[:, moving]
    final_residual = predicted_u[-1, moving] - truth_u[-1, moving]

    truth_s = np.asarray(case.stress[1:steps, stress_mask, :1], dtype=np.float64)
    if predicted_s.ndim == 2:
        predicted_s = predicted_s[..., None]
    # Standard artifact convention stores stress for frames 1..T-1. Accept a
    # full T sequence as well, dropping its reference frame.
    if predicted_s.shape[0] == steps:
        predicted_s = predicted_s[1:]
    stress_steps = min(predicted_s.shape[0], truth_s.shape[0])
    predicted_s = predicted_s[:stress_steps, stress_mask, :1]
    truth_s = truth_s[:stress_steps]
    finite_s = bool(np.isfinite(predicted_s).all())
    if not finite_s:
        predicted_s = np.nan_to_num(
            predicted_s, nan=1.0e12, posinf=1.0e12, neginf=-1.0e12
        )
    stress_residual = predicted_s - truth_s
    if truth_s.size:
        threshold = float(np.quantile(np.abs(truth_s), 0.95))
        peak = np.abs(truth_s) >= threshold
    else:
        peak = np.zeros_like(truth_s, dtype=bool)

    sums = ErrorSums(
        u_error=float(np.sum(np.square(u_residual), dtype=np.float64)),
        u_reference=float(np.sum(np.square(truth_u[:, moving]), dtype=np.float64)),
        u_count=int(u_residual.size),
        final_error=float(np.sum(np.square(final_residual), dtype=np.float64)),
        final_reference=float(np.sum(np.square(truth_u[-1, moving]), dtype=np.float64)),
        final_count=int(final_residual.size),
        stress_error=float(np.sum(np.square(stress_residual), dtype=np.float64)),
        stress_reference=float(np.sum(np.square(truth_s), dtype=np.float64)),
        stress_count=int(stress_residual.size),
        p95_error=float(np.sum(np.square(stress_residual[peak]), dtype=np.float64)),
        p95_reference=float(np.sum(np.square(truth_s[peak]), dtype=np.float64)),
        p95_count=int(np.count_nonzero(peak)),
        diverged=int(not finite_u or not finite_s or steps < case.num_steps),
    )
    metrics = sums.metrics()
    metrics["case_id"] = case.case_id
    metrics["evaluated_frames"] = float(steps)
    return sums, metrics


def evaluate_prediction_directory(
    case_root: str | Path,
    split_file: str | Path,
    split: str,
    prediction_root: str | Path,
    *,
    output_path: str | Path | None = None,
    max_cases: int | None = None,
    case_selection: str = "head",
) -> dict[str, Any]:
    """Evaluate standardized ``<case>/U_pred.npy,S_pred.npy`` artifacts."""

    case_root = Path(case_root)
    prediction_root = Path(prediction_root)
    case_ids = read_split_file(split_file, split)
    case_ids = select_case_ids(case_ids, max_cases, case_selection)
    total = ErrorSums()
    per_case: list[dict[str, float]] = []
    raw_sums: list[ErrorSums] = []
    missing: list[str] = []
    for case_id in case_ids:
        artifact = prediction_root / case_id
        if not (artifact / "U_pred.npy").exists() or not (artifact / "S_pred.npy").exists():
            missing.append(case_id)
            continue
        case = load_case(case_root / case_id)
        sums, metrics = evaluate_prediction(
            case,
            np.load(artifact / "U_pred.npy", allow_pickle=False),
            np.load(artifact / "S_pred.npy", allow_pickle=False),
        )
        total = total + sums
        raw_sums.append(sums)
        per_case.append(metrics)
    summary = total.metrics()
    summary["evaluated_cases"] = float(len(per_case))
    summary["missing_cases"] = float(len(missing))
    result: dict[str, Any] = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "evaluation": {
            "split": str(split),
            "case_selection": str(case_selection),
            "requested_case_ids": list(case_ids),
            "evaluated_case_ids": [value["case_id"] for value in per_case],
        },
        "metric_definition": {
            "displacement_mask": "~(fixed|prescribed)",
            "stress_mask": "~prescribed",
            "p95_region": "per-trajectory truth top 5% stress values",
            "aggregation": "pooled physical-unit squared errors",
        },
        "summary": summary,
        "per_case": per_case,
        "missing_case_ids": missing,
        "_error_sums": [_sums_dict(value) for value in raw_sums],
    }
    if output_path is not None:
        _write_json(Path(output_path), result)
    return result


def validate_reference_protocol(
    payload: dict[str, Any],
    *,
    split_file: str | Path,
    split: str,
    case_count: int,
    frame_count: int | None = None,
    case_selection: str = "even",
) -> None:
    """Reject checkpoint references evaluated on a different development set."""

    if int(payload.get("schema_version", 0)) < METRIC_SCHEMA_VERSION:
        raise ValueError("native reference uses a metric schema without provenance")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("native reference is missing evaluation provenance")
    actual_split = str(evaluation.get("split", ""))
    if actual_split != str(split):
        raise ValueError(
            f"native reference split mismatch: expected {split!r}, got {actual_split!r}"
        )
    expected_ids = select_case_ids(
        read_split_file(split_file, split),
        case_count,
        case_selection,
    )
    actual_ids = [str(value.get("case_id")) for value in payload.get("per_case", [])]
    if actual_ids != expected_ids:
        raise ValueError(
            "native reference case set does not match the ordered validation subset"
        )
    if frame_count is not None:
        wrong_frames = [
            value.get("case_id")
            for value in payload.get("per_case", [])
            if int(value.get("evaluated_frames", -1)) != int(frame_count)
        ]
        if wrong_frames:
            raise ValueError(
                "native reference frame count does not match validation: "
                + ", ".join(map(str, wrong_frames[:3]))
            )


def compare_experiments(
    experiments: dict[str, dict[str, Any]],
    *,
    baseline: str,
    candidate: str,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Create paired trajectory bootstrap CIs and strict acceptance evidence."""

    if baseline not in experiments or candidate not in experiments:
        raise KeyError("baseline and candidate must be present in experiments")
    base_ids = [value["case_id"] for value in experiments[baseline]["per_case"]]
    candidate_ids = [value["case_id"] for value in experiments[candidate]["per_case"]]
    if base_ids != candidate_ids:
        raise ValueError("paired comparison requires identical ordered case ids")
    base_sums = [_dict_sums(value) for value in experiments[baseline]["_error_sums"]]
    candidate_sums = [
        _dict_sums(value) for value in experiments[candidate]["_error_sums"]
    ]
    if len(base_sums) != len(candidate_sums) or not base_sums:
        raise ValueError("paired comparison requires the same non-empty case set")
    rng = np.random.default_rng(int(seed))
    improvements = {key: [] for key in PRIMARY_METRICS}
    for _ in range(max(int(bootstrap_samples), 1)):
        indices = rng.integers(0, len(base_sums), size=len(base_sums))
        base_total = _sum_selected(base_sums, indices).metrics()
        candidate_total = _sum_selected(candidate_sums, indices).metrics()
        for key in PRIMARY_METRICS:
            denominator = max(float(base_total[key]), 1.0e-30)
            improvements[key].append(
                (float(base_total[key]) - float(candidate_total[key])) / denominator
            )
    intervals = {
        key: {
            "mean_improvement": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in improvements.items()
    }
    complete = all(
        float(experiments[name]["summary"].get("missing_cases", 0.0)) == 0.0
        for name in (baseline, candidate)
    )
    acceptance = complete and all(
        value["mean_improvement"] >= 0.10 and value["ci95_low"] > 0.0
        for value in intervals.values()
    )
    public = {
        name: {key: value for key, value in result.items() if key != "_error_sums"}
        for name, result in experiments.items()
    }
    return {
        "standard_reference": {key: 0.0 for key in PRIMARY_METRICS},
        "experiments": public,
        "paired_bootstrap": intervals,
        "acceptance": {
            "all_primary_metrics_improve_at_least_10_percent": acceptance,
            "complete_paired_case_set": complete,
            "candidate": candidate,
            "baseline": baseline,
        },
    }


def save_comparison(path: str | Path, comparison: dict[str, Any]) -> None:
    _write_json(Path(path), comparison)


def native_reference_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(result.get("schema_version", METRIC_SCHEMA_VERSION)),
        "evaluation": dict(result.get("evaluation", {})),
        "per_case": [
            {
                "case_id": value["case_id"],
                "evaluated_frames": value["evaluated_frames"],
            }
            for value in result.get("per_case", [])
        ],
        "rollout": {
            key: float(result["summary"][key]) for key in PRIMARY_METRICS
        }
    }


def _sum_selected(values: list[ErrorSums], indices: Iterable[int]) -> ErrorSums:
    total = ErrorSums()
    for index in indices:
        total = total + values[int(index)]
    return total


def _sums_dict(value: ErrorSums) -> dict[str, float | int]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


def _dict_sums(value: dict[str, Any]) -> ErrorSums:
    return ErrorSums(**{field: value[field] for field in ErrorSums.__dataclass_fields__})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)
