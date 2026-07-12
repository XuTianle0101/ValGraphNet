from types import SimpleNamespace
import json

import numpy as np
import pytest

from valgraphnet.physical_evaluation import (
    _case_has_full_cell_stress,
    CELL_TENSOR_STRESS_SOURCE,
    NODAL_STRESS_FALLBACK_SOURCE,
    ErrorSums,
    compare_experiments,
    evaluate_prediction,
    native_reference_payload,
    select_case_ids,
    validate_reference_protocol,
)


def _case():
    displacement = np.zeros((4, 3, 3), dtype=np.float32)
    displacement[:, 0, 0] = np.arange(4, dtype=np.float32)
    displacement[:, 1, 1] = 0.5 * np.arange(4, dtype=np.float32)
    stress = np.zeros((4, 3, 1), dtype=np.float32)
    stress[:, 0, 0] = np.asarray([0, 1, 2, 4], dtype=np.float32)
    stress[:, 1, 0] = np.asarray([0, 2, 4, 8], dtype=np.float32)
    return SimpleNamespace(
        case_id="synthetic",
        num_steps=4,
        num_nodes=3,
        displacement=displacement,
        stress=stress,
        fixed_mask=np.asarray([False, False, True]),
        prescribed_mask=np.asarray([False, False, True]),
    )


def test_empty_cell_tensor_channel_selects_nodal_fallback():
    absent = SimpleNamespace(
        case_id="nodal-only",
        num_steps=400,
        num_cells=17,
        cell_stress=np.zeros((400, 17, 0), dtype=np.float32),
    )
    assert not _case_has_full_cell_stress(absent)

    partial = SimpleNamespace(
        case_id="partial",
        num_steps=400,
        num_cells=17,
        cell_stress=np.zeros((400, 17, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match=r"\[T,M,6\]"):
        _case_has_full_cell_stress(partial)


def _result(sums: ErrorSums):
    return {
        "summary": sums.metrics(),
        "per_case": [{"case_id": "synthetic"}],
        "_error_sums": [
            {field: getattr(sums, field) for field in sums.__dataclass_fields__}
        ],
    }


def test_exact_standard_solution_has_zero_errors():
    case = _case()
    sums, metrics = evaluate_prediction(
        case, case.displacement.copy(), case.stress[1:].copy()
    )
    assert sums.diverged == 0
    assert metrics["moving_displacement_relative_rmse"] == 0.0
    assert metrics["stress_relative_rmse"] == 0.0
    assert metrics["stress_p95_relative_rmse"] == 0.0


def test_zero_prediction_is_unit_relative_error_and_uses_masks():
    case = _case()
    sums, metrics = evaluate_prediction(
        case,
        np.zeros_like(case.displacement),
        np.zeros_like(case.stress[1:]),
    )
    assert sums.u_count == 4 * 2 * 3
    assert sums.stress_count == 3 * 2
    assert np.isclose(metrics["moving_displacement_relative_rmse"], 1.0)
    assert np.isclose(metrics["stress_relative_rmse"], 1.0)
    assert metrics["stress_metric_source"] == NODAL_STRESS_FALLBACK_SOURCE
    json.dumps(metrics, allow_nan=False)


def test_full_cell_tensor_drives_primary_stress_metrics_and_requires_prediction():
    case = _case()
    case.num_cells = 1
    case.cell_stress = np.zeros((4, 1, 6), dtype=np.float32)
    case.cell_stress[1:, 0] = np.asarray(
        [2.0, -1.0, 0.5, 0.25, -0.75, 1.25], dtype=np.float32
    )
    exact_cell = np.zeros((3, 1, 3, 3), dtype=np.float32)
    exact_cell[..., 0, 0] = case.cell_stress[1:, :, 0]
    exact_cell[..., 1, 1] = case.cell_stress[1:, :, 1]
    exact_cell[..., 2, 2] = case.cell_stress[1:, :, 2]
    exact_cell[..., 0, 1] = exact_cell[..., 1, 0] = case.cell_stress[1:, :, 3]
    exact_cell[..., 0, 2] = exact_cell[..., 2, 0] = case.cell_stress[1:, :, 4]
    exact_cell[..., 1, 2] = exact_cell[..., 2, 1] = case.cell_stress[1:, :, 5]
    deliberately_wrong_nodal = np.zeros_like(case.stress[1:])

    _, metrics = evaluate_prediction(
        case,
        case.displacement.copy(),
        deliberately_wrong_nodal,
        exact_cell,
    )

    assert metrics["stress_metric_source"] == CELL_TENSOR_STRESS_SOURCE
    assert metrics["stress_relative_rmse"] == 0.0
    assert metrics["stress_p95_relative_rmse"] == 0.0
    assert metrics["nodal_stress_relative_rmse"] == 1.0
    assert metrics["cell_stress_tensor_case_coverage"] == 1.0
    json.dumps(metrics, allow_nan=False)

    with pytest.raises(ValueError, match="require S_cell_pred"):
        evaluate_prediction(
            case,
            case.displacement.copy(),
            deliberately_wrong_nodal,
        )


def test_mixed_tensor_and_nodal_stress_protocol_fails_closed():
    mixed = ErrorSums(
        stress_error=1.0,
        stress_reference=1.0,
        stress_count=1,
        p95_error=1.0,
        p95_reference=1.0,
        p95_count=1,
        cell_tensor_error=1.0,
        cell_tensor_reference=1.0,
        cell_tensor_count=1,
        cell_vm_p95_error=1.0,
        cell_vm_p95_reference=1.0,
        cell_vm_p95_count=1,
        tensor_label_cases=1,
        nodal_label_cases=1,
    )
    with pytest.raises(ValueError, match="mixes full cell-tensor"):
        mixed.metrics()


def test_paired_bootstrap_requires_every_metric_to_improve():
    baseline = ErrorSums(
        u_error=4,
        u_reference=4,
        u_count=4,
        final_error=1,
        final_reference=1,
        final_count=1,
        stress_error=9,
        stress_reference=9,
        stress_count=3,
        p95_error=4,
        p95_reference=4,
        p95_count=1,
    )
    candidate = ErrorSums(
        u_error=1,
        u_reference=4,
        u_count=4,
        final_error=0.25,
        final_reference=1,
        final_count=1,
        stress_error=2.25,
        stress_reference=9,
        stress_count=3,
        p95_error=1,
        p95_reference=4,
        p95_count=1,
    )
    comparison = compare_experiments(
        {"native": _result(baseline), "chp": _result(candidate)},
        baseline="native",
        candidate="chp",
        bootstrap_samples=100,
    )
    assert comparison["acceptance"][
        "all_primary_metrics_improve_at_least_10_percent"
    ]
    assert all(
        interval["ci95_low"] > 0
        for interval in comparison["paired_bootstrap"].values()
    )
    assert set(comparison["standard_reference"].values()) == {0.0}
    assert "not a trained-model result" in comparison["standard_reference_definition"]


def test_native_reference_protocol_requires_same_even_validation_subset(tmp_path):
    split_file = tmp_path / "splits.json"
    val_ids = [f"val_{index:05d}" for index in range(100)]
    split_file.write_text(
        json.dumps({"val": val_ids, "test": ["test_00000"]}),
        encoding="utf-8",
    )
    expected = select_case_ids(val_ids, 20, "even")
    payload = {
        "schema_version": 2,
        "evaluation": {"split": "val"},
        "per_case": [
            {"case_id": case_id, "evaluated_frames": 400}
            for case_id in expected
        ],
    }
    validate_reference_protocol(
        payload,
        split_file=split_file,
        split="val",
        case_count=20,
        frame_count=400,
        case_selection="even",
    )
    reference = native_reference_payload(
        {
            **payload,
            "summary": {
                "moving_displacement_relative_rmse": 1.0,
                "final_displacement_relative_rmse": 2.0,
                "stress_relative_rmse": 3.0,
                "stress_p95_relative_rmse": 4.0,
            },
        }
    )
    validate_reference_protocol(
        reference,
        split_file=split_file,
        split="val",
        case_count=20,
        frame_count=400,
        case_selection="even",
    )

    leaked = {**payload, "evaluation": {"split": "test"}}
    with pytest.raises(ValueError, match="split mismatch"):
        validate_reference_protocol(
            leaked,
            split_file=split_file,
            split="val",
            case_count=20,
            frame_count=400,
            case_selection="even",
        )

    wrong_subset = {
        **payload,
        "per_case": [
            {"case_id": case_id, "evaluated_frames": 400}
            for case_id in val_ids[:20]
        ],
    }
    with pytest.raises(ValueError, match="case set"):
        validate_reference_protocol(
            wrong_subset,
            split_file=split_file,
            split="val",
            case_count=20,
            frame_count=400,
            case_selection="even",
        )
