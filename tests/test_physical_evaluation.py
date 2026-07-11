from types import SimpleNamespace

import numpy as np

from valgraphnet.physical_evaluation import (
    ErrorSums,
    compare_experiments,
    evaluate_prediction,
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
