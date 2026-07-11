from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "data" / "deforming_plate_cases"
NATIVE_ROOT = ROOT / "outputs" / "deforming_plate_full_case_backed" / "rollout_eval"
BASELINE_ROOT = ROOT / "outputs" / "deforming_plate_case_full_gpu"
OPTIMIZED_ROOT = ROOT / "outputs" / "deforming_plate_case_optimized_stress_gpu"
MAIN_OPT_ROOT = ROOT / "outputs" / "deforming_plate_case_optimized_gpu"
OUT = ROOT / "outputs" / "deforming_plate_optimization"
MODEL_NAMES = ("native", "repo_baseline", "repo_optimized")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    case_ids = _read_json(CASE_ROOT / "splits.json")["test"]
    totals = {name: _MetricTotals() for name in MODEL_NAMES}
    reference = _ReferenceTotals()
    displacement_curves = {name: [] for name in MODEL_NAMES}
    moving_curves = {name: [] for name in MODEL_NAMES}
    stress_curves = {name: [] for name in MODEL_NAMES}
    per_case = []

    for case_id in case_ids:
        case_dir = CASE_ROOT / case_id
        exact_u = np.load(case_dir / "U.npy", allow_pickle=False)
        exact_s = np.load(case_dir / "S.npy", allow_pickle=False)
        node_type = np.load(case_dir / "node_type.npy", allow_pickle=False).reshape(-1)
        moving = node_type == 0
        native = np.load(NATIVE_ROOT / f"{case_id}.npz")
        errors_u = {
            "native": native["pred_world_pos"] - native["exact_world_pos"],
            "repo_baseline": _load_u_error(BASELINE_ROOT, case_id, exact_u),
            "repo_optimized": _load_u_error(OPTIMIZED_ROOT, case_id, exact_u),
        }
        errors_s = {
            "native": native["pred_stress"] - native["exact_stress"],
            "repo_baseline": _load_s_error(BASELINE_ROOT, case_id, exact_s),
            "repo_optimized": _load_s_error(OPTIMIZED_ROOT, case_id, exact_s),
        }

        sequence = {"case_id": case_id, "models": {}}
        for name in MODEL_NAMES:
            error_u = errors_u[name]
            error_s = errors_s[name]
            totals[name].update(error_u, error_s, moving)
            displacement_curves[name].append(_frame_rmse(error_u))
            moving_curves[name].append(_frame_rmse(error_u[:, moving]))
            stress_curves[name].append(_frame_rmse(error_s))
            sequence["models"][name] = {
                "displacement_rmse": _rmse(error_u),
                "moving_displacement_rmse": _rmse(error_u[:, moving]),
                "moving_final_displacement_rmse": _rmse(error_u[-1, moving]),
                "stress_rmse": _rmse(error_s),
            }
        reference.update(exact_u, exact_s, moving)
        per_case.append(sequence)

    metrics = {name: totals[name].finalize(reference) for name in MODEL_NAMES}
    metrics["standard_reference"] = {
        "displacement_rmse": 0.0,
        "moving_displacement_rmse": 0.0,
        "moving_final_displacement_rmse": 0.0,
        "stress_rmse": 0.0,
        "moving_stress_rmse": 0.0,
    }
    curves = {
        "displacement_frame": list(range(len(displacement_curves["native"][0]))),
        "stress_frame": list(range(1, len(stress_curves["native"][0]) + 1)),
    }
    for name in MODEL_NAMES:
        curves[f"{name}_displacement_rmse"] = _mean_curve(displacement_curves[name])
        curves[f"{name}_moving_displacement_rmse"] = _mean_curve(moving_curves[name])
        curves[f"{name}_stress_rmse"] = _mean_curve(stress_curves[name])

    training = _training_summary()
    result = {
        "experiment": {
            "case_ids": case_ids,
            "displacement_frames": 200,
            "stress_frames": 199,
            "standard_solution": "DeepMind deforming_plate ground-truth trajectory",
            "native": "PhysicsNeMo native deforming_plate model",
            "repo_baseline": "ValGraphNet joint decoder, one-step checkpoint selection",
            "repo_optimized": (
                "ValGraphNet independent stress decoder, 3-step rollout loss, "
                "rollout checkpoint selection"
            ),
        },
        "reference_rms": reference.finalize(),
        "metrics": metrics,
        "improvement_percent": {
            "optimized_vs_native": _improvements(metrics["native"], metrics["repo_optimized"]),
            "optimized_vs_repo_baseline": _improvements(
                metrics["repo_baseline"], metrics["repo_optimized"]
            ),
        },
        "training": training,
        "per_case": per_case,
        "milestones": _milestones(curves, [0, 49, 99, 149, 199]),
        "curves": curves,
    }
    _write_json(OUT / "results.json", result)
    _write_report(OUT / "report.md", result)
    _save_plots(result)
    print(json.dumps(result["metrics"], indent=2))
    print(f"optimization analysis written to: {OUT}")
    return 0


class _MetricTotals:
    def __init__(self) -> None:
        self.values = {
            key: [0.0, 0]
            for key in (
                "u",
                "u_moving",
                "u_final_moving",
                "stress",
                "stress_moving",
            )
        }

    def update(self, error_u: np.ndarray, error_s: np.ndarray, moving: np.ndarray) -> None:
        self._add("u", error_u)
        self._add("u_moving", error_u[:, moving])
        self._add("u_final_moving", error_u[-1, moving])
        self._add("stress", error_s)
        self._add("stress_moving", error_s[:, moving])

    def _add(self, key: str, value: np.ndarray) -> None:
        value = value.astype(np.float64)
        self.values[key][0] += float(np.square(value).sum())
        self.values[key][1] += int(value.size)

    def finalize(self, reference: "_ReferenceTotals") -> dict[str, float]:
        rms = {
            key: float(np.sqrt(total / max(count, 1)))
            for key, (total, count) in self.values.items()
        }
        ref = reference.finalize()
        return {
            "displacement_rmse": rms["u"],
            "moving_displacement_rmse": rms["u_moving"],
            "moving_final_displacement_rmse": rms["u_final_moving"],
            "moving_displacement_relative_rmse": (
                rms["u_moving"] / ref["moving_displacement_rms"]
            ),
            "stress_rmse": rms["stress"],
            "stress_relative_rmse": rms["stress"] / ref["stress_rms"],
            "moving_stress_rmse": rms["stress_moving"],
            "moving_stress_relative_rmse": (
                rms["stress_moving"] / ref["moving_stress_rms"]
            ),
        }


class _ReferenceTotals:
    def __init__(self) -> None:
        self.values = {
            key: [0.0, 0]
            for key in ("u", "u_moving", "stress", "stress_moving")
        }

    def update(self, exact_u: np.ndarray, exact_s: np.ndarray, moving: np.ndarray) -> None:
        for key, value in (
            ("u", exact_u),
            ("u_moving", exact_u[:, moving]),
            ("stress", exact_s[1:]),
            ("stress_moving", exact_s[1:, moving]),
        ):
            value = value.astype(np.float64)
            self.values[key][0] += float(np.square(value).sum())
            self.values[key][1] += int(value.size)

    def finalize(self) -> dict[str, float]:
        rms = {
            key: float(np.sqrt(total / max(count, 1)))
            for key, (total, count) in self.values.items()
        }
        return {
            "displacement_rms": rms["u"],
            "moving_displacement_rms": rms["u_moving"],
            "stress_rms": rms["stress"],
            "moving_stress_rms": rms["stress_moving"],
        }


def _load_u_error(root: Path, case_id: str, exact: np.ndarray) -> np.ndarray:
    pred = np.load(root / f"rollout_{case_id}" / "U_pred.npy", allow_pickle=False)
    return pred - exact[: pred.shape[0]]


def _load_s_error(root: Path, case_id: str, exact: np.ndarray) -> np.ndarray:
    pred = np.load(root / f"rollout_{case_id}" / "S_pred.npy", allow_pickle=False)
    return pred - exact[1 : pred.shape[0] + 1]


def _training_summary() -> dict[str, Any]:
    baseline = _read_json(BASELINE_ROOT / "history.json")
    main = _read_json(MAIN_OPT_ROOT / "history.json")
    stress = _read_json(OPTIMIZED_ROOT / "history.json")
    baseline_best = min(baseline, key=lambda item: item["val"]["total"])
    main_best = min(main, key=lambda item: item["rollout_val"]["score"])
    stress_best = min(stress, key=lambda item: item["rollout_val"]["score"])
    return {
        "repo_baseline": {
            "selection_metric": "one-step validation total",
            "best_epoch": baseline_best["epoch"],
            "best_score": baseline_best["val"]["total"],
            "gpu_minutes": sum(item["seconds"] for item in baseline) / 60.0,
        },
        "rollout_main": {
            "selection_metric": "5-case rollout score",
            "best_epoch": main_best["epoch"],
            "best_score": main_best["rollout_val"]["score"],
            "gpu_minutes": sum(item["seconds"] for item in main) / 60.0,
        },
        "rollout_stress_decoupled": {
            "selection_metric": "5-case rollout score",
            "best_epoch": stress_best["epoch"],
            "best_score": stress_best["rollout_val"]["score"],
            "gpu_minutes": sum(item["seconds"] for item in stress) / 60.0,
        },
    }


def _improvements(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        key: 100.0 * (baseline[key] - candidate[key]) / max(abs(baseline[key]), 1.0e-12)
        for key in (
            "displacement_rmse",
            "moving_displacement_rmse",
            "moving_final_displacement_rmse",
            "stress_rmse",
            "moving_stress_rmse",
        )
    }


def _milestones(curves: dict[str, list[float]], frames: list[int]) -> list[dict[str, float]]:
    result = []
    for frame in frames:
        if frame >= len(curves["displacement_frame"]):
            continue
        item: dict[str, float] = {"frame": float(frame)}
        for name in MODEL_NAMES:
            item[name] = curves[f"{name}_moving_displacement_rmse"][frame]
        result.append(item)
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    improvement = result["improvement_percent"]
    training = result["training"]
    displacement_wins = sum(
        case["models"]["repo_optimized"]["moving_displacement_rmse"]
        < case["models"]["repo_baseline"]["moving_displacement_rmse"]
        for case in result["per_case"]
    )
    stress_wins = sum(
        case["models"]["repo_optimized"]["stress_rmse"]
        < case["models"]["repo_baseline"]["stress_rmse"]
        for case in result["per_case"]
    )
    baseline_stress_curve = np.asarray(result["curves"]["repo_baseline_stress_rmse"])
    optimized_stress_curve = np.asarray(result["curves"]["repo_optimized_stress_rmse"])
    worse_frames = np.flatnonzero(optimized_stress_curve > baseline_stress_curve)
    first_worse_stress_frame = int(worse_frames[0] + 1) if worse_frames.size else None
    optimization_minutes = (
        training["rollout_main"]["gpu_minutes"]
        + training["rollout_stress_decoupled"]["gpu_minutes"]
    )
    lines = [
        "# Deforming Plate 优化迭代报告",
        "",
        "## 对比口径",
        "",
        "- 标准解：DeepMind deforming_plate 数据集中 5 条 test 真值轨迹，误差定义为 0。",
        "- native：PhysicsNeMo 原生 deforming_plate 路径。",
        "- repo baseline：上一轮 ValGraphNet joint decoder 模型。",
        "- repo optimized：独立 stress decoder、3-step rollout loss、rollout checkpoint 模型。",
        "- 位移统计 200 帧，应力统计 199 个预测帧；moving 节点统一使用 `node_type == 0`。",
        "",
        "## 聚合结果",
        "",
        "| 模型 | 全节点位移 RMSE | moving 位移 RMSE | moving 末帧 RMSE | 应力 RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("standard_reference", *MODEL_NAMES):
        value = metrics[name]
        lines.append(
            f"| {name} | {value['displacement_rmse']:.6g} | "
            f"{value['moving_displacement_rmse']:.6g} | "
            f"{value['moving_final_displacement_rmse']:.6g} | "
            f"{value['stress_rmse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## 优化收益",
            "",
            f"- 相对 native，optimized moving 位移改善 "
            f"`{improvement['optimized_vs_native']['moving_displacement_rmse']:.2f}%`，"
            f"末帧改善 `{improvement['optimized_vs_native']['moving_final_displacement_rmse']:.2f}%`。",
            f"- 相对 repo baseline，optimized moving 位移变化 "
            f"`{improvement['optimized_vs_repo_baseline']['moving_displacement_rmse']:.2f}%`，"
            f"应力变化 `{improvement['optimized_vs_repo_baseline']['stress_rmse']:.2f}%`。",
            f"- optimized 在 `{displacement_wins}/5` 条 test 轨迹上改善 moving 位移；"
            f"在 `{stress_wins}/5` 条轨迹上改善应力。",
            f"- stress 曲线在约第 `{first_worse_stress_frame}` 个预测帧后超过 baseline，"
            "后半段误差累积是 test stress 退化的主要来源。",
            "",
            "## Checkpoint 选择",
            "",
            f"- 主多步实验 best epoch `{training['rollout_main']['best_epoch']}`，"
            f"rollout score `{training['rollout_main']['best_score']:.6g}`。",
            f"- stress 解耦实验 best epoch `{training['rollout_stress_decoupled']['best_epoch']}`，"
            f"rollout score `{training['rollout_stress_decoupled']['best_score']:.6g}`。",
            "- one-step 最优轮次与 rollout 最优轮次不一致，最终模型严格按 rollout score 选择。",
            f"- 两阶段优化 GPU 训练合计约 `{optimization_minutes:.2f}` 分钟。",
            "",
            "## 结论与边界",
            "",
            "- 多步 rollout loss 对位移闭环稳定性有效，且收益在 5 条 test 轨迹上方向一致。",
            "- 独立 stress decoder 与 teacher-forced stress 监督显著改善了验证 rollout score，"
            "但 test stress 未超过 repo baseline，说明 stress 泛化仍是当前主要短板。",
            "- native 的 stress RMSE 仍最低；若任务以应力为第一目标，当前应保留 native 作为首选。",
            "- 不再依据这 5 条 test 轨迹继续调参，以避免 test-set leakage；后续 stress 改进应扩大"
            "验证轨迹覆盖并加入长程 stress rollout 选择指标。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_plots(result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = result["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in MODEL_NAMES:
        axes[0].plot(
            curves["displacement_frame"],
            curves[f"{name}_moving_displacement_rmse"],
            label=name,
        )
        axes[1].plot(
            curves["stress_frame"], curves[f"{name}_stress_rmse"], label=name
        )
    axes[0].set(title="Moving-node rollout error", xlabel="frame", ylabel="RMSE")
    axes[1].set(title="Stress rollout error", xlabel="frame", ylabel="RMSE")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "three_way_rollout.png", dpi=180)
    plt.close(fig)

    main_history = _read_json(MAIN_OPT_ROOT / "history.json")
    stress_history = _read_json(OPTIMIZED_ROOT / "history.json")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        [item["epoch"] for item in main_history],
        [item["rollout_val"]["score"] for item in main_history],
        marker="o",
        label="3-step joint stress supervision",
    )
    ax.plot(
        [item["epoch"] for item in stress_history],
        [item["rollout_val"]["score"] for item in stress_history],
        marker="o",
        label="step-0 detached stress supervision",
    )
    ax.set(title="Rollout checkpoint score", xlabel="epoch", ylabel="score")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "rollout_checkpoint_selection.png", dpi=180)
    plt.close(fig)


def _frame_rmse(error: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, error.ndim))
    return np.sqrt(np.mean(np.square(error.astype(np.float64)), axis=axes))


def _mean_curve(curves: list[np.ndarray]) -> list[float]:
    length = min(len(curve) for curve in curves)
    return np.mean(np.stack([curve[:length] for curve in curves]), axis=0).tolist()


def _rmse(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
