from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "outputs" / "deforming_plate_full_experiment"
NATIVE_OUT = ROOT / "outputs" / "deforming_plate_full_case_backed"
CASE_OUT = ROOT / "outputs" / "deforming_plate_case_full_gpu"
CASE_ROOT = ROOT / "data" / "deforming_plate_cases"


def main() -> int:
    comparison = read_json(EXP_DIR / "comparison.json")
    split = read_json(CASE_ROOT / "splits.json")
    case_ids = split["test"]

    native_time = []
    case_time = []
    native_moving_time = []
    case_moving_time = []
    native_stress_time = []
    case_stress_time = []
    sequence_analysis = []
    totals = _Totals()

    for case_id in case_ids:
        native = np.load(NATIVE_OUT / "rollout_eval" / f"{case_id}.npz")
        case_dir = CASE_ROOT / case_id
        case_rollout = CASE_OUT / f"rollout_{case_id}"
        nodes = np.load(case_dir / "nodes.npy", allow_pickle=False)
        exact_u = np.load(case_dir / "U.npy", allow_pickle=False)
        exact_s = np.load(case_dir / "S.npy", allow_pickle=False)
        node_type = np.load(case_dir / "node_type.npy", allow_pickle=False).reshape(-1)
        moving = node_type == 0

        native_error = native["pred_world_pos"] - native["exact_world_pos"]
        case_pred_u = np.load(case_rollout / "U_pred.npy", allow_pickle=False)
        case_error = case_pred_u - exact_u[: case_pred_u.shape[0]]
        native_stress_error = native["pred_stress"] - native["exact_stress"]
        case_pred_s = np.load(case_rollout / "S_pred.npy", allow_pickle=False)
        case_exact_s = exact_s[1 : 1 + case_pred_s.shape[0]]
        case_stress_error = case_pred_s - case_exact_s

        native_time.append(_time_rmse(native_error))
        case_time.append(_time_rmse(case_error))
        native_moving_time.append(_time_rmse(native_error[:, moving]))
        case_moving_time.append(_time_rmse(case_error[:, moving]))
        native_stress_time.append(_time_rmse(native_stress_error))
        case_stress_time.append(_time_rmse(case_stress_error))

        exact_u_moving = exact_u[: case_pred_u.shape[0], moving]
        totals.update(
            native_error[:, moving],
            case_error[:, moving],
            exact_u_moving,
            native_stress_error,
            case_stress_error,
            case_exact_s,
        )
        sequence_analysis.append(
            {
                "case_id": case_id,
                "moving_displacement_rmse": {
                    "native": _rmse(native_error[:, moving]),
                    "valgraphnet_case": _rmse(case_error[:, moving]),
                },
                "moving_final_rmse": {
                    "native": _rmse(native_error[-1, moving]),
                    "valgraphnet_case": _rmse(case_error[-1, moving]),
                },
                "exact_displacement_rms": _rms(exact_u_moving),
            }
        )

    curves = {
        "frame": list(range(len(native_time[0]))),
        "native_displacement_rmse": _mean_curve(native_time),
        "case_displacement_rmse": _mean_curve(case_time),
        "native_moving_displacement_rmse": _mean_curve(native_moving_time),
        "case_moving_displacement_rmse": _mean_curve(case_moving_time),
        "native_stress_rmse": _mean_curve(native_stress_time),
        "case_stress_rmse": _mean_curve(case_stress_time),
    }
    native_history = comparison["training_history"]["native"]
    case_history = comparison["training_history"]["valgraphnet_case"]
    native_best = min(native_history, key=lambda item: item["val_loss"])
    case_best = min(case_history, key=lambda item: item["val"]["total"])
    official = comparison["native"], comparison["valgraphnet_case"]
    analysis = {
        "experiment": comparison["experiment"],
        "official_metrics": {
            "native": official[0],
            "valgraphnet_case": official[1],
            "case_improvement_percent": {
                "displacement_rmse": _improvement(
                    official[0]["displacement_rmse"], official[1]["displacement_rmse"]
                ),
                "rollout_rmse": _improvement(
                    official[0]["rollout_rmse"], official[1]["rollout_rmse"]
                ),
                "stress_rmse": _improvement(
                    official[0]["stress_rmse"], official[1]["stress_rmse"]
                ),
            },
        },
        "moving_node_metrics": totals.finalize(),
        "training": {
            "native": {
                "best_epoch": native_best["epoch"],
                "best_val_loss": native_best["val_loss"],
                "first_val_loss": native_history[0]["val_loss"],
                "improvement_percent": _improvement(
                    native_history[0]["val_loss"], native_best["val_loss"]
                ),
                "total_minutes": sum(item["seconds"] for item in native_history) / 60.0,
            },
            "valgraphnet_case": {
                "best_epoch": case_best["epoch"],
                "best_val_total": case_best["val"]["total"],
                "best_val_components": case_best["val"],
                "first_val_total": case_history[0]["val"]["total"],
                "improvement_percent": _improvement(
                    case_history[0]["val"]["total"], case_best["val"]["total"]
                ),
                "total_minutes": sum(item["seconds"] for item in case_history) / 60.0,
            },
        },
        "milestones": _milestones(curves, [0, 49, 99, 149, 199]),
        "sequences": sequence_analysis,
        "curves": curves,
    }
    write_json(EXP_DIR / "analysis.json", analysis)
    write_report(EXP_DIR / "report.md", analysis)
    save_plots(analysis, native_history, case_history)
    print(f"analysis written to: {EXP_DIR}")
    return 0


class _Totals:
    def __init__(self) -> None:
        self.values = {key: [0.0, 0] for key in (
            "native_error",
            "case_error",
            "exact_u",
            "native_stress_error",
            "case_stress_error",
            "exact_stress",
        )}

    def update(self, *arrays: np.ndarray) -> None:
        for key, array in zip(self.values, arrays, strict=True):
            self.values[key][0] += float(np.square(array.astype(np.float64)).sum())
            self.values[key][1] += int(array.size)

    def finalize(self) -> dict[str, float]:
        rms = {key: float(np.sqrt(total / max(count, 1))) for key, (total, count) in self.values.items()}
        return {
            "native_displacement_rmse": rms["native_error"],
            "case_displacement_rmse": rms["case_error"],
            "case_displacement_improvement_percent": _improvement(
                rms["native_error"], rms["case_error"]
            ),
            "exact_displacement_rms": rms["exact_u"],
            "native_displacement_relative_rmse": rms["native_error"] / rms["exact_u"],
            "case_displacement_relative_rmse": rms["case_error"] / rms["exact_u"],
            "native_stress_rmse": rms["native_stress_error"],
            "case_stress_rmse": rms["case_stress_error"],
            "exact_stress_rms": rms["exact_stress"],
            "native_stress_relative_rmse": rms["native_stress_error"] / rms["exact_stress"],
            "case_stress_relative_rmse": rms["case_stress_error"] / rms["exact_stress"],
        }


def _time_rmse(error: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(error.astype(np.float64)), axis=tuple(range(1, error.ndim))))


def _mean_curve(curves: list[np.ndarray]) -> list[float]:
    length = min(len(curve) for curve in curves)
    return np.mean(np.stack([curve[:length] for curve in curves]), axis=0).tolist()


def _milestones(curves: dict[str, list], frames: list[int]) -> list[dict[str, float | int]]:
    out = []
    for frame in frames:
        if frame >= len(curves["frame"]):
            continue
        out.append(
            {
                "frame": frame,
                "native_moving_rmse": curves["native_moving_displacement_rmse"][frame],
                "case_moving_rmse": curves["case_moving_displacement_rmse"][frame],
            }
        )
    return out


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error.astype(np.float64)))))


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _improvement(baseline: float, candidate: float) -> float:
    return 100.0 * (float(baseline) - float(candidate)) / max(abs(float(baseline)), 1.0e-12)


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    official = analysis["official_metrics"]
    moving = analysis["moving_node_metrics"]
    training = analysis["training"]
    lines = [
        "# Deforming Plate 全量实验报告",
        "",
        "## 结论",
        "",
        f"- case 路径平均位移 RMSE 为 `{official['valgraphnet_case']['displacement_rmse']:.6g}`，"
        f"相对 native 改善 `{official['case_improvement_percent']['displacement_rmse']:.2f}%`。",
        f"- case 路径末帧 RMSE 为 `{official['valgraphnet_case']['rollout_rmse']:.6g}`，"
        f"相对 native 改善 `{official['case_improvement_percent']['rollout_rmse']:.2f}%`。",
        f"- case 路径应力 RMSE 为 `{official['valgraphnet_case']['stress_rmse']:.6g}`，"
        f"相对 native 变化 `{official['case_improvement_percent']['stress_rmse']:.2f}%`；应力仍是主要短板。",
        "",
        "## Moving 节点",
        "",
        f"- native moving-node 相对位移 RMSE：`{moving['native_displacement_relative_rmse']:.4f}`。",
        f"- case moving-node 相对位移 RMSE：`{moving['case_displacement_relative_rmse']:.4f}`。",
        f"- native 相对应力 RMSE：`{moving['native_stress_relative_rmse']:.4f}`。",
        f"- case 相对应力 RMSE：`{moving['case_stress_relative_rmse']:.4f}`。",
        "",
        "## 训练",
        "",
        f"- native 最佳 epoch `{training['native']['best_epoch']}`，验证损失 "
        f"`{training['native']['best_val_loss']:.6g}`，相对首轮下降 "
        f"`{training['native']['improvement_percent']:.2f}%`。",
        f"- case 最佳 epoch `{training['valgraphnet_case']['best_epoch']}`，验证总损失 "
        f"`{training['valgraphnet_case']['best_val_total']:.6g}`，相对首轮下降 "
        f"`{training['valgraphnet_case']['improvement_percent']:.2f}%`。",
        "",
        "## 后续优化建议",
        "",
        "1. 位移任务优先采用 case 路径；它在五条测试轨迹上总体更稳。",
        "2. 应力任务保留 native 路径，或为 case 模型增加独立 stress decoder 与更均衡的应力时间采样。",
        "3. 下一轮实验应按 moving-node rollout 指标选 checkpoint，而不只用 one-step 加权验证损失。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(
    analysis: dict[str, Any],
    native_history: list[dict[str, Any]],
    case_history: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([x["epoch"] for x in native_history], [x["train_loss"] for x in native_history], label="train")
    axes[0].plot([x["epoch"] for x in native_history], [x["val_loss"] for x in native_history], label="val")
    axes[0].set(title="Native loss", xlabel="epoch", yscale="log")
    axes[0].legend()
    axes[1].plot([x["epoch"] for x in case_history], [x["train"]["total"] for x in case_history], label="train")
    axes[1].plot([x["epoch"] for x in case_history], [x["val"]["total"] for x in case_history], label="val")
    axes[1].set(title="Case multi-task loss", xlabel="epoch", yscale="log")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(EXP_DIR / "training_curves.png", dpi=160)
    plt.close(fig)

    curves = analysis["curves"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curves["frame"], curves["native_moving_displacement_rmse"], label="native")
    ax.plot(curves["frame"], curves["case_moving_displacement_rmse"], label="case")
    ax.set(title="Moving-node rollout error", xlabel="frame", ylabel="RMSE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EXP_DIR / "rollout_curves.png", dpi=160)
    plt.close(fig)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
