from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "outputs" / "deforming_plate_full_experiment"
LOG_DIR = EXP_DIR / "logs"
NATIVE_CFG = ROOT / "examples" / "deforming_plate" / "config.full_case_backed.yaml"
CASE_CFG = ROOT / "configs" / "deforming_plate_case.full_gpu.yaml"
CASE_ROOT = ROOT / "data" / "deforming_plate_cases"
SPLIT_FILE = CASE_ROOT / "splits.json"
NATIVE_OUT = ROOT / "outputs" / "deforming_plate_full_case_backed"
CASE_OUT = ROOT / "outputs" / "deforming_plate_case_full_gpu"


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest()
    ensure_cases()
    run_native()
    run_native_rollout()
    run_case_path()
    run_case_rollouts()
    write_comparison()
    return 0


def ensure_cases() -> None:
    if SPLIT_FILE.exists():
        splits = read_json(SPLIT_FILE)
        if {key: len(splits.get(key, [])) for key in ("train", "val", "test")} == {
            "train": 1000,
            "val": 100,
            "test": 5,
        }:
            return
    run_step(
        "convert_to_cases",
        [
            sys.executable,
            "-m",
            "examples.deforming_plate.convert_to_cases",
            "--config",
            "examples/deforming_plate/config.yaml",
            "--out",
            "data/deforming_plate_cases",
        ],
    )


def run_native() -> None:
    if checkpoint_complete(NATIVE_OUT / "latest.pt", target_epoch=configured_epochs(NATIVE_CFG)):
        return
    run_step(
        "native_train",
        [
            sys.executable,
            "-m",
            "examples.deforming_plate.train",
            "--config",
            str(NATIVE_CFG.relative_to(ROOT)),
        ],
    )


def run_native_rollout() -> None:
    metrics = NATIVE_OUT / "rollout_eval" / "metrics.json"
    if metrics.exists():
        return
    run_step(
        "native_rollout_eval",
        [
            sys.executable,
            "-m",
            "examples.deforming_plate.rollout_eval",
            "--config",
            str(NATIVE_CFG.relative_to(ROOT)),
            "--checkpoint",
            str((NATIVE_OUT / "best.pt").relative_to(ROOT)),
        ],
    )


def run_case_path() -> None:
    if checkpoint_complete(CASE_OUT / "latest.pt", target_epoch=configured_epochs(CASE_CFG)):
        return
    run_step(
        "case_train",
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(CASE_CFG.relative_to(ROOT)),
        ],
    )


def run_case_rollouts() -> None:
    splits = read_json(SPLIT_FILE)
    for case_id in splits["test"]:
        out_dir = CASE_OUT / f"rollout_{case_id}"
        if (out_dir / "U_pred.npy").exists() and (out_dir / "S_pred.npy").exists():
            continue
        run_step(
            f"case_rollout_{case_id}",
            [
                sys.executable,
                "scripts/rollout.py",
                "--config",
                str(CASE_CFG.relative_to(ROOT)),
                "--checkpoint",
                str((CASE_OUT / "best.pt").relative_to(ROOT)),
                "--case",
                str((CASE_ROOT / case_id).relative_to(ROOT)),
                "--out",
                str(out_dir.relative_to(ROOT)),
            ],
        )


def write_comparison() -> None:
    native_metrics = read_json(NATIVE_OUT / "rollout_eval" / "metrics.json")
    case_metrics = []
    for case_id in read_json(SPLIT_FILE)["test"]:
        case_dir = CASE_ROOT / case_id
        rollout_dir = CASE_OUT / f"rollout_{case_id}"
        case_metrics.append(case_rollout_metrics(case_id, case_dir, rollout_dir))
    comparison = {
        "experiment": "deforming_plate_full",
        "generated_at": now_iso(),
        "native_config": str(NATIVE_CFG.relative_to(ROOT)),
        "case_config": str(CASE_CFG.relative_to(ROOT)),
        "native": native_metrics["summary"],
        "valgraphnet_case": aggregate(case_metrics),
        "valgraphnet_case_sequences": case_metrics,
        "training_history": {
            "native": read_json(NATIVE_OUT / "history.json"),
            "valgraphnet_case": read_json(CASE_OUT / "history.json"),
        },
    }
    write_json(EXP_DIR / "comparison.json", comparison)


def case_rollout_metrics(case_id: str, case_dir: Path, rollout_dir: Path) -> dict[str, float | str]:
    u_exact = np.load(case_dir / "U.npy", allow_pickle=False, mmap_mode="r")
    s_exact = np.load(case_dir / "S.npy", allow_pickle=False, mmap_mode="r")
    u_pred = np.load(rollout_dir / "U_pred.npy", allow_pickle=False)
    s_pred = np.load(rollout_dir / "S_pred.npy", allow_pickle=False)
    steps = min(u_pred.shape[0], u_exact.shape[0])
    stress_steps = min(s_pred.shape[0], max(s_exact.shape[0] - 1, 0))
    return {
        "case_id": case_id,
        "displacement_rmse": float(np.sqrt(np.mean((u_pred[:steps] - u_exact[:steps]) ** 2))),
        "rollout_rmse": float(np.sqrt(np.mean((u_pred[steps - 1] - u_exact[steps - 1]) ** 2))),
        "stress_rmse": (
            float(np.sqrt(np.mean((s_pred[:stress_steps] - s_exact[1 : 1 + stress_steps]) ** 2)))
            if stress_steps
            else 0.0
        ),
    }


def aggregate(metrics: list[dict[str, float | str]]) -> dict[str, float]:
    keys = ["displacement_rmse", "rollout_rmse", "stress_rmse"]
    return {key: float(np.mean([float(item[key]) for item in metrics])) for key in keys}


def checkpoint_complete(path: Path, target_epoch: int) -> bool:
    if not path.exists():
        return False
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return int(checkpoint.get("epoch", 0)) >= int(target_epoch)
    except Exception:
        return False


def configured_epochs(path: Path) -> int:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg["training"]["epochs"])


def run_step(name: str, command: list[str]) -> None:
    log_path = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now_iso()}] START {' '.join(command)}\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"[{now_iso()}] EXIT {proc.returncode}\n")
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed; see {log_path}")


def write_manifest() -> None:
    manifest = {
        "generated_at": now_iso(),
        "root": str(ROOT),
        "python": sys.version,
        "commands": {
            "native_train": f"{sys.executable} -m examples.deforming_plate.train --config {NATIVE_CFG.relative_to(ROOT)}",
            "native_rollout": f"{sys.executable} -m examples.deforming_plate.rollout_eval --config {NATIVE_CFG.relative_to(ROOT)} --checkpoint {NATIVE_OUT.relative_to(ROOT) / 'best.pt'}",
            "case_train": f"{sys.executable} scripts/train.py --config {CASE_CFG.relative_to(ROOT)}",
            "case_rollout": f"{sys.executable} scripts/rollout.py --config {CASE_CFG.relative_to(ROOT)} --checkpoint {CASE_OUT.relative_to(ROOT) / 'best.pt'} --case data/deforming_plate_cases/<case_id> --out {CASE_OUT.relative_to(ROOT)}/rollout_<case_id>",
        },
        "git": {
            "rev": capture(["git", "rev-parse", "HEAD"]),
            "status": capture(["git", "status", "--short"]),
        },
        "cuda": capture(["nvidia-smi"]),
        "dataset": dataset_manifest(),
    }
    write_json(EXP_DIR / "manifest.json", manifest)


def dataset_manifest() -> dict[str, Any]:
    raw_dir = ROOT / "raw_dataset" / "deforming_plate" / "deforming_plate"
    files = {}
    for name in ["meta.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"]:
        path = raw_dir / name
        files[name] = path.stat().st_size if path.exists() else None
    splits = read_json(SPLIT_FILE) if SPLIT_FILE.exists() else {}
    return {
        "raw_files": files,
        "case_splits": {key: len(value) for key, value in splits.items()},
    }


def capture(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).stdout.strip()
    except FileNotFoundError as exc:
        return str(exc)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
