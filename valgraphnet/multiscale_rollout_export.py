"""Standardized CUDA rollout export for the two-level MultiScale MGN."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from valgraphnet.config import get_cfg
from valgraphnet.data import ValveGraphDataset
from valgraphnet.data.case import read_split_file
from valgraphnet.fair_baseline import integrate_position_delta
from valgraphnet.fair_train import require_cuda_bf16
from valgraphnet.multiscale_train import load_multiscale_checkpoint
from valgraphnet.physical_evaluation import (
    evaluate_prediction_directory,
    select_case_ids,
)


@torch.inference_mode()
def export_multiscale_rollouts(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = None,
    max_cases: int | None = None,
    case_selection: str = "head",
) -> dict[str, Any]:
    """Export frozen trajectories; this function never selects checkpoints."""

    device = require_cuda_bf16(cfg)
    model, normalizers, stress_transform, checkpoint = load_multiscale_checkpoint(
        cfg, checkpoint_path
    )
    effective = deepcopy(checkpoint.get("cfg", cfg))
    effective["data"] = deepcopy(cfg.get("data", effective.get("data", {})))
    effective.setdefault("training", {})["device"] = "cuda"
    root = get_cfg(effective, "data.root", None)
    split_file = get_cfg(effective, "data.split_file", None)
    selected_split = split or str(get_cfg(effective, "data.test_split", "test"))
    if root is None or split_file is None:
        raise ValueError("MultiScale rollout export requires data root and split file")
    case_ids = select_case_ids(
        read_split_file(split_file, selected_split), max_cases, case_selection
    )
    dataset = ValveGraphDataset(
        root,
        effective,
        case_ids=case_ids,
        normalizers=normalizers,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    delta_scale = normalizers.target_scale[:3].to(device)
    stress_transform = stress_transform.to(device)
    requested_steps = get_cfg(cfg, "evaluation.steps", None)
    divergence_position = float(
        get_cfg(cfg, "evaluation.divergence_position", 10.0)
    )
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    manifest_cases = []

    for case_index, case in enumerate(
        tqdm(dataset.cases, desc="two-level MultiScale MGN rollout")
    ):
        tensors = dataset.gpu_builder.case_tensors(case, device)
        state = dataset.gpu_builder.state(case, 0, device)
        steps = case.num_steps - 1
        if requested_steps is not None:
            steps = min(steps, int(requested_steps))
        displacement = [state["U"].float().cpu().numpy()]
        stress = []
        diverged_at = None
        for step in range(steps):
            graph = dataset.make_graph_gpu(case_index, step, device, state=state)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(graph)
            delta_position = prediction["delta_x"].float() * delta_scale
            physical_stress = stress_transform.inverse(
                prediction["stress_transformed"].float()
            )
            integrated = integrate_position_delta(
                tensors["nodes"] + state["U"],
                state["V"],
                delta_position,
                tensors["times"][step + 1] - tensors["times"][step],
                fixed_mask=tensors["fixed"],
                prescribed_mask=tensors["prescribed"],
                prescribed_position=tensors["nodes"] + tensors["U"][step + 1],
                prescribed_velocity=tensors["V"][step + 1],
            )
            state = {
                "U": integrated.next_position - tensors["nodes"],
                "V": integrated.next_velocity,
                "A": integrated.acceleration,
            }
            finite = all(
                bool(torch.isfinite(value).all().item()) for value in state.values()
            )
            finite = finite and bool(torch.isfinite(physical_stress).all().item())
            finite = (
                finite
                and float(state["U"].abs().max().item()) < divergence_position
            )
            if not finite:
                diverged_at = step + 1
                break
            displacement.append(state["U"].float().cpu().numpy())
            stress.append(physical_stress[:, :1].float().cpu().numpy())
        if diverged_at is not None:
            displacement.extend(
                [np.full_like(displacement[0], np.nan)]
                * (steps + 1 - len(displacement))
            )
            stress.extend(
                [np.full((case.num_nodes, 1), np.nan, dtype=np.float32)]
                * (steps - len(stress))
            )
        case_output = output / case.case_id
        case_output.mkdir(parents=True, exist_ok=True)
        np.save(case_output / "U_pred.npy", np.asarray(displacement, dtype=np.float32))
        np.save(case_output / "S_pred.npy", np.asarray(stress, dtype=np.float32))
        manifest_cases.append(
            {
                "case_id": case.case_id,
                "frames": len(displacement),
                "diverged_at": diverged_at,
            }
        )
    torch.cuda.synchronize(device)
    manifest = {
        "schema_version": 1,
        "model": "two-level topology-bi-stride MultiScale MGN",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_split": checkpoint.get("checkpoint_split"),
        "evaluation_split": selected_split,
        "case_selection": case_selection,
        "device": torch.cuda.get_device_name(device),
        "precision": "bfloat16",
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "seconds": time.perf_counter() - start_time,
        "cases": manifest_cases,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    metrics = evaluate_prediction_directory(
        root,
        split_file,
        selected_split,
        output,
        output_path=output / "metrics.json",
        max_cases=max_cases,
        case_selection=case_selection,
    )
    return {"manifest": manifest, "metrics": metrics}
