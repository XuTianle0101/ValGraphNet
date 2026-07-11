"""CUDA rollout export for schema-v2 CHP-GNS checkpoints."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from valgraphnet.chp_model import CHPGNS, CHPState
from valgraphnet.chp_train import (
    CHPCaseCache,
    _amp_dtype,
    _autocast,
    _require_cuda,
    _torch_load,
    contact_pairs_at,
    external_force_at,
    validate_chp_checkpoint_semantics,
)
from valgraphnet.config import get_cfg
from valgraphnet.data import ValveGraphDataset
from valgraphnet.physical_evaluation import evaluate_prediction_directory


@torch.no_grad()
def run_chp_rollouts(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Export complete GPU predictions and standardized physical metrics."""

    device = _require_cuda(cfg)
    checkpoint = _torch_load(checkpoint_path, device)
    validate_chp_checkpoint_semantics(
        checkpoint, source=checkpoint_path, require_scientific_gate=True
    )
    effective_cfg = deepcopy(checkpoint.get("config", cfg))
    effective_cfg["data"] = deepcopy(cfg.get("data", effective_cfg.get("data", {})))
    effective_cfg["training"] = {
        **effective_cfg.get("training", {}),
        **cfg.get("training", {}),
        "device": "cuda",
    }
    material_dim = int(checkpoint.get("material_dim", 0))
    model = CHPGNS(effective_cfg, material_dim=material_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    amp_dtype = _amp_dtype(effective_cfg)

    root = get_cfg(effective_cfg, "data.root", get_cfg(effective_cfg, "data.case_dir", None))
    split_file = get_cfg(
        effective_cfg,
        "data.split_file",
        get_cfg(effective_cfg, "data.case_split_file", None),
    )
    selected_split = split or str(get_cfg(effective_cfg, "data.test_split", "test"))
    if root is None or split_file is None:
        raise ValueError("CHP rollout requires data root and split file")
    dataset = ValveGraphDataset(
        root, effective_cfg, split=selected_split, split_file=split_file
    )
    cases = dataset.cases
    limit = max_cases
    if limit is None:
        limit = get_cfg(cfg, "evaluation.max_cases", None)
    if limit is not None:
        cases = cases[: max(int(limit), 0)]
    cache = CHPCaseCache(
        cases,
        device,
        material_dim=material_dim,
        cache_size=int(get_cfg(cfg, "evaluation.gpu_case_cache_size", 1)),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pressure_sign = float(get_cfg(effective_cfg, "data.pressure_sign", 1.0))
    requested_steps = get_cfg(cfg, "evaluation.steps", None)
    save_cell_stress = bool(get_cfg(cfg, "evaluation.save_cell_stress", False))
    divergence_position = float(get_cfg(cfg, "evaluation.divergence_position", 10.0))
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    manifest_cases: list[dict[str, Any]] = []

    for case_index, case in enumerate(tqdm(cases, desc="CHP rollout")):
        trajectory = cache.get(case_index)
        state = CHPState(
            trajectory.static.reference_position + trajectory.displacement[0],
            trajectory.velocity[0],
        )
        steps = case.num_steps - 1
        if requested_steps is not None:
            steps = min(steps, int(requested_steps))
        displacement = [trajectory.displacement[0].detach().cpu().numpy()]
        velocity = [trajectory.velocity[0].detach().cpu().numpy()]
        acceleration = [np.asarray(case.acceleration[0], dtype=np.float32)]
        stress: list[np.ndarray] = []
        cell_stress: list[np.ndarray] = []
        diagnostics: list[dict[str, float]] = []
        diverged_at: int | None = None
        for step in range(steps):
            dt = trajectory.times[step + 1] - trajectory.times[step]
            pairs = contact_pairs_at(trajectory, state, effective_cfg)
            with _autocast(device, amp_dtype):
                physical = model(
                    trajectory.static,
                    state,
                    contact_pairs=pairs,
                    external_force=external_force_at(
                        trajectory, step, pressure_sign
                    ),
                    dt=dt,
                    time_fraction=trajectory.time_fraction[step],
                    prescribed_position=(
                        trajectory.static.reference_position
                        + trajectory.displacement[step + 1]
                    ),
                    prescribed_velocity=trajectory.velocity[step + 1],
                )
            state = CHPState(physical.next_position, physical.next_velocity)
            is_finite = bool(
                (
                    torch.isfinite(state.position).all()
                    & torch.isfinite(state.velocity).all()
                    & (physical.energy_diagnostics["integration_valid"] >= 0.5)
                ).item()
            )
            if is_finite:
                is_finite = float(state.position.abs().max().item()) < divergence_position
            if not is_finite:
                diverged_at = step + 1
                break
            displacement.append(
                (state.position - trajectory.static.reference_position)
                .float()
                .cpu()
                .numpy()
            )
            velocity.append(state.velocity.float().cpu().numpy())
            acceleration.append(physical.acceleration.float().cpu().numpy())
            stress.append(physical.nodal_stress.float().cpu().numpy())
            if save_cell_stress:
                cell_stress.append(physical.cell_stress_tensor.float().cpu().numpy())
            diagnostics.append(
                {
                    key: float(value.detach().float().item())
                    for key, value in physical.energy_diagnostics.items()
                    if value.numel() == 1
                }
            )

        if diverged_at is not None:
            remaining = steps + 1 - len(displacement)
            displacement.extend(
                [np.full_like(displacement[0], np.nan)] * remaining
            )
            velocity.extend([np.full_like(velocity[0], np.nan)] * remaining)
            acceleration.extend(
                [np.full_like(acceleration[0], np.nan)] * remaining
            )
            stress.extend(
                [np.full((case.num_nodes, 1), np.nan, dtype=np.float32)]
                * (steps - len(stress))
            )
        case_output = output / case.case_id
        case_output.mkdir(parents=True, exist_ok=True)
        np.save(case_output / "U_pred.npy", np.asarray(displacement, dtype=np.float32))
        np.save(case_output / "V_pred.npy", np.asarray(velocity, dtype=np.float32))
        np.save(case_output / "A_pred.npy", np.asarray(acceleration, dtype=np.float32))
        np.save(case_output / "S_pred.npy", np.asarray(stress, dtype=np.float32))
        if save_cell_stress and cell_stress:
            np.save(
                case_output / "S_cell_pred.npy",
                np.asarray(cell_stress, dtype=np.float32),
            )
        with (case_output / "diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2, allow_nan=False)
        manifest_cases.append(
            {
                "case_id": case.case_id,
                "frames": len(displacement),
                "diverged_at": diverged_at,
            }
        )
        cache.clear_gpu()

    torch.cuda.synchronize(device)
    manifest = {
        "schema_version": 1,
        "model": "CHP-GNS",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "device": torch.cuda.get_device_name(device),
        "amp_dtype": str(amp_dtype).removeprefix("torch."),
        "seconds": time.perf_counter() - start_time,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
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
    )
    return {"manifest": manifest, "metrics": metrics}
