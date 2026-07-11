"""Standardized GPU rollout export for the existing ValGraphNet checkpoint."""

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
from valgraphnet.gpu_graph import update_state
from valgraphnet.model import build_model
from valgraphnet.normalization import Normalizers, split_target
from valgraphnet.physical_evaluation import evaluate_prediction_directory
from valgraphnet.train import autocast_context


@torch.no_grad()
def export_legacy_rollouts(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("legacy comparison rollout requires CUDA")
    device = torch.device("cuda")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    effective = deepcopy(checkpoint.get("cfg", cfg))
    effective["data"] = deepcopy(cfg.get("data", effective.get("data", {})))
    effective.setdefault("training", {})["device"] = "cuda"
    effective.setdefault("model", {})["num_processor_checkpoint_segments"] = 0
    root = get_cfg(effective, "data.root", get_cfg(effective, "data.case_dir", None))
    split_file = get_cfg(
        effective, "data.split_file", get_cfg(effective, "data.case_split_file", None)
    )
    selected_split = split or str(get_cfg(effective, "data.test_split", "test"))
    if root is None or split_file is None:
        raise ValueError("legacy export requires data root and split file")
    dataset = ValveGraphDataset(
        root, effective, split=selected_split, split_file=split_file
    )
    cases = dataset.cases[:max_cases] if max_cases is not None else dataset.cases
    output_dim = int(checkpoint["output_dim"])
    model = build_model(effective, output_dim=output_dim).to(device)
    model.load_compatible_state_dict(checkpoint["model"])
    model.eval()
    normalizers = None
    if checkpoint.get("normalizers") is not None:
        normalizers = Normalizers.from_state_dict(checkpoint["normalizers"]).to(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested_steps = get_cfg(cfg, "evaluation.steps", None)
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    manifest_cases = []
    for case in tqdm(cases, desc="ValGraphNet rollout"):
        tensors = dataset.gpu_builder.case_tensors(case, device)
        state = dataset.gpu_builder.state(case, 0, device)
        steps = case.num_steps - 1
        if requested_steps is not None:
            steps = min(steps, int(requested_steps))
        displacement = [state["U"].float().cpu().numpy()]
        stress = []
        diverged_at = None
        for step in range(steps):
            graph = dataset.gpu_builder.make_graph(case, step, device, state=state)
            if normalizers is not None:
                graph = normalizers.transform_data(graph)
            with autocast_context(effective, device):
                prediction = model(graph)
            concatenated = torch.cat(
                [
                    prediction["delta_u"],
                    prediction["delta_v"],
                    prediction["accel"],
                    prediction["stress"],
                ],
                dim=1,
            )
            if normalizers is not None:
                concatenated = normalizers.inverse_target(concatenated)
            physical = split_target(concatenated)
            state = update_state(physical, state, tensors, step + 1)
            if not bool(torch.isfinite(state["U"]).all().item()):
                diverged_at = step + 1
                break
            displacement.append(state["U"].float().cpu().numpy())
            stress.append(physical["stress"][:, :1].float().cpu().numpy())
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
        "model": "ValGraphNet legacy",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "device": torch.cuda.get_device_name(device),
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
    )
    return {"manifest": manifest, "metrics": metrics}
