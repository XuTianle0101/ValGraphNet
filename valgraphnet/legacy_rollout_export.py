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

from valgraphnet.checkpoint_provenance import (
    build_repo_data_contract,
    compact_data_contract,
    config_sha256,
    sha256_file,
    strict_checkpoint_provenance,
    validate_repo_checkpoint,
)
from valgraphnet.config import get_cfg
from valgraphnet.data import ValveGraphDataset
from valgraphnet.data.case import read_split_file
from valgraphnet.gpu_graph import update_state
from valgraphnet.model import build_model
from valgraphnet.normalization import Normalizers, split_target
from valgraphnet.physical_evaluation import (
    evaluate_prediction_directory,
    select_case_ids,
)
from valgraphnet.train import autocast_context


@torch.no_grad()
def export_legacy_rollouts(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = None,
    max_cases: int | None = None,
    case_selection: str | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    strict = strict_checkpoint_provenance(cfg)
    if strict:
        if split is None or max_cases is None or case_selection is None:
            raise ValueError(
                "strict_v2 legacy export requires explicit --split, "
                "--max-cases, and --case-selection"
            )
        configured_selection = str(
            get_cfg(cfg, "evaluation.case_selection", "")
        )
        if case_selection != configured_selection:
            raise ValueError(
                "strict_v2 case selection differs from evaluation.case_selection"
            )
    data_contract = build_repo_data_contract(cfg) if strict else None
    validate_repo_checkpoint(
        checkpoint,
        cfg,
        data_contract,
        purpose="export",
        source=checkpoint_path,
    )
    effective = deepcopy(checkpoint.get("cfg", cfg))
    if not strict:
        # Historical checkpoints did not bind their data root.  Preserve their
        # previous relocation behavior outside formal strict_v2 experiments.
        effective["data"] = deepcopy(cfg.get("data", effective.get("data", {})))
    effective.setdefault("training", {})["device"] = "cuda"
    effective.setdefault("model", {})["num_processor_checkpoint_segments"] = 0
    root = get_cfg(effective, "data.root", get_cfg(effective, "data.case_dir", None))
    split_file = get_cfg(
        effective, "data.split_file", get_cfg(effective, "data.case_split_file", None)
    )
    selected_split = split or str(get_cfg(effective, "data.test_split", "test"))
    selection = case_selection or str(
        get_cfg(cfg, "evaluation.case_selection", "head")
    )
    if root is None or split_file is None:
        raise ValueError("legacy export requires data root and split file")
    selected_case_ids = _select_export_case_ids(
        split_file,
        selected_split,
        max_cases,
        selection,
    )
    if not selected_case_ids:
        raise ValueError("legacy export selected no cases")
    dataset = ValveGraphDataset(
        root,
        effective,
        case_ids=selected_case_ids,
    )
    cases = dataset.cases
    if not torch.cuda.is_available():
        raise RuntimeError("legacy comparison rollout requires CUDA")
    device = torch.device("cuda")
    output_dim = int(checkpoint["output_dim"])
    model = build_model(effective, output_dim=output_dim).to(device)
    if strict:
        model.load_state_dict(checkpoint["model"])
    else:
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
            if not bool(
                torch.isfinite(state["U"]).all().item()
                and torch.isfinite(physical["stress"]).all().item()
            ):
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
        "schema_version": 2,
        "model": "ValGraphNet legacy",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": config_sha256(cfg),
        "data_contract": compact_data_contract(data_contract),
        "split": selected_split,
        "max_cases": max_cases,
        "case_selection": selection,
        "selected_case_ids": selected_case_ids,
        "device": torch.cuda.get_device_name(device),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "seconds": time.perf_counter() - start_time,
        "cases": manifest_cases,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    metrics = _evaluate_exported_predictions(
        root,
        split_file,
        selected_split,
        output,
        output_path=output / "metrics.json",
        max_cases=max_cases,
        case_selection=selection,
    )
    return {"manifest": manifest, "metrics": metrics}


def _select_export_case_ids(
    split_file: str | Path,
    split: str,
    max_cases: int | None,
    case_selection: str,
) -> list[str]:
    return select_case_ids(
        read_split_file(split_file, split),
        max_cases,
        case_selection,
    )


def _evaluate_exported_predictions(
    root: str | Path,
    split_file: str | Path,
    split: str,
    output: str | Path,
    *,
    output_path: str | Path,
    max_cases: int | None,
    case_selection: str,
) -> dict[str, Any]:
    """Evaluate exactly the same case subset that was exported."""

    return evaluate_prediction_directory(
        root,
        split_file,
        split,
        output,
        output_path=output_path,
        max_cases=max_cases,
        case_selection=case_selection,
    )
