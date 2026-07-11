"""Autoregressive rollout utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from valgraphnet.config import get_cfg
from valgraphnet.data import ValveGraphDataset
from valgraphnet.gpu_graph import update_state
from valgraphnet.model import build_model
from valgraphnet.normalization import Normalizers, split_target
from valgraphnet.train import autocast_context, resolve_device


@torch.no_grad()
def run_rollout(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    case_dir: str | Path,
    out_dir: str | Path,
    steps: int | None = None,
) -> Path:
    """Run autoregressive rollout for one exported case."""

    device = resolve_device(str(get_cfg(cfg, "training.device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("cfg", cfg)
    output_dim = int(checkpoint["output_dim"])
    normalizers = None
    if checkpoint.get("normalizers") is not None:
        normalizers = Normalizers.from_state_dict(checkpoint["normalizers"]).to(device)

    dataset = ValveGraphDataset(data_root=case_dir, cfg=ckpt_cfg, normalizers=None)
    case = dataset.cases[0]
    inference_cfg = copy.deepcopy(ckpt_cfg)
    inference_cfg.setdefault("model", {})["num_processor_checkpoint_segments"] = 0
    model = build_model(inference_cfg, output_dim=output_dim).to(device)
    model.load_compatible_state_dict(checkpoint["model"])
    model.eval()

    n_steps = case.num_steps - 1 if steps is None else min(int(steps), case.num_steps - 1)
    case_tensors = dataset.gpu_builder.case_tensors(case, device)
    state = dataset.gpu_builder.state(case, 0, device)

    u_pred = [state["U"].clone()]
    v_pred = [state["V"].clone()]
    a_pred = [state["A"].clone()]
    stress_pred = []

    for step in range(n_steps):
        graph = dataset.gpu_builder.make_graph(case, step, device, state=state)
        if normalizers is not None:
            graph = normalizers.transform_data(graph)
        with autocast_context(ckpt_cfg, device):
            pred = model(graph)
        pred_concat = torch.cat([pred["delta_u"], pred["delta_v"], pred["accel"], pred["stress"]], dim=1)
        if normalizers is not None:
            pred_concat = normalizers.inverse_target(pred_concat)
        pred_phys = split_target(pred_concat)

        state = update_state(pred_phys, state, case_tensors, step + 1)
        u_pred.append(state["U"].clone())
        v_pred.append(state["V"].clone())
        a_pred.append(state["A"].clone())
        stress_pred.append(pred_phys["stress"].clone())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "U_pred.npy", torch.stack(u_pred).float().cpu().numpy())
    np.save(out / "V_pred.npy", torch.stack(v_pred).float().cpu().numpy())
    np.save(out / "A_pred.npy", torch.stack(a_pred).float().cpu().numpy())
    np.save(out / "S_pred.npy", torch.stack(stress_pred).float().cpu().numpy())
    np.save(out / "times.npy", case.times[: n_steps + 1])
    return out
