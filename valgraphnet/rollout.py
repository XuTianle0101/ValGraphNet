"""Autoregressive rollout utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from valgraphnet.config import get_cfg
from valgraphnet.data import ValveGraphDataset
from valgraphnet.model import build_model
from valgraphnet.normalization import Normalizers, split_target
from valgraphnet.train import resolve_device


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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = checkpoint.get("cfg", cfg)
    output_dim = int(checkpoint["output_dim"])
    normalizers = None
    if checkpoint.get("normalizers") is not None:
        normalizers = Normalizers.from_state_dict(checkpoint["normalizers"]).to(device)

    dataset = ValveGraphDataset(data_root=case_dir, cfg=ckpt_cfg, normalizers=None)
    case = dataset.cases[0]
    model = build_model(ckpt_cfg, output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    n_steps = case.num_steps - 1 if steps is None else min(int(steps), case.num_steps - 1)
    u = case.displacement[0].copy()
    v = case.velocity[0].copy()
    a = case.acceleration[0].copy()

    u_pred = [u.copy()]
    v_pred = [v.copy()]
    a_pred = [a.copy()]
    stress_pred = []

    for step in range(n_steps):
        graph = dataset.make_graph(case, step, state={"U": u, "V": v, "A": a})
        if normalizers is not None:
            graph = normalizers.transform_data(graph)
        graph = graph.to(device)
        pred = model(graph)
        pred_concat = torch.cat([pred["delta_u"], pred["delta_v"], pred["accel"], pred["stress"]], dim=1)
        if normalizers is not None:
            pred_concat = normalizers.inverse_target(pred_concat)
        pred_phys = split_target(pred_concat)

        du = pred_phys["delta_u"].cpu().numpy()
        dv = pred_phys["delta_v"].cpu().numpy()
        next_a = pred_phys["accel"].cpu().numpy()
        du[case.fixed_mask, :] = 0.0
        dv[case.fixed_mask, :] = 0.0
        next_a[case.fixed_mask, :] = 0.0

        u = u + du
        v = v + dv
        a = next_a
        u_pred.append(u.copy())
        v_pred.append(v.copy())
        a_pred.append(a.copy())
        stress_pred.append(pred_phys["stress"].cpu().numpy())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "U_pred.npy", np.asarray(u_pred, dtype=np.float32))
    np.save(out / "V_pred.npy", np.asarray(v_pred, dtype=np.float32))
    np.save(out / "A_pred.npy", np.asarray(a_pred, dtype=np.float32))
    np.save(out / "S_pred.npy", np.asarray(stress_pred, dtype=np.float32))
    np.save(out / "times.npy", case.times[: n_steps + 1])
    return out
