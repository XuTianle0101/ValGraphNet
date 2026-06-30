"""Loss functions for valve graph dynamics."""

from __future__ import annotations

from typing import Any

import torch

from valgraphnet.normalization import split_target


def valve_loss(
    pred: dict[str, torch.Tensor],
    data,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted one-step loss."""

    target = split_target(data.target)
    free_mask = (~data.fixed_mask).float()[:, None]
    weights = data.nodal_area.float()[:, None].clamp_min(1.0e-12)
    weights = weights / weights.mean().clamp_min(1.0e-12)

    loss_cfg = cfg.get("loss", {})
    components: dict[str, torch.Tensor] = {}
    components["delta_u"] = _weighted_mse(pred["delta_u"], target["delta_u"], weights, free_mask)
    components["delta_v"] = _weighted_mse(pred["delta_v"], target["delta_v"], weights, free_mask)
    components["accel"] = _weighted_mse(pred["accel"], target["accel"], weights, free_mask)
    if target["stress"].numel() > 0:
        components["stress"] = _weighted_mse(pred["stress"], target["stress"], weights, free_mask)
    else:
        components["stress"] = pred["delta_u"].new_tensor(0.0)

    total = pred["delta_u"].new_tensor(0.0)
    metrics: dict[str, float] = {}
    for name, value in components.items():
        factor = float(loss_cfg.get(name, 0.0))
        total = total + factor * value
        metrics[name] = float(value.detach().cpu())
    metrics["total"] = float(total.detach().cpu())
    return total, metrics


def _weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    residual2 = (pred - target) ** 2
    while weights.ndim < residual2.ndim:
        weights = weights.unsqueeze(-1)
    while mask.ndim < residual2.ndim:
        mask = mask.unsqueeze(-1)
    weighted = residual2 * weights * mask
    denom = torch.clamp((weights * mask).sum() * residual2.shape[-1], min=1.0)
    return weighted.sum() / denom

