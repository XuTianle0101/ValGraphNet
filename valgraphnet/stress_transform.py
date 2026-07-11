"""Robust, invertible transforms and losses for heavy-tailed stress fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class AsinhStressTransform:
    """Standardize ``asinh(stress / reference_scale)`` without clipping labels."""

    reference_scale: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1.0e-8

    def transform(self, stress: torch.Tensor) -> torch.Tensor:
        scale = self.reference_scale.to(stress.device, stress.dtype).clamp_min(self.eps)
        mean = self.mean.to(stress.device, stress.dtype)
        std = self.std.to(stress.device, stress.dtype).clamp_min(self.eps)
        return (torch.asinh(stress / scale) - mean) / std

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        scale = self.reference_scale.to(value.device, value.dtype).clamp_min(self.eps)
        mean = self.mean.to(value.device, value.dtype)
        std = self.std.to(value.device, value.dtype).clamp_min(self.eps)
        return torch.sinh(value * std + mean) * scale

    def to(self, device: torch.device | str) -> "AsinhStressTransform":
        return AsinhStressTransform(
            reference_scale=self.reference_scale.to(device),
            mean=self.mean.to(device),
            std=self.std.to(device),
            eps=self.eps,
        )

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "reference_scale": self.reference_scale,
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "AsinhStressTransform":
        return cls(
            reference_scale=state["reference_scale"],
            mean=state["mean"],
            std=state["std"],
            eps=float(state.get("eps", 1.0e-8)),
        )

    @classmethod
    def fit(
        cls,
        stress_tensors: Iterable[torch.Tensor],
        *,
        max_values: int = 1_000_000,
        eps: float = 1.0e-8,
    ) -> "AsinhStressTransform":
        """Fit deterministic robust statistics from a bounded CPU sample."""

        samples: list[torch.Tensor] = []
        remaining = int(max_values)
        for raw in stress_tensors:
            if remaining <= 0:
                break
            value = raw.detach().float().cpu().reshape(-1)
            if value.numel() > remaining:
                positions = torch.linspace(0, value.numel() - 1, remaining).round().long()
                value = value[positions]
            samples.append(value)
            remaining -= value.numel()
        if not samples:
            raise ValueError("Cannot fit a stress transform from no values")
        values = torch.cat(samples)
        positive = values.abs()[values.abs() > eps]
        reference = positive.median() if positive.numel() else torch.tensor(1.0)
        transformed = torch.asinh(values / reference.clamp_min(eps))
        mean = transformed.mean()
        std = transformed.std(unbiased=False).clamp_min(eps)
        return cls(reference.reshape(1), mean.reshape(1), std.reshape(1), eps=eps)


def robust_stress_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    ranking_target: torch.Tensor | None = None,
    peak_fraction: float = 0.1,
    peak_weight: float = 0.5,
    delta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Huber field loss plus an explicit high-stress-tail term."""

    base = F.huber_loss(prediction, target, reduction="mean", delta=float(delta))
    count = target.numel()
    if count == 0 or peak_fraction <= 0.0 or peak_weight <= 0.0:
        peak = base.new_zeros(())
    else:
        topk = max(1, min(count, int(round(count * float(peak_fraction)))))
        ranking = target if ranking_target is None else ranking_target
        if ranking.shape != target.shape:
            raise ValueError("ranking_target must have the same shape as target")
        indices = ranking.detach().abs().reshape(-1).topk(topk).indices
        pred_flat = prediction.reshape(-1)[indices]
        target_flat = target.reshape(-1)[indices]
        peak = F.huber_loss(pred_flat, target_flat, reduction="mean", delta=float(delta))
    total = base + float(peak_weight) * peak
    return total, {"stress_base": base.detach(), "stress_peak": peak.detach()}
