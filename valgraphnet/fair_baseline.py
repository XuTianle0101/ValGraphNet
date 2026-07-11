"""Fair four-output MeshGraphNet baseline and consistent state integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from valgraphnet.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


@dataclass
class IntegratedState:
    next_position: torch.Tensor
    next_velocity: torch.Tensor
    acceleration: torch.Tensor
    delta_u: torch.Tensor
    delta_v: torch.Tensor


def integrate_position_delta(
    current_position: torch.Tensor,
    current_velocity: torch.Tensor,
    delta_position: torch.Tensor,
    dt: torch.Tensor | float,
    *,
    fixed_mask: torch.Tensor,
    prescribed_mask: torch.Tensor | None = None,
    prescribed_position: torch.Tensor | None = None,
    prescribed_velocity: torch.Tensor | None = None,
) -> IntegratedState:
    """Derive every kinematic channel from one predicted position increment."""

    dt_tensor = torch.as_tensor(dt, dtype=current_position.dtype, device=current_position.device)
    dt_tensor = dt_tensor.clamp_min(torch.finfo(current_position.dtype).eps)
    fixed = fixed_mask[:, None]
    delta = torch.where(fixed, torch.zeros_like(delta_position), delta_position)
    next_position = current_position + delta
    next_velocity = delta / dt_tensor
    if prescribed_mask is not None and prescribed_position is not None:
        prescribed = prescribed_mask[:, None]
        next_position = torch.where(prescribed, prescribed_position, next_position)
        if prescribed_velocity is None:
            exact_velocity = (prescribed_position - current_position) / dt_tensor
        else:
            exact_velocity = prescribed_velocity
        next_velocity = torch.where(prescribed, exact_velocity, next_velocity)
    next_velocity = torch.where(fixed, torch.zeros_like(next_velocity), next_velocity)
    acceleration = (next_velocity - current_velocity) / dt_tensor
    acceleration = torch.where(fixed, torch.zeros_like(acceleration), acceleration)
    return IntegratedState(
        next_position=next_position,
        next_velocity=next_velocity,
        acceleration=acceleration,
        delta_u=next_position - current_position,
        delta_v=next_velocity - current_velocity,
    )


def add_position_noise(
    current_position: torch.Tensor,
    next_position: torch.Tensor,
    moving_mask: torch.Tensor,
    std: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply native-style state noise and return the corrected delta target."""

    noise = torch.randn(
        current_position.shape,
        dtype=current_position.dtype,
        device=current_position.device,
        generator=generator,
    ) * float(std)
    noise = noise * moving_mask[:, None].to(noise.dtype)
    noisy_position = current_position + noise
    corrected_delta = next_position - noisy_position
    return noisy_position, corrected_delta, noise


class FairDeformingPlateBaseline(nn.Module):
    """HybridMeshGraphNet that predicts only ``delta_x`` and transformed stress."""

    output_dim = 4

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        try:
            from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
        except ImportError as exc:
            raise ImportError("PhysicsNeMo is required for the fair baseline") from exc
        self.net = HybridMeshGraphNet(
            input_dim_nodes=NODE_FEATURE_DIM,
            input_dim_edges=EDGE_FEATURE_DIM,
            output_dim=self.output_dim,
            processor_size=int(model_cfg.get("processor_size", 15)),
            hidden_dim_processor=int(model_cfg.get("hidden_dim_processor", 128)),
            hidden_dim_node_decoder=int(model_cfg.get("hidden_dim_node_decoder", 128)),
            num_layers_node_processor=int(model_cfg.get("node_layers", 2)),
            num_layers_edge_processor=int(model_cfg.get("edge_layers", 2)),
            num_layers_node_decoder=int(model_cfg.get("decoder_layers", 2)),
            aggregation=str(model_cfg.get("aggregation", "sum")),
            mlp_activation_fn=str(model_cfg.get("activation", "relu")),
            num_processor_checkpoint_segments=int(
                model_cfg.get("num_processor_checkpoint_segments", 0)
            ),
        )

    def forward(self, data) -> dict[str, torch.Tensor]:
        output = self.net(
            data.node_features,
            data.mesh_edge_features,
            data.world_edge_features,
            data,
        )
        delta_x = output[:, :3]
        fixed = data.fixed_mask[:, None]
        return {
            "delta_x": torch.where(fixed, torch.zeros_like(delta_x), delta_x),
            "stress_transformed": output[:, 3:4],
        }
