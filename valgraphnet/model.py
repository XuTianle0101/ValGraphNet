"""Model wrappers around PhysicsNeMo MeshGraphNet modules."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from valgraphnet.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from valgraphnet.normalization import split_target


class ValveGraphNet(nn.Module):
    """Valve dynamics wrapper for PhysicsNeMo MeshGraphNet variants."""

    def __init__(self, cfg: dict[str, Any], output_dim: int) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        model_type = str(model_cfg.get("type", "hybrid")).lower()
        self.model_type = model_type
        self.output_dim = int(output_dim)

        try:
            if model_type == "hybrid":
                from physicsnemo.models.meshgraphnet import HybridMeshGraphNet as Net
            elif model_type == "mesh":
                from physicsnemo.models.meshgraphnet import MeshGraphNet as Net
            else:
                raise ValueError(f"Unsupported model.type: {model_type}")
        except ImportError as exc:
            raise ImportError(
                "PhysicsNeMo is required for ValveGraphNet. Install NVIDIA PhysicsNeMo "
                "in the training environment."
            ) from exc

        kwargs = {
            "input_dim_nodes": NODE_FEATURE_DIM,
            "input_dim_edges": EDGE_FEATURE_DIM,
            "output_dim": self.output_dim,
            "processor_size": int(model_cfg.get("processor_size", 15)),
            "mlp_activation_fn": str(model_cfg.get("activation", "relu")),
            "hidden_dim_processor": int(model_cfg.get("hidden_dim_processor", 128)),
            "num_layers_node_processor": int(model_cfg.get("node_layers", 2)),
            "num_layers_edge_processor": int(model_cfg.get("edge_layers", 2)),
            "num_layers_node_decoder": int(model_cfg.get("decoder_layers", 2)),
            "aggregation": str(model_cfg.get("aggregation", "sum")),
            "do_concat_trick": bool(model_cfg.get("do_concat_trick", False)),
            "num_processor_checkpoint_segments": int(
                model_cfg.get("num_processor_checkpoint_segments", 0)
            ),
            "checkpoint_offloading": bool(model_cfg.get("checkpoint_offloading", False)),
            "recompute_activation": bool(model_cfg.get("recompute_activation", False)),
        }
        self.net = Net(**kwargs)

    def forward(self, data) -> dict[str, torch.Tensor]:
        if self.model_type == "hybrid":
            out = self.net(
                data.node_features,
                data.mesh_edge_features,
                data.world_edge_features,
                data,
            )
        else:
            edge_features = torch.cat([data.mesh_edge_features, data.world_edge_features], dim=0)
            out = self.net(data.node_features, edge_features, data)

        pred = split_target(out)
        fixed = data.fixed_mask[:, None]
        pred["delta_u"] = torch.where(fixed, torch.zeros_like(pred["delta_u"]), pred["delta_u"])
        pred["delta_v"] = torch.where(fixed, torch.zeros_like(pred["delta_v"]), pred["delta_v"])
        pred["accel"] = torch.where(fixed, torch.zeros_like(pred["accel"]), pred["accel"])
        return pred


def build_model(cfg: dict[str, Any], output_dim: int) -> ValveGraphNet:
    """Factory used by train and rollout scripts."""

    return ValveGraphNet(cfg=cfg, output_dim=output_dim)

