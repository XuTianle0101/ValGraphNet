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
        self.dynamics_dim = min(9, self.output_dim)
        self.stress_dim = max(0, self.output_dim - self.dynamics_dim)
        self.independent_stress_decoder = bool(
            model_cfg.get("independent_stress_decoder", False)
        ) and self.stress_dim > 0
        self.detach_stress_latent = bool(
            model_cfg.get("detach_stress_latent", False)
        )

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
            "output_dim": (
                self.dynamics_dim if self.independent_stress_decoder else self.output_dim
            ),
            "processor_size": int(model_cfg.get("processor_size", 15)),
            "mlp_activation_fn": str(model_cfg.get("activation", "relu")),
            "hidden_dim_processor": int(model_cfg.get("hidden_dim_processor", 128)),
            "hidden_dim_node_decoder": int(
                model_cfg.get("hidden_dim_node_decoder", 128)
            ),
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
        if self.independent_stress_decoder:
            from physicsnemo.nn import get_activation
            from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP

            decoder_hidden_dim = int(
                model_cfg.get(
                    "stress_decoder_hidden_dim",
                    model_cfg.get("hidden_dim_node_decoder", 128),
                )
            )
            decoder_layers = int(
                model_cfg.get(
                    "stress_decoder_layers", model_cfg.get("decoder_layers", 2)
                )
            )
            self.stress_decoder = MeshGraphMLP(
                int(model_cfg.get("hidden_dim_processor", 128)),
                output_dim=self.stress_dim,
                hidden_dim=decoder_hidden_dim,
                hidden_layers=decoder_layers,
                activation_fn=get_activation(str(model_cfg.get("activation", "relu"))),
                norm_type=None,
                recompute_activation=False,
            )

    def forward(self, data) -> dict[str, torch.Tensor]:
        if self.independent_stress_decoder:
            latent = self._encode_and_process(data)
            dynamics = self.net.node_decoder(latent)
            stress_input = latent.detach() if self.detach_stress_latent else latent
            stress = self.stress_decoder(stress_input)
            out = torch.cat([dynamics, stress], dim=1)
        elif self.model_type == "hybrid":
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

    def _encode_and_process(self, data) -> torch.Tensor:
        """Run the shared encoders and processor before task-specific decoding."""

        node_features = self.net.node_encoder(data.node_features)
        if self.model_type == "hybrid":
            mesh_features = self.net.mesh_edge_encoder(data.mesh_edge_features)
            world_features = self.net.world_edge_encoder(data.world_edge_features)
            return self.net.processor(
                node_features, mesh_features, world_features, data
            )

        edge_features = torch.cat(
            [data.mesh_edge_features, data.world_edge_features], dim=0
        )
        edge_features = self.net.edge_encoder(edge_features)
        return self.net.processor(node_features, edge_features, data)

    def load_compatible_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load this model or migrate a legacy joint 10-channel decoder."""

        if not self.independent_stress_decoder or any(
            key.startswith("stress_decoder.") for key in state
        ):
            self.load_state_dict(state)
            return

        migrated = self.state_dict()
        for key, value in state.items():
            if key in migrated and migrated[key].shape == value.shape:
                migrated[key] = value

        dynamics_prefix = "net.node_decoder.model."
        stress_prefix = "stress_decoder.model."
        dynamics_keys = sorted(
            key for key in migrated if key.startswith(dynamics_prefix)
        )
        stress_keys = sorted(key for key in migrated if key.startswith(stress_prefix))
        for key in dynamics_keys:
            source = state.get(key)
            if source is None or source.ndim == 0:
                continue
            target = migrated[key]
            if (
                source.ndim == target.ndim
                and source.shape[1:] == target.shape[1:]
                and source.shape[0] >= target.shape[0]
            ):
                migrated[key] = source[: target.shape[0]]
        for key in stress_keys:
            suffix = key[len(stress_prefix) :]
            source = state.get(dynamics_prefix + suffix)
            if source is None:
                continue
            target = migrated[key]
            if source.shape == target.shape:
                migrated[key] = source
            elif source.shape[0] >= self.dynamics_dim + target.shape[0]:
                migrated[key] = source[
                    self.dynamics_dim : self.dynamics_dim + target.shape[0]
                ]
        self.load_state_dict(migrated)

    def load_stress_decoder_state_dict(
        self, state: dict[str, torch.Tensor]
    ) -> None:
        """Load only the stress head from a split or legacy joint checkpoint."""

        if not self.independent_stress_decoder:
            raise ValueError("A separate stress decoder is not enabled")
        target = self.stress_decoder.state_dict()
        split_prefix = "stress_decoder."
        joint_prefix = "net.node_decoder."
        for key, value in list(target.items()):
            split_value = state.get(split_prefix + key)
            if split_value is not None and split_value.shape == value.shape:
                target[key] = split_value
                continue
            joint_value = state.get(joint_prefix + key)
            if joint_value is None:
                continue
            if joint_value.shape == value.shape:
                target[key] = joint_value
            elif (
                joint_value.ndim == value.ndim
                and joint_value.shape[1:] == value.shape[1:]
                and joint_value.shape[0] >= self.dynamics_dim + value.shape[0]
            ):
                target[key] = joint_value[
                    self.dynamics_dim : self.dynamics_dim + value.shape[0]
                ]
        self.stress_decoder.load_state_dict(target)


def build_model(cfg: dict[str, Any], output_dim: int) -> ValveGraphNet:
    """Factory used by train and rollout scripts."""

    return ValveGraphNet(cfg=cfg, output_dim=output_dim)

