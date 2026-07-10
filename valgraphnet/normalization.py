"""Feature normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Normalizers:
    """Standardize input features and scale targets without shifting zero."""

    node_mean: torch.Tensor
    node_std: torch.Tensor
    mesh_edge_mean: torch.Tensor
    mesh_edge_std: torch.Tensor
    world_edge_mean: torch.Tensor
    world_edge_std: torch.Tensor
    target_scale: torch.Tensor
    eps: float = 1.0e-8

    def transform_data(self, data):
        node_mean = self.node_mean.to(data.node_features.device)
        node_std = self.node_std.to(data.node_features.device)
        mesh_mean = self.mesh_edge_mean.to(data.mesh_edge_features.device)
        mesh_std = self.mesh_edge_std.to(data.mesh_edge_features.device)
        target_scale = self.target_scale.to(data.target.device)

        data.node_features = (data.node_features - node_mean) / node_std
        data.mesh_edge_features = (data.mesh_edge_features - mesh_mean) / mesh_std
        if data.world_edge_features.numel() > 0:
            world_mean = self.world_edge_mean.to(data.world_edge_features.device)
            world_std = self.world_edge_std.to(data.world_edge_features.device)
            data.world_edge_features = (
                data.world_edge_features - world_mean
            ) / world_std
        data.target = data.target / target_scale
        data.target_scale = target_scale
        return data

    def inverse_target(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.target_scale.to(target.device)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "node_mean": self.node_mean,
            "node_std": self.node_std,
            "mesh_edge_mean": self.mesh_edge_mean,
            "mesh_edge_std": self.mesh_edge_std,
            "world_edge_mean": self.world_edge_mean,
            "world_edge_std": self.world_edge_std,
            "target_scale": self.target_scale,
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "Normalizers":
        return cls(
            node_mean=state["node_mean"],
            node_std=state["node_std"],
            mesh_edge_mean=state["mesh_edge_mean"],
            mesh_edge_std=state["mesh_edge_std"],
            world_edge_mean=state["world_edge_mean"],
            world_edge_std=state["world_edge_std"],
            target_scale=state["target_scale"],
            eps=float(state.get("eps", 1.0e-8)),
        )

    def to(self, device: torch.device | str) -> "Normalizers":
        return Normalizers(
            node_mean=self.node_mean.to(device),
            node_std=self.node_std.to(device),
            mesh_edge_mean=self.mesh_edge_mean.to(device),
            mesh_edge_std=self.mesh_edge_std.to(device),
            world_edge_mean=self.world_edge_mean.to(device),
            world_edge_std=self.world_edge_std.to(device),
            target_scale=self.target_scale.to(device),
            eps=self.eps,
        )


def fit_normalizers(dataset, max_samples: int | None = None, eps: float = 1.0e-8) -> Normalizers:
    """Fit normalization statistics from raw dataset samples."""

    node_acc = _Moments()
    mesh_acc = _Moments()
    world_acc = _Moments()
    target_scale_acc = _SecondMoment()

    count = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    if count == len(dataset):
        indices = range(count)
    else:
        indices = torch.linspace(0, len(dataset) - 1, steps=count).round().long().tolist()
    for idx in indices:
        sample = dataset[idx]
        node_acc.update(sample.node_features)
        mesh_acc.update(sample.mesh_edge_features)
        if sample.world_edge_features.numel() > 0:
            world_acc.update(sample.world_edge_features)
        target_scale_acc.update(sample.target)

    node_mean, node_std = node_acc.finalize(eps)
    mesh_mean, mesh_std = mesh_acc.finalize(eps)
    if world_acc.count == 0:
        world_mean = torch.zeros_like(mesh_mean)
        world_std = torch.ones_like(mesh_std)
    else:
        world_mean, world_std = world_acc.finalize(eps)
    target_scale = target_scale_acc.finalize(eps)
    return Normalizers(
        node_mean=node_mean,
        node_std=node_std,
        mesh_edge_mean=mesh_mean,
        mesh_edge_std=mesh_std,
        world_edge_mean=world_mean,
        world_edge_std=world_std,
        target_scale=target_scale,
        eps=eps,
    )


class _Moments:
    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.sum_sq: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().float().cpu()
        if tensor.numel() == 0:
            return
        if self.sum is None:
            self.sum = tensor.sum(dim=0)
            self.sum_sq = (tensor * tensor).sum(dim=0)
        else:
            self.sum += tensor.sum(dim=0)
            self.sum_sq += (tensor * tensor).sum(dim=0)
        self.count += tensor.shape[0]

    def finalize(self, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0 or self.sum is None or self.sum_sq is None:
            raise ValueError("Cannot finalize empty moments")
        mean = self.sum / self.count
        var = torch.clamp(self.sum_sq / self.count - mean * mean, min=0.0)
        std = torch.sqrt(var + eps)
        return mean, std


class _SecondMoment:
    def __init__(self) -> None:
        self.count = 0
        self.sum_sq: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().float().cpu()
        if self.sum_sq is None:
            self.sum_sq = (tensor * tensor).sum(dim=0)
        else:
            self.sum_sq += (tensor * tensor).sum(dim=0)
        self.count += tensor.shape[0]

    def finalize(self, eps: float) -> torch.Tensor:
        if self.count == 0 or self.sum_sq is None:
            raise ValueError("Cannot finalize empty second moment")
        return torch.sqrt(self.sum_sq / self.count + eps)


def move_data_to_device(data, device: torch.device | str):
    """Move tensor attributes of a PyG Data object to a device."""

    return data.to(device)


def unnormalize_predictions(pred: dict[str, torch.Tensor], normalizers: Normalizers | None) -> dict[str, torch.Tensor]:
    """Convert normalized split predictions back to physical units."""

    if normalizers is None:
        return pred
    scale = normalizers.target_scale.to(next(iter(pred.values())).device)
    out = torch.cat([pred["delta_u"], pred["delta_v"], pred["accel"], pred["stress"]], dim=1)
    out = out * scale
    return split_target(out)


def split_target(target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split concatenated target/output tensor."""

    return {
        "delta_u": target[:, 0:3],
        "delta_v": target[:, 3:6],
        "accel": target[:, 6:9],
        "stress": target[:, 9:],
    }
