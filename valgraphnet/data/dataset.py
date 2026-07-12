"""PyTorch dataset that converts exported cases to graph samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from valgraphnet.config import get_cfg
from valgraphnet.constants import BASE_OUTPUT_DIM
from valgraphnet.data.case import ValveCase, discover_case_dirs, load_case, read_split_file
from valgraphnet.data.valve_ood import (
    validate_case_collection_requirements,
    validate_case_requirements,
)
from valgraphnet.geometry import (
    ContactConfig,
    build_contact_edges,
    build_edge_features,
    build_node_features,
)
from valgraphnet.gpu_graph import GpuGraphBuilder


class ValveGraphDataset(Dataset):
    """One-step graph samples from exported valve simulation cases."""

    def __init__(
        self,
        data_root: str | Path,
        cfg: dict[str, Any] | None = None,
        split: str | None = None,
        split_file: str | Path | None = None,
        case_ids: list[str] | None = None,
        normalizers: Any | None = None,
    ) -> None:
        self.cfg = cfg or {}
        self.normalizers = normalizers
        if split_file and split and Path(split_file).exists():
            case_ids = read_split_file(split_file, split)

        case_dirs = discover_case_dirs(data_root, case_ids)
        if not case_dirs:
            raise FileNotFoundError(f"No exported cases found under {data_root}")

        self.cases = [load_case(path) for path in case_dirs]
        for case in self.cases:
            validate_case_requirements(case, self.cfg)
        validate_case_collection_requirements(self.cases, self.cfg)
        self.samples: list[tuple[int, int]] = []
        self.trajectory_index_groups: list[range] = []
        for case_idx, case in enumerate(self.cases):
            start = len(self.samples)
            for step in range(case.num_steps - 1):
                self.samples.append((case_idx, step))
            self.trajectory_index_groups.append(range(start, len(self.samples)))

        if not self.samples:
            raise ValueError("No one-step samples were found. Each case needs at least two frames.")

        self.stress_dim = max(case.stress_dim for case in self.cases)
        self.output_dim = BASE_OUTPUT_DIM + self.stress_dim
        self.gpu_builder = GpuGraphBuilder(self.cfg, self.stress_dim)
        self._device_normalizers: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        case_idx, step = self.samples[index]
        case = self.cases[case_idx]
        data = self.make_graph(case, step)
        if self.normalizers is not None:
            data = self.normalizers.transform_data(data)
        return data

    def make_graph(
        self,
        case: ValveCase,
        step: int,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Build a PyG Data object for one case and time step."""

        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError("torch-geometric is required to build graph samples") from exc

        u = case.displacement[step] if state is None else state["U"]
        v = case.velocity[step] if state is None else state["V"]
        a = case.acceleration[step] if state is None else state["A"]
        current_pos = case.nodes + u

        dt = float(case.times[step + 1] - case.times[step])
        if dt <= 0.0:
            raise ValueError(f"{case.root}: non-positive dt at step {step}")

        pressure_sign = float(get_cfg(self.cfg, "data.pressure_sign", 1.0))
        pressure_k = pressure_sign * float(case.pressure[step])
        pressure_next = pressure_sign * float(case.pressure[step + 1])
        pressure_rate = (pressure_next - pressure_k) / dt
        phase = _phase(case.times, step)

        node_features = build_node_features(
            nodes=case.nodes,
            displacement=u,
            velocity=v,
            acceleration=a,
            fixed_mask=case.fixed_mask,
            pressure_mask=case.pressure_mask,
            leaflet_id=case.leaflet_id,
            normals=case.normals,
            nodal_area=case.nodal_area,
            thickness=case.thickness,
            pressure_k=pressure_k,
            pressure_next=pressure_next,
            pressure_rate=pressure_rate,
            phase=phase,
            pressure_sign=1.0,
        )

        contact_cfg = ContactConfig(
            enabled=bool(get_cfg(self.cfg, "contact.enabled", True)),
            radius=get_cfg(self.cfg, "contact.radius", None),
            radius_factor=float(get_cfg(self.cfg, "contact.radius_factor", 2.5)),
            max_edges=int(get_cfg(self.cfg, "contact.max_edges", 200_000)),
            max_neighbors=get_cfg(self.cfg, "contact.max_neighbors", None),
            different_leaflets_only=bool(get_cfg(self.cfg, "contact.different_leaflets_only", True)),
        )
        world_edge_index = build_contact_edges(
            current_pos=current_pos,
            leaflet_id=case.leaflet_id,
            mesh_edge_index=case.mesh_edge_index,
            cfg=contact_cfg,
            fixed_mask=case.fixed_mask,
        )

        mesh_edge_features = build_edge_features(
            nodes=case.nodes,
            current_pos=current_pos,
            normals=case.normals,
            edge_index=case.mesh_edge_index,
            pressure_mask=case.pressure_mask,
            fixed_mask=case.fixed_mask,
            leaflet_id=case.leaflet_id,
            is_world_edge=False,
        )
        world_edge_features = build_edge_features(
            nodes=case.nodes,
            current_pos=current_pos,
            normals=case.normals,
            edge_index=world_edge_index,
            pressure_mask=case.pressure_mask,
            fixed_mask=case.fixed_mask,
            leaflet_id=case.leaflet_id,
            is_world_edge=True,
        )

        edge_index = np.concatenate([case.mesh_edge_index, world_edge_index], axis=1)
        target_delta_u = case.displacement[step + 1] - u
        target_delta_v = case.velocity[step + 1] - v
        target_accel = case.acceleration[step + 1]
        target_stress = _pad_stress(case.stress[step + 1], self.stress_dim)
        target = np.concatenate([target_delta_u, target_delta_v, target_accel, target_stress], axis=1)

        data = Data(edge_index=torch.as_tensor(edge_index, dtype=torch.long), num_nodes=case.num_nodes)
        data.pos = torch.as_tensor(current_pos, dtype=torch.float32)
        data.reference_pos = torch.from_numpy(
            np.array(case.nodes, dtype=np.float32, copy=True)
        )
        data.node_features = torch.as_tensor(node_features, dtype=torch.float32)
        data.mesh_edge_features = torch.as_tensor(mesh_edge_features, dtype=torch.float32)
        data.world_edge_features = torch.as_tensor(world_edge_features, dtype=torch.float32)
        data.target = torch.as_tensor(target, dtype=torch.float32)
        data.fixed_mask = torch.as_tensor(case.fixed_mask, dtype=torch.bool)
        data.prescribed_mask = torch.as_tensor(case.prescribed_mask, dtype=torch.bool)
        data.pressure_mask = torch.as_tensor(case.pressure_mask, dtype=torch.bool)
        data.nodal_area = torch.as_tensor(case.nodal_area, dtype=torch.float32)
        data.case_id = case.case_id
        data.case_root = str(case.root)
        data.step = int(step)
        data.dt = float(dt)
        data.mesh_edge_count = int(case.mesh_edge_index.shape[1])
        data.world_edge_count = int(world_edge_index.shape[1])
        return data

    def make_graph_gpu(
        self,
        case_index: int,
        step: int,
        device: torch.device,
        state: dict[str, torch.Tensor] | None = None,
    ):
        """Build and optionally normalize a graph without leaving CUDA."""

        if device.type != "cuda":
            raise ValueError("make_graph_gpu requires a CUDA device")
        data = self.gpu_builder.make_graph(
            self.cases[int(case_index)], int(step), device, state=state
        )
        if self.normalizers is not None:
            key = str(device)
            normalizers = self._device_normalizers.get(key)
            if normalizers is None:
                normalizers = self.normalizers.to(device)
                self._device_normalizers[key] = normalizers
            data = normalizers.transform_data(data)
        return data


def _phase(times: np.ndarray, step: int) -> float:
    start = float(times[0])
    stop = float(times[-1])
    span = max(stop - start, 1.0e-12)
    return (float(times[step]) - start) / span


def _pad_stress(stress: np.ndarray, stress_dim: int) -> np.ndarray:
    if stress_dim == 0:
        return np.zeros((stress.shape[0], 0), dtype=np.float32)
    out = np.zeros((stress.shape[0], stress_dim), dtype=np.float32)
    if stress.size:
        out[:, : stress.shape[1]] = stress.astype(np.float32)
    return out

