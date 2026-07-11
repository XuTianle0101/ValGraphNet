"""DeepMind deforming-plate TFRecord loading and graph sample construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F


NODE_TYPE_MOVING = 0
NODE_TYPE_OBJECT = 1
NODE_TYPE_CLAMPED = 3


@dataclass
class DeformingPlateSequence:
    """One deforming-plate trajectory decoded from TFRecord."""

    sample_id: str
    mesh_pos: np.ndarray
    world_pos: np.ndarray
    cells: np.ndarray
    node_type: np.ndarray
    stress: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.world_pos.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.world_pos.shape[1])


@dataclass
class DeformingPlateGraphSample:
    """One time-step graph sample for the native deforming-plate example."""

    graph: Any
    mesh_edge_features: torch.Tensor
    world_edge_features: torch.Tensor


class SequenceDataset:
    """Small iterable wrapper over DeepMind deforming-plate TFRecords."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        num_samples: int | None = None,
        num_steps: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = str(split)
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.meta = _load_meta(self.data_dir)
        self.path = self.data_dir / f"{self.split}.tfrecord"
        if not self.path.exists():
            raise FileNotFoundError(f"Missing TFRecord file: {self.path}")

    def __iter__(self) -> Iterator[DeformingPlateSequence]:
        for idx, record in enumerate(_iter_tfrecord_records(self.path, self.meta)):
            if self.num_samples is not None and idx >= int(self.num_samples):
                break
            yield _sequence_from_record(
                record,
                sample_id=f"{self.split}_{idx:05d}",
                num_steps=self.num_steps,
            )


def make_graph_sample(
    sequence: DeformingPlateSequence,
    step: int,
    edge_stats: dict[str, torch.Tensor] | None = None,
    node_stats: dict[str, torch.Tensor] | None = None,
    world_pos_override: torch.Tensor | None = None,
    world_edge_radius: float = 0.03,
    max_world_neighbors: int | None = None,
    add_noise: bool = False,
    noise_std: float = 0.003,
) -> DeformingPlateGraphSample:
    """Build one PyG graph sample with mesh and world-edge feature tensors."""

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("torch-geometric is required for deforming_plate samples") from exc

    if step < 0 or step >= sequence.num_steps - 1:
        raise IndexError(f"step must be in [0, {sequence.num_steps - 2}], got {step}")

    mesh_pos = torch.as_tensor(sequence.mesh_pos, dtype=torch.float32)
    world_pos = torch.as_tensor(sequence.world_pos[step], dtype=torch.float32)
    next_world_pos = torch.as_tensor(sequence.world_pos[step + 1], dtype=torch.float32)
    if world_pos_override is not None:
        world_pos = world_pos_override.detach().float().cpu()

    node_type = torch.as_tensor(sequence.node_type, dtype=torch.long).view(-1)
    target_delta = next_world_pos - world_pos
    target_stress = torch.as_tensor(sequence.stress[step + 1], dtype=torch.float32)
    if target_stress.ndim == 1:
        target_stress = target_stress[:, None]

    if add_noise:
        noise_mask = node_type == NODE_TYPE_MOVING
        noise = torch.normal(mean=0.0, std=float(noise_std), size=world_pos.shape)
        noise = noise * noise_mask[:, None].float()
        world_pos = world_pos + noise
        target_delta = target_delta - noise

    edge_index = cells_to_edges(sequence.cells)
    mesh_ref_features = edge_features(edge_index, mesh_pos)
    mesh_world_features = edge_features(edge_index, world_pos)
    mesh_edge_features = torch.cat([mesh_ref_features, mesh_world_features], dim=1)

    world_edge_index = radius_world_edges(
        world_pos=world_pos,
        radius=float(world_edge_radius),
        mesh_edge_index=edge_index,
        max_neighbors=max_world_neighbors,
    )
    world_base_features = edge_features(world_edge_index, world_pos)
    world_edge_features = torch.cat([world_base_features, world_base_features], dim=1)
    combined_edge_index = torch.cat([edge_index, world_edge_index], dim=1)

    if edge_stats is not None:
        mesh_edge_features = normalize(
            mesh_edge_features,
            edge_stats["edge_mean"],
            edge_stats["edge_std"],
        )
        if world_edge_features.numel() > 0:
            world_edge_features = normalize(
                world_edge_features,
                edge_stats["edge_mean"],
                edge_stats["edge_std"],
            )
    target = torch.cat([target_delta, target_stress[:, :1]], dim=1)
    if node_stats is not None:
        target = torch.cat(
            [
                normalize(target[:, 0:3], node_stats["velocity_mean"], node_stats["velocity_std"]),
                normalize(target[:, 3:4], node_stats["stress_mean"], node_stats["stress_std"]),
            ],
            dim=1,
        )

    graph = Data(edge_index=combined_edge_index.long(), num_nodes=sequence.num_nodes)
    graph.x = one_hot_node_type(node_type).float()
    graph.y = target.float()
    graph.world_pos = world_pos.float()
    graph.next_world_pos = next_world_pos.float()
    graph.mesh_pos = mesh_pos.float()
    graph.cells = torch.as_tensor(sequence.cells, dtype=torch.long)
    graph.node_type = node_type.long()
    graph.sample_id = sequence.sample_id
    graph.step = int(step)
    graph.mesh_edge_count = int(edge_index.shape[1])
    graph.world_edge_count = int(world_edge_index.shape[1])
    return DeformingPlateGraphSample(
        graph=graph,
        mesh_edge_features=mesh_edge_features.float(),
        world_edge_features=world_edge_features.float(),
    )


def cells_to_edges(cells: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Create directed, coalesced tetrahedral mesh edges from cells."""

    try:
        import torch_geometric as pyg
    except ImportError as exc:
        raise ImportError("torch-geometric is required to coalesce mesh edges") from exc

    cells_t = torch.as_tensor(cells, dtype=torch.long)
    if cells_t.ndim != 2 or cells_t.shape[1] < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    undirected = []
    for i in range(cells_t.shape[1]):
        for j in range(i + 1, cells_t.shape[1]):
            undirected.append(torch.stack([cells_t[:, i], cells_t[:, j]], dim=0))
    edges = torch.cat(undirected, dim=1) if undirected else torch.zeros((2, 0), dtype=torch.long)
    edges = pyg.utils.to_undirected(edges)
    edges = pyg.utils.coalesce(edges)
    if isinstance(edges, tuple):
        edges = edges[0]
    return edges.long()


def edge_features(edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Return displacement and distance features for edges."""

    if edge_index.numel() == 0:
        return pos.new_zeros((0, pos.shape[1] + 1))
    src, dst = edge_index.long()
    disp = pos[src] - pos[dst]
    dist = torch.linalg.norm(disp, dim=-1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


def radius_world_edges(
    world_pos: torch.Tensor,
    radius: float,
    mesh_edge_index: torch.Tensor,
    max_neighbors: int | None = None,
) -> torch.Tensor:
    """Build directed world edges within a radius, excluding self and mesh edges.

    When ``max_neighbors`` is positive, retain the nearest world-space neighbors
    for every source node. Dense contact frames can otherwise approach a fully
    connected graph and exhaust GPU memory even with a batch size of one.
    """

    if radius <= 0.0 or world_pos.shape[0] < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    if max_neighbors is not None and int(max_neighbors) > 0:
        nearest = _radius_nearest_neighbors(
            world_pos,
            radius=float(radius),
            mesh_edge_index=mesh_edge_index,
            max_neighbors=int(max_neighbors),
        )
        if nearest is not None:
            return nearest
    pairs = _radius_pairs(world_pos, radius)
    mesh_edges = {tuple(edge) for edge in mesh_edge_index.t().tolist()}
    directed: list[tuple[int, int]] = []
    for i, j in pairs:
        if (i, j) not in mesh_edges:
            directed.append((i, j))
        if (j, i) not in mesh_edges:
            directed.append((j, i))
    if not directed:
        return torch.zeros((2, 0), dtype=torch.long)
    if max_neighbors is not None and int(max_neighbors) > 0:
        directed = _nearest_neighbors(
            directed,
            world_pos=world_pos,
            max_neighbors=int(max_neighbors),
        )
    return torch.as_tensor(directed, dtype=torch.long).t().contiguous()


def one_hot_node_type(node_type: torch.Tensor) -> torch.Tensor:
    """Map DeepMind node types {0, 1, 3} to three one-hot channels."""

    node_type = node_type.view(-1).long()
    mapped = torch.full_like(node_type, fill_value=-1)
    mapped[node_type == NODE_TYPE_MOVING] = 0
    mapped[node_type == NODE_TYPE_OBJECT] = 1
    mapped[node_type == NODE_TYPE_CLAMPED] = 2
    if (mapped < 0).any():
        bad = torch.unique(node_type[mapped < 0]).tolist()
        raise ValueError(f"Unsupported node_type values: {bad}")
    return F.one_hot(mapped, num_classes=3)


def rollout_masks(node_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return moving, object, and clamped masks with shape [N, 1]."""

    node_type = node_type.view(-1).long()
    moving = (node_type == NODE_TYPE_MOVING)[:, None]
    obj = (node_type == NODE_TYPE_OBJECT)[:, None]
    clamped = (node_type == NODE_TYPE_CLAMPED)[:, None]
    return moving, obj, clamped


def fit_stats(
    sequences: list[DeformingPlateSequence],
    world_edge_radius: float,
    max_world_neighbors: int | None = None,
    max_steps: int | None = None,
    eps: float = 1.0e-8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Fit edge and target normalization stats from raw sequences."""

    edge_moments = _Moments()
    vel_moments = _Moments()
    stress_moments = _Moments()
    for sequence in sequences:
        steps = (
            sequence.num_steps - 1
            if max_steps is None
            else min(sequence.num_steps - 1, int(max_steps))
        )
        for step in range(steps):
            sample = make_graph_sample(
                sequence,
                step,
                edge_stats=None,
                node_stats=None,
                world_edge_radius=world_edge_radius,
                max_world_neighbors=max_world_neighbors,
            )
            edge_moments.update(sample.mesh_edge_features)
            if sample.world_edge_features.numel() > 0:
                edge_moments.update(sample.world_edge_features)
            vel_moments.update(sample.graph.y[:, 0:3])
            stress_moments.update(sample.graph.y[:, 3:4])
    edge_mean, edge_std = edge_moments.finalize(eps)
    vel_mean, vel_std = vel_moments.finalize(eps)
    stress_mean, stress_std = stress_moments.finalize(eps)
    return (
        {"edge_mean": edge_mean, "edge_std": edge_std},
        {
            "velocity_mean": vel_mean,
            "velocity_std": vel_std,
            "stress_mean": stress_mean,
            "stress_std": stress_std,
        },
    )


def normalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor with broadcastable stats."""

    return (tensor - mean.to(tensor.device)) / std.to(tensor.device).clamp_min(1.0e-12)


def denormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert normalization."""

    return tensor * std.to(tensor.device) + mean.to(tensor.device)


def save_stats(path: str | Path, stats: dict[str, torch.Tensor]) -> None:
    """Save stats as a torch file."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in stats.items()}, path)


def load_stats(path: str | Path) -> dict[str, torch.Tensor]:
    """Load stats saved by save_stats."""

    return torch.load(path, map_location="cpu")


def load_sequences(
    data_dir: str | Path,
    split: str,
    num_samples: int | None,
    num_steps: int | None,
) -> list[DeformingPlateSequence]:
    """Load TFRecord sequences into memory."""

    return list(SequenceDataset(data_dir, split, num_samples=num_samples, num_steps=num_steps))


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing deforming_plate meta.json: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_tfrecord_records(path: Path, meta: dict[str, Any]) -> Iterator[dict[str, np.ndarray]]:
    try:
        from tfrecord.torch.dataset import TFRecordDataset
    except ImportError as exc:
        raise ImportError("Install the 'tfrecord' package to read deforming_plate data") from exc

    description = {key: "byte" for key in meta["field_names"]}
    dataset = TFRecordDataset(
        str(path),
        None,
        description,
        transform=lambda rec: _decode_record(rec, meta),
    )
    yield from dataset


def _decode_record(record: dict[str, bytes], meta: dict[str, Any]) -> dict[str, np.ndarray]:
    decoded = {}
    for key, value in record.items():
        feature = meta["features"][key]
        dtype = getattr(np, feature["dtype"])
        shape = tuple(feature["shape"])
        decoded[key] = np.frombuffer(value, dtype=dtype).reshape(shape)
    return decoded


def _sequence_from_record(
    record: dict[str, np.ndarray],
    sample_id: str,
    num_steps: int | None,
) -> DeformingPlateSequence:
    stop = None if num_steps is None else int(num_steps)
    node_type = record["node_type"][0] if record["node_type"].ndim == 3 else record["node_type"]
    return DeformingPlateSequence(
        sample_id=sample_id,
        mesh_pos=record["mesh_pos"][0].astype(np.float32),
        world_pos=record["world_pos"][:stop].astype(np.float32),
        cells=record["cells"][0].astype(np.int64),
        node_type=node_type.astype(np.int64).reshape(-1),
        stress=record["stress"][:stop].astype(np.float32),
    )


def _radius_pairs(points: torch.Tensor, radius: float) -> list[tuple[int, int]]:
    points_np = points.detach().cpu().numpy()
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points_np)
        return [(int(i), int(j)) for i, j in tree.query_pairs(radius)]
    except Exception:
        dist = torch.cdist(points.float(), points.float())
        src, dst = torch.where((dist <= float(radius)) & (dist > 0.0))
        return [(int(i), int(j)) for i, j in zip(src.tolist(), dst.tolist(), strict=False) if i < j]


def _nearest_neighbors(
    directed: list[tuple[int, int]],
    world_pos: torch.Tensor,
    max_neighbors: int,
) -> list[tuple[int, int]]:
    """Return a deterministic per-source nearest-neighbor subset."""

    edges = np.asarray(directed, dtype=np.int64)
    points = world_pos.detach().cpu().numpy()
    delta = points[edges[:, 0]] - points[edges[:, 1]]
    distance_sq = np.einsum("ij,ij->i", delta, delta)
    order = np.lexsort((edges[:, 1], distance_sq, edges[:, 0]))
    sorted_edges = edges[order]
    source = sorted_edges[:, 0]
    group_start = np.r_[True, source[1:] != source[:-1]]
    starts = np.maximum.accumulate(np.where(group_start, np.arange(source.size), 0))
    keep = np.arange(source.size) - starts < int(max_neighbors)
    return [tuple(edge) for edge in sorted_edges[keep].tolist()]


def _radius_nearest_neighbors(
    world_pos: torch.Tensor,
    radius: float,
    mesh_edge_index: torch.Tensor,
    max_neighbors: int,
) -> torch.Tensor | None:
    """Query bounded radius neighbors directly, avoiding dense candidate pairs."""

    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None

    points = world_pos.detach().cpu().numpy()
    num_nodes = int(points.shape[0])
    mesh_edges = {tuple(edge) for edge in mesh_edge_index.t().tolist()}
    mesh_degree = torch.bincount(mesh_edge_index[0], minlength=num_nodes)
    max_mesh_degree = int(mesh_degree.max()) if mesh_degree.numel() else 0
    query_k = min(num_nodes, 1 + int(max_neighbors) + max_mesh_degree)
    distances, neighbors = cKDTree(points).query(
        points,
        k=query_k,
        distance_upper_bound=float(radius),
        workers=-1,
    )
    if query_k == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]

    directed: list[tuple[int, int]] = []
    for src in range(num_nodes):
        candidates = sorted(
            (
                (float(distance), int(dst))
                for distance, dst in zip(distances[src], neighbors[src], strict=False)
                if int(dst) < num_nodes
                and int(dst) != src
                and (src, int(dst)) not in mesh_edges
            ),
            key=lambda item: (item[0], item[1]),
        )
        directed.extend((src, dst) for _, dst in candidates[:max_neighbors])
    if not directed:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.as_tensor(directed, dtype=torch.long).t().contiguous()


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
        self.count += int(tensor.shape[0])

    def finalize(self, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0 or self.sum is None or self.sum_sq is None:
            raise ValueError("Cannot finalize empty statistics")
        mean = self.sum / self.count
        var = torch.clamp(self.sum_sq / self.count - mean * mean, min=0.0)
        return mean, torch.sqrt(var + float(eps))
