from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from valgraphnet.data.case import load_case, read_split_file


def conditional_variance_ratio(
    invariants: np.ndarray,
    stress: np.ndarray,
    *,
    neighbors: int = 16,
) -> float:
    """Estimate Var(stress | nearby invariants) / Var(stress)."""

    features = np.asarray(invariants, dtype=np.float64)
    target = np.asarray(stress, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != target.shape[0]:
        raise ValueError("invariants and stress must share their sample dimension")
    if features.shape[0] < 3:
        raise ValueError("at least three samples are required")
    scale = features.std(axis=0)
    normalized = (features - features.mean(axis=0)) / np.maximum(scale, 1.0e-12)
    try:
        from scipy.spatial import cKDTree

        _, indices = cKDTree(normalized).query(
            normalized, k=min(max(int(neighbors), 2) + 1, len(normalized))
        )
    except ImportError:
        import torch

        value = torch.from_numpy(normalized).float()
        indices = torch.cdist(value, value).topk(
            min(max(int(neighbors), 2) + 1, len(normalized)),
            largest=False,
        ).indices.numpy()
    local = target[indices[:, 1:]]
    conditional = float(np.mean(np.var(local, axis=1)))
    global_variance = float(np.var(target))
    return conditional / max(global_variance, 1.0e-30)


def sample_constitutive_pairs(
    case_root: str | Path,
    split_file: str | Path,
    split: str,
    *,
    cases: int = 20,
    frames: int = 20,
    cells: int = 512,
    max_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(case_root)
    ids = read_split_file(split_file, split)
    selected_ids = np.asarray(ids, dtype=object)[
        np.linspace(0, len(ids) - 1, min(cases, len(ids))).round().astype(int)
    ]
    feature_parts = []
    stress_parts = []
    for case_id in selected_ids:
        case = load_case(root / str(case_id))
        frame_ids = np.linspace(
            0, case.num_steps - 1, min(frames, case.num_steps)
        ).round().astype(int)
        cell_ids = np.linspace(
            0, case.num_cells - 1, min(cells, case.num_cells)
        ).round().astype(int)
        connectivity = case.cells[cell_ids]
        dm_inv = case.dm_inv[cell_ids].astype(np.float64)
        for frame in frame_ids:
            position = case.nodes.astype(np.float64) + case.displacement[frame].astype(
                np.float64
            )
            coordinates = position[connectivity]
            ds = np.stack(
                [
                    coordinates[:, 1] - coordinates[:, 0],
                    coordinates[:, 2] - coordinates[:, 0],
                    coordinates[:, 3] - coordinates[:, 0],
                ],
                axis=2,
            )
            deformation = ds @ dm_inv
            c = np.swapaxes(deformation, 1, 2) @ deformation
            i1 = np.trace(c, axis1=1, axis2=2)
            i2 = 0.5 * (
                i1**2 - np.trace(c @ c, axis1=1, axis2=2)
            )
            j = np.linalg.det(deformation)
            safe_j = np.maximum(j, 1.0e-8)
            feature_parts.append(
                np.stack(
                    [i1 * safe_j ** (-2.0 / 3.0), i2 * safe_j ** (-4.0 / 3.0), j],
                    axis=1,
                )
            )
            if case.cell_stress.shape[-1] > 0:
                cell_stress = case.cell_stress[frame, cell_ids, 0]
            else:
                cell_stress = case.stress[frame, connectivity, 0].mean(axis=1)
            stress_parts.append(np.asarray(cell_stress, dtype=np.float64))
    features = np.concatenate(feature_parts)
    targets = np.concatenate(stress_parts)
    if len(features) > int(max_samples):
        keep = np.linspace(0, len(features) - 1, int(max_samples)).round().astype(int)
        features = features[keep]
        targets = targets[keep]
    return features, targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether cell memory is justified by constitutive ambiguity."
    )
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    invariants, stress = sample_constitutive_pairs(
        args.case_root,
        args.split_file,
        args.split,
        cases=args.cases,
        frames=args.frames,
        cells=args.cells,
        max_samples=args.max_samples,
    )
    ratio = conditional_variance_ratio(
        invariants, stress, neighbors=args.neighbors
    )
    result = {
        "samples": int(len(stress)),
        "neighbors": int(args.neighbors),
        "conditional_to_global_variance": ratio,
        "trigger_threshold": float(args.threshold),
        "enable_cell_memory": bool(ratio > args.threshold),
        "stress_source": "cell tensor if available, otherwise tetra nodal mean",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
