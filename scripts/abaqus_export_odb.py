"""Export Abaqus ODB valve simulations to ValGraphNet case directories.

Run inside Abaqus Python, for example:

    abaqus python scripts/abaqus_export_odb.py -- \
      --odb case.odb \
      --out data/processed/case_001 \
      --instance VALVE \
      --fixed-set ATTACHMENT \
      --pressure-surface VENTRICULAR_SURFACE \
      --leaflet-sets LEAFLET_1,LEAFLET_2,LEAFLET_3 \
      --pressure-csv pressure.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    args = parse_args(sys.argv[1:])

    try:
        from odbAccess import openOdb
    except ImportError as exc:
        raise RuntimeError("This script must run inside Abaqus Python.") from exc

    odb = openOdb(path=args.odb, readOnly=True)
    try:
        instance = select_instance(odb, args.instance)
        labels, nodes = export_nodes(instance)
        label_to_index = {int(label): idx for idx, label in enumerate(labels)}
        elements = export_elements(instance, label_to_index)
        element_labels = export_element_labels(instance)
        cells, cell_source_indices, cell_element_labels = export_tetrahedral_cells(
            instance,
            label_to_index,
        )
        fixed_mask = export_node_mask(odb, instance, args.fixed_set, label_to_index)
        pressure_mask = export_pressure_mask(odb, instance, args.pressure_surface, label_to_index)
        leaflet_id = export_leaflet_ids(odb, instance, args.leaflet_sets, label_to_index)
        times, u, v, a, s = export_frames(
            odb=odb,
            instance=instance,
            labels=labels,
            label_to_index=label_to_index,
            u_name=args.u_field,
            v_name=args.v_field,
            a_name=args.a_field,
            s_name=args.s_field,
        )
        pressure = load_pressure(args.pressure_csv, times)
        thickness = np.full(nodes.shape[0], args.thickness, dtype=np.float32)
        s_element, s_integration_point, integration_point_mask = export_element_tensor_frames(
            odb,
            instance,
            element_labels,
            args.s_field,
        )
        le_element, _, strain_integration_point_mask = export_element_tensor_frames(
            odb,
            instance,
            element_labels,
            args.strain_field,
        )
        s_cell = s_element[:, cell_source_indices]
        s_cell_integration_point = s_integration_point[:, cell_source_indices]
        cell_integration_point_mask = integration_point_mask[:, cell_source_indices]
        le_cell = le_element[:, cell_source_indices]
        material, density, fiber_direction, material_features = load_material_sidecars(
            args,
            cell_element_labels,
        )
        dm_inv, reference_volume, shape_gradients = tetra_reference_geometry(nodes, cells)
        lumped_mass = lumped_tetra_mass(
            nodes.shape[0],
            cells,
            reference_volume,
            density,
        )

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "nodes.npy", nodes)
        np.save(out / "elements.npy", elements)
        np.save(out / "element_labels.npy", element_labels)
        np.save(out / "cells.npy", cells)
        np.save(out / "cell_element_labels.npy", cell_element_labels)
        np.save(out / "times.npy", times)
        np.save(out / "pressure.npy", pressure)
        np.save(out / "U.npy", u)
        np.save(out / "V.npy", v)
        np.save(out / "A.npy", a)
        np.save(out / "S.npy", s)
        np.save(out / "S_element.npy", s_element)
        np.save(out / "S_element_integration_point.npy", s_integration_point)
        np.save(out / "element_integration_point_mask.npy", integration_point_mask)
        np.save(out / "S_cell.npy", s_cell)
        np.save(out / "S_integration_point.npy", s_cell_integration_point)
        np.save(out / "integration_point_mask.npy", cell_integration_point_mask)
        if np.any(strain_integration_point_mask):
            np.save(out / "LE_element.npy", le_element)
            np.save(out / "LE_cell.npy", le_cell)
        np.save(out / "fixed_mask.npy", fixed_mask)
        np.save(out / "pressure_mask.npy", pressure_mask)
        np.save(out / "leaflet_id.npy", leaflet_id)
        np.save(out / "thickness.npy", thickness)
        np.save(out / "Dm_inv.npy", dm_inv)
        np.save(out / "reference_volume.npy", reference_volume)
        np.save(out / "shape_gradients.npy", shape_gradients)
        np.save(out / "lumped_mass.npy", lumped_mass)
        np.save(out / "density.npy", density)
        np.save(out / "fiber_direction.npy", fiber_direction)
        np.save(out / "material_features.npy", material_features)
        if material:
            with (out / "material.json").open("w", encoding="utf-8") as f:
                json.dump(material, f, indent=2)

        metadata = {
            "case_id": args.case_id or out.name,
            "schema_version": 2,
            "odb": str(Path(args.odb).resolve()),
            "instance": instance.name,
            "fixed_set": args.fixed_set,
            "pressure_surface": args.pressure_surface,
            "leaflet_sets": split_csv_arg(args.leaflet_sets),
            "field_names": {
                "U": args.u_field,
                "V": args.v_field,
                "A": args.a_field,
                "S": args.s_field,
                "LE": args.strain_field,
            },
            "stress_tensor_components": ["S11", "S22", "S33", "S12", "S13", "S23"],
            "strain_tensor_components": ["LE11", "LE22", "LE33", "LE12", "LE13", "LE23"],
            "element_stress_representation": "integration-point values and their masked mean",
            "cell_representation": "four-node C3D4-family linear tetrahedra",
            "material_json": args.material_json,
            "num_nodes": int(nodes.shape[0]),
            "num_elements": int(elements.shape[0]),
            "num_cells": int(cells.shape[0]),
            "num_frames": int(times.shape[0]),
        }
        with (out / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print("Exported ValGraphNet case:", out)
    finally:
        odb.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--fixed-set", required=True)
    parser.add_argument("--pressure-surface", required=True)
    parser.add_argument("--leaflet-sets", default="")
    parser.add_argument("--pressure-csv", default=None)
    parser.add_argument("--thickness", type=float, default=1.0)
    parser.add_argument("--u-field", default="U")
    parser.add_argument("--v-field", default="V")
    parser.add_argument("--a-field", default="A")
    parser.add_argument("--s-field", default="S")
    parser.add_argument("--strain-field", default="LE")
    parser.add_argument(
        "--material-json",
        default=None,
        help="Optional material metadata JSON copied into the exported case.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Uniform cell density in the ODB unit system.",
    )
    parser.add_argument(
        "--density-file",
        default=None,
        help="Optional .npy/.csv/.json scalar density sidecar aligned by cell or element label.",
    )
    parser.add_argument(
        "--fiber-direction-file",
        default=None,
        help="Optional .npy/.csv/.json [M,3] fiber sidecar aligned by cell or element label.",
    )
    return parser.parse_args(argv)


def select_instance(odb, instance_name: str | None):
    instances = odb.rootAssembly.instances
    if instance_name:
        key = instance_name if instance_name in instances else instance_name.upper()
        if key not in instances:
            raise KeyError("Instance not found: %s. Available: %s" % (instance_name, list(instances.keys())))
        return instances[key]
    if len(instances) != 1:
        raise ValueError("Multiple instances found. Pass --instance. Available: %s" % list(instances.keys()))
    return list(instances.values())[0]


def export_nodes(instance) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(instance.nodes, key=lambda node: int(node.label))
    labels = np.asarray([int(node.label) for node in ordered], dtype=np.int64)
    coords = np.asarray([node.coordinates for node in ordered], dtype=np.float32)
    return labels, coords


def export_elements(instance, label_to_index: dict[int, int]) -> np.ndarray:
    rows = []
    max_len = 0
    for element in sorted(instance.elements, key=lambda elem: int(elem.label)):
        conn = [label_to_index[int(label)] for label in element.connectivity]
        rows.append(conn)
        max_len = max(max_len, len(conn))
    out = -np.ones((len(rows), max_len), dtype=np.int64)
    for row_idx, conn in enumerate(rows):
        out[row_idx, : len(conn)] = conn
    return out


def export_element_labels(instance) -> np.ndarray:
    """Return Abaqus element labels in the same order as ``export_elements``."""

    return np.asarray(
        [
            int(element.label)
            for element in sorted(instance.elements, key=lambda elem: int(elem.label))
        ],
        dtype=np.int64,
    )


def export_tetrahedral_cells(
    instance,
    label_to_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Export real four-node C3D4-family cells and their all-element row indices."""

    cells = []
    source_indices = []
    labels = []
    ordered = sorted(instance.elements, key=lambda elem: int(elem.label))
    for source_idx, element in enumerate(ordered):
        element_type = str(getattr(element, "type", "")).upper()
        connectivity = list(element.connectivity)
        if not element_type.startswith("C3D4") or len(connectivity) != 4:
            continue
        cells.append([label_to_index[int(label)] for label in connectivity])
        source_indices.append(source_idx)
        labels.append(int(element.label))
    return (
        np.asarray(cells, dtype=np.int64).reshape(-1, 4),
        np.asarray(source_indices, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )


def export_frames(
    odb,
    instance,
    labels: np.ndarray,
    label_to_index: dict[int, int],
    u_name: str,
    v_name: str,
    a_name: str,
    s_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    for step in odb.steps.values():
        frames.extend(step.frames)
    if not frames:
        raise ValueError("ODB contains no frames")

    n_frames = len(frames)
    n_nodes = labels.shape[0]
    times = np.asarray([float(frame.frameValue) for frame in frames], dtype=np.float32)
    u = np.zeros((n_frames, n_nodes, 3), dtype=np.float32)
    v = np.zeros_like(u)
    a = np.zeros_like(u)
    stress_arrays = []

    for frame_idx, frame in enumerate(frames):
        u[frame_idx] = nodal_vector_field(frame, instance, label_to_index, u_name, width=3)
        v[frame_idx] = nodal_vector_field(frame, instance, label_to_index, v_name, width=3)
        a[frame_idx] = nodal_vector_field(frame, instance, label_to_index, a_name, width=3)
        stress_arrays.append(nodal_stress_field(frame, instance, label_to_index, s_name))

    stress_width = max(array.shape[1] for array in stress_arrays) if stress_arrays else 0
    s = np.zeros((n_frames, n_nodes, stress_width), dtype=np.float32)
    for frame_idx, array in enumerate(stress_arrays):
        if array.size:
            s[frame_idx, :, : array.shape[1]] = array
    return times, u, v, a, s


def nodal_vector_field(frame, instance, label_to_index: dict[int, int], field_name: str, width: int) -> np.ndarray:
    field = frame.fieldOutputs.get(field_name)
    out = np.zeros((len(label_to_index), width), dtype=np.float32)
    if field is None:
        print("Warning: missing field %s at frame %s" % (field_name, frame.frameValue))
        return out
    subset = field.getSubset(region=instance)
    for value in subset.values:
        node_label = getattr(value, "nodeLabel", None)
        if node_label is None or int(node_label) not in label_to_index:
            continue
        data = np.asarray(value.data, dtype=np.float32).reshape(-1)
        out[label_to_index[int(node_label)], : min(width, data.size)] = data[:width]
    return out


def nodal_stress_field(frame, instance, label_to_index: dict[int, int], field_name: str) -> np.ndarray:
    field = frame.fieldOutputs.get(field_name)
    n_nodes = len(label_to_index)
    if field is None:
        print("Warning: missing field %s at frame %s" % (field_name, frame.frameValue))
        return np.zeros((n_nodes, 0), dtype=np.float32)

    try:
        from abaqusConstants import ELEMENT_NODAL

        subset = field.getSubset(region=instance, position=ELEMENT_NODAL)
    except Exception:
        subset = field.getSubset(region=instance)

    width = 0
    for value in subset.values:
        width = max(width, np.asarray(value.data).reshape(-1).size)
    if width == 0:
        return np.zeros((n_nodes, 0), dtype=np.float32)

    accum = np.zeros((n_nodes, width), dtype=np.float64)
    counts = np.zeros((n_nodes, 1), dtype=np.float64)
    for value in subset.values:
        node_label = getattr(value, "nodeLabel", None)
        if node_label is None or int(node_label) not in label_to_index:
            continue
        data = np.asarray(value.data, dtype=np.float64).reshape(-1)
        idx = label_to_index[int(node_label)]
        accum[idx, : data.size] += data
        counts[idx] += 1.0
    counts[counts == 0.0] = 1.0
    return (accum / counts).astype(np.float32)


TENSOR_COMPONENTS = ("11", "22", "33", "12", "13", "23")


def export_element_tensor_frames(
    odb,
    instance,
    element_labels: np.ndarray,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Export canonical symmetric tensors at each element integration point.

    The returned arrays are ``[T,E,6]``, ``[T,E,I,6]`` and ``[T,E,I]``.
    The last mask makes padding explicit when element types use different numbers
    of integration points or a field is unavailable at a frame.
    """

    frames = []
    for step in odb.steps.values():
        frames.extend(step.frames)
    label_to_index = {int(label): idx for idx, label in enumerate(element_labels)}
    records = []
    max_points = 0
    missing_warned = False
    for frame in frames:
        field = frame.fieldOutputs.get(field_name)
        by_element: dict[int, list[np.ndarray]] = {}
        if field is None:
            if not missing_warned:
                print("Warning: missing element field %s" % field_name)
                missing_warned = True
            records.append(by_element)
            continue
        try:
            from abaqusConstants import INTEGRATION_POINT

            subset = field.getSubset(region=instance, position=INTEGRATION_POINT)
        except Exception:
            subset = field.getSubset(region=instance)
        component_labels = tuple(
            str(label).upper() for label in getattr(field, "componentLabels", ())
        )
        for value in subset.values:
            element_label = getattr(value, "elementLabel", None)
            if element_label is None or int(element_label) not in label_to_index:
                continue
            tensor = canonical_tensor6(value.data, component_labels, field_name)
            by_element.setdefault(int(element_label), []).append(tensor)
        if by_element:
            max_points = max(max_points, max(len(values) for values in by_element.values()))
        records.append(by_element)

    num_frames = len(frames)
    num_elements = int(element_labels.shape[0])
    integration = np.zeros((num_frames, num_elements, max_points, 6), dtype=np.float32)
    mask = np.zeros((num_frames, num_elements, max_points), dtype=bool)
    for frame_idx, by_element in enumerate(records):
        for element_label, values in by_element.items():
            element_idx = label_to_index[element_label]
            count = len(values)
            integration[frame_idx, element_idx, :count] = np.asarray(values, dtype=np.float32)
            mask[frame_idx, element_idx, :count] = True

    element_mean = np.zeros((num_frames, num_elements, 6), dtype=np.float32)
    counts = mask.sum(axis=2, keepdims=True)
    if max_points:
        total = (integration * mask[..., None]).sum(axis=2)
        np.divide(total, np.maximum(counts, 1), out=element_mean, where=counts > 0)
    return element_mean, integration, mask


def canonical_tensor6(data, component_labels: tuple[str, ...], field_name: str) -> np.ndarray:
    """Map Abaqus tensor data to 11,22,33,12,13,23 component order."""

    values = np.asarray(data, dtype=np.float32).reshape(-1)
    out = np.zeros((6,), dtype=np.float32)
    if component_labels and len(component_labels) == values.size:
        normalized = [label.removeprefix(field_name.upper()) for label in component_labels]
        for value, suffix in zip(values, normalized, strict=False):
            if suffix in TENSOR_COMPONENTS:
                out[TENSOR_COMPONENTS.index(suffix)] = value
        return out
    if values.size >= 6:
        out[:] = values[:6]
    elif values.size == 4:
        out[[0, 1, 2, 3]] = values
    else:
        out[: values.size] = values
    return out


def tetra_reference_geometry(
    nodes: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute linear-tetrahedron reference geometry."""

    if cells.shape[0] == 0:
        return (
            np.zeros((0, 3, 3), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 4, 3), dtype=np.float32),
        )
    vertices = np.asarray(nodes, dtype=np.float64)[cells]
    dm = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=-1,
    )
    determinant = np.linalg.det(dm)
    if np.any(np.abs(determinant) <= np.finfo(np.float64).eps):
        bad = np.flatnonzero(np.abs(determinant) <= np.finfo(np.float64).eps)[:5].tolist()
        raise ValueError("Degenerate tetrahedral cells at indices %s" % bad)
    dm_inv = np.linalg.inv(dm)
    gradients = np.empty((cells.shape[0], 4, 3), dtype=np.float64)
    gradients[:, 1:, :] = dm_inv
    gradients[:, 0, :] = -dm_inv.sum(axis=1)
    return (
        dm_inv.astype(np.float32),
        (np.abs(determinant) / 6.0).astype(np.float32)[:, None],
        gradients.astype(np.float32),
    )


def lumped_tetra_mass(
    num_nodes: int,
    cells: np.ndarray,
    reference_volume: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    mass = np.zeros((num_nodes, 1), dtype=np.float32)
    if cells.size:
        contribution = reference_volume[:, 0] * density[:, 0] / 4.0
        for local_idx in range(4):
            np.add.at(mass[:, 0], cells[:, local_idx], contribution)
    return mass


def load_material_sidecars(
    args: argparse.Namespace,
    cell_element_labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Load optional material metadata and cell-aligned numeric sidecars."""

    material: dict[str, Any] = {}
    if args.material_json:
        with Path(args.material_json).open("r", encoding="utf-8") as f:
            material = json.load(f)
        if not isinstance(material, dict):
            raise ValueError("--material-json must contain a JSON object")

    num_cells = int(cell_element_labels.shape[0])
    density_source: Any = args.density
    if density_source is None:
        density_source = material.get("density", 1.0)
    if args.density_file:
        density_source = load_numeric_sidecar(args.density_file)
    density = align_cell_values(
        density_source,
        cell_element_labels,
        width=1,
        name="density",
    )
    if np.any(density <= 0.0):
        raise ValueError("density must be strictly positive")

    fiber_source: Any = material.get("fiber_direction", np.zeros((num_cells, 3)))
    if args.fiber_direction_file:
        fiber_source = load_numeric_sidecar(args.fiber_direction_file)
    fiber_direction = align_cell_values(
        fiber_source,
        cell_element_labels,
        width=3,
        name="fiber_direction",
    )
    norms = np.linalg.norm(fiber_direction, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0.0
    fiber_direction[nonzero] /= norms[nonzero]

    feature_source: Any = material.get("material_features", np.zeros((num_cells, 0)))
    material_features = align_material_features(feature_source, cell_element_labels)
    return material, density, fiber_direction, material_features


def load_numeric_sidecar(path: str | Path) -> Any:
    """Load a numeric .npy, .csv or .json sidecar without implicit pickle data."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                try:
                    rows.append([float(value) for value in row])
                except ValueError:
                    continue
        if not rows:
            raise ValueError("No numeric rows found in %s" % path)
        return np.asarray(rows, dtype=np.float32)
    raise ValueError("Unsupported sidecar extension: %s" % path.suffix)


def align_cell_values(
    value: Any,
    cell_element_labels: np.ndarray,
    width: int,
    name: str,
) -> np.ndarray:
    """Broadcast or element-label align scalar/vector cell data."""

    num_cells = int(cell_element_labels.shape[0])
    if isinstance(value, dict):
        if "values" in value:
            value = value["values"]
        else:
            rows = []
            for label in cell_element_labels:
                key = str(int(label))
                if key not in value:
                    raise ValueError("%s is missing element label %s" % (name, key))
                item = np.asarray(value[key], dtype=np.float32).reshape(-1)
                rows.append(item)
            value = np.asarray(rows, dtype=np.float32)

    array = np.asarray(value, dtype=np.float32)
    if width == 1 and array.ndim == 0:
        array = np.full((num_cells, 1), float(array), dtype=np.float32)
    elif array.shape == (width,):
        array = np.broadcast_to(array[None, :], (num_cells, width)).copy()
    elif width == 1 and array.shape == (num_cells,):
        array = array[:, None]
    elif array.ndim == 2 and array.shape == (num_cells, width + 1):
        labels = array[:, 0].astype(np.int64)
        by_label = {int(label): row[1:] for label, row in zip(labels, array, strict=False)}
        try:
            array = np.asarray(
                [by_label[int(label)] for label in cell_element_labels],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise ValueError("%s is missing element label %s" % (name, exc.args[0])) from exc
    elif array.shape == (1, width) and num_cells != 1:
        array = np.broadcast_to(array, (num_cells, width)).copy()
    if array.shape != (num_cells, width):
        raise ValueError("%s must be [%d], [M,%d], or label-prefixed [M,%d]" % (
            name,
            width,
            width,
            width + 1,
        ))
    if not np.all(np.isfinite(array)):
        raise ValueError("%s must contain finite values" % name)
    return array


def align_material_features(value: Any, cell_element_labels: np.ndarray) -> np.ndarray:
    num_cells = int(cell_element_labels.shape[0])
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = np.broadcast_to(array[None, :], (num_cells, array.shape[0])).copy()
    elif array.ndim == 2 and array.shape[0] == 1 and num_cells != 1:
        array = np.broadcast_to(array, (num_cells, array.shape[1])).copy()
    if array.ndim != 2 or array.shape[0] != num_cells:
        raise ValueError("material_features must have shape [P] or [M,P]")
    if not np.all(np.isfinite(array)):
        raise ValueError("material_features must contain finite values")
    return array


def export_node_mask(odb, instance, set_name: str, label_to_index: dict[int, int]) -> np.ndarray:
    labels = labels_from_region(resolve_region(odb, instance, set_name))
    return mask_from_labels(labels, label_to_index)


def export_pressure_mask(odb, instance, surface_name: str, label_to_index: dict[int, int]) -> np.ndarray:
    region = resolve_region(odb, instance, surface_name, prefer_surface=True)
    labels = labels_from_region(region)
    return mask_from_labels(labels, label_to_index)


def export_leaflet_ids(
    odb,
    instance,
    leaflet_sets: str,
    label_to_index: dict[int, int],
) -> np.ndarray:
    ids = np.zeros((len(label_to_index),), dtype=np.int64)
    for leaflet_idx, set_name in enumerate(split_csv_arg(leaflet_sets), start=1):
        labels = labels_from_region(resolve_region(odb, instance, set_name))
        mask = mask_from_labels(labels, label_to_index)
        ids[mask] = leaflet_idx
    return ids


def resolve_region(odb, instance, name: str, prefer_surface: bool = False):
    assembly = odb.rootAssembly
    containers = []
    if prefer_surface:
        containers.extend([getattr(instance, "surfaces", {}), getattr(assembly, "surfaces", {})])
    containers.extend([getattr(instance, "nodeSets", {}), getattr(assembly, "nodeSets", {})])
    if not prefer_surface:
        containers.extend([getattr(instance, "surfaces", {}), getattr(assembly, "surfaces", {})])

    for container in containers:
        if name in container:
            return container[name]
        upper = name.upper()
        if upper in container:
            return container[upper]
    raise KeyError("Region not found: %s" % name)


def labels_from_region(region) -> set[int]:
    labels: set[int] = set()
    if hasattr(region, "nodes"):
        collect_node_labels(region.nodes, labels)
    if hasattr(region, "elements"):
        for element in flatten_region_items(region.elements):
            for label in getattr(element, "connectivity", []):
                labels.add(int(label))
    return labels


def collect_node_labels(nodes_obj, labels: set[int]) -> None:
    for item in flatten_region_items(nodes_obj):
        if hasattr(item, "label"):
            labels.add(int(item.label))


def flatten_region_items(obj):
    if obj is None:
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from flatten_region_items(item)
    else:
        try:
            iterator = iter(obj)
        except TypeError:
            yield obj
            return
        for item in iterator:
            if isinstance(item, (list, tuple)):
                yield from flatten_region_items(item)
            else:
                yield item


def mask_from_labels(labels: set[int], label_to_index: dict[int, int]) -> np.ndarray:
    mask = np.zeros((len(label_to_index),), dtype=bool)
    for label in labels:
        idx = label_to_index.get(int(label))
        if idx is not None:
            mask[idx] = True
    return mask


def load_pressure(path: str | None, times: np.ndarray) -> np.ndarray:
    if not path:
        print("Warning: --pressure-csv not provided; pressure.npy will be all zeros")
        return np.zeros_like(times, dtype=np.float32)

    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError("No numeric time,pressure rows found in %s" % path)
    pressure_time = np.asarray([row[0] for row in rows], dtype=np.float64)
    pressure_value = np.asarray([row[1] for row in rows], dtype=np.float64)
    return np.interp(times.astype(np.float64), pressure_time, pressure_value).astype(np.float32)


def split_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()

