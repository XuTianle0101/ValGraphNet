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

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "nodes.npy", nodes)
        np.save(out / "elements.npy", elements)
        np.save(out / "times.npy", times)
        np.save(out / "pressure.npy", pressure)
        np.save(out / "U.npy", u)
        np.save(out / "V.npy", v)
        np.save(out / "A.npy", a)
        np.save(out / "S.npy", s)
        np.save(out / "fixed_mask.npy", fixed_mask)
        np.save(out / "pressure_mask.npy", pressure_mask)
        np.save(out / "leaflet_id.npy", leaflet_id)
        np.save(out / "thickness.npy", thickness)

        metadata = {
            "case_id": args.case_id or out.name,
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
            },
            "num_nodes": int(nodes.shape[0]),
            "num_elements": int(elements.shape[0]),
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

