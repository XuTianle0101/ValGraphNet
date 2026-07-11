"""Parse ASCII CalculiX results and export HyperContact-3D ValveCases.

CalculiX stores nodal field output in FRD and integration-point element output
in DAT.  HyperContact decks use ``*NODE FILE``/``*EL FILE`` (ASCII FRD) and
``*EL PRINT, ELSET=BLOCK`` (ASCII DAT), avoiding binary-format ambiguity.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from valgraphnet.hypercontact_solver import load_manifest, select_cases


_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")
_EXPONENTIAL = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+)[EeDd][+-]?\d+")
_TIME = re.compile(
    r"\btime\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
    re.IGNORECASE,
)
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MATERIAL_FEATURE_NAMES = ("c10_pa", "poisson_ratio", "density_kg_m3")


@dataclass(frozen=True)
class InputDeckMesh:
    node_labels: np.ndarray
    nodes: np.ndarray
    block_node_labels: np.ndarray
    indenter_node_labels: np.ndarray
    element_labels: np.ndarray
    cells: np.ndarray
    indenter_triangles: np.ndarray
    bottom_node_labels: np.ndarray
    top_node_labels: np.ndarray


@dataclass(frozen=True)
class FrdDataset:
    name: str
    time: float | None
    components: tuple[str, ...]
    values: dict[int, np.ndarray]


@dataclass(frozen=True)
class DatStressFrame:
    time: float
    values: dict[int, np.ndarray]


def _numbers(line: str) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in _NUMBER.findall(line)]


def _record_code(line: str) -> int | None:
    match = re.match(r"^\s*(-?\d+)", line)
    return int(match.group(1)) if match else None


def parse_hypercontact_deck(path: str | Path) -> InputDeckMesh:
    """Read deformable and rigid-surface topology from a generated deck."""

    deck = Path(path)
    block_nodes: dict[int, tuple[float, float, float]] = {}
    indenter_nodes: dict[int, tuple[float, float, float]] = {}
    cells: dict[int, tuple[int, int, int, int]] = {}
    indenter_triangles: dict[int, tuple[int, int, int]] = {}
    node_sets: dict[str, list[int]] = {"BOTTOM": [], "BLOCK_TOP": []}
    section: tuple[str, dict[str, str | bool]] | None = None

    for raw in deck.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            parts = [part.strip() for part in line.split(",")]
            keyword = parts[0][1:].upper()
            parameters: dict[str, str | bool] = {}
            for part in parts[1:]:
                if "=" in part:
                    key, value = part.split("=", 1)
                    parameters[key.strip().upper()] = value.strip().upper()
                elif part:
                    parameters[part.upper()] = True
            section = (keyword, parameters)
            continue
        if section is None:
            continue
        keyword, parameters = section
        fields = [field.strip() for field in line.split(",") if field.strip()]
        node_set = parameters.get("NSET")
        if keyword == "NODE" and node_set in {"BLOCK_NODES", "INDENTER_NODES"}:
            if len(fields) < 4:
                raise ValueError(f"{deck}: malformed {node_set} record: {raw!r}")
            label = int(fields[0])
            destination = block_nodes if node_set == "BLOCK_NODES" else indenter_nodes
            destination[label] = tuple(float(value) for value in fields[1:4])
        elif (
            keyword == "ELEMENT"
            and parameters.get("ELSET") == "BLOCK"
            and str(parameters.get("TYPE", "")).startswith("C3D4")
        ):
            if len(fields) != 5:
                raise ValueError(f"{deck}: BLOCK must contain one-line C3D4 records")
            cells[int(fields[0])] = tuple(int(value) for value in fields[1:5])
        elif (
            keyword == "ELEMENT"
            and parameters.get("ELSET") == "INDENTER"
            and str(parameters.get("TYPE", "")).startswith("S3")
        ):
            if len(fields) != 4:
                raise ValueError(f"{deck}: INDENTER must contain one-line S3 records")
            indenter_triangles[int(fields[0])] = tuple(int(value) for value in fields[1:4])
        elif keyword == "NSET" and parameters.get("NSET") in node_sets:
            name = str(parameters["NSET"])
            values = [int(value) for value in fields]
            if "GENERATE" in parameters:
                if len(values) % 3:
                    raise ValueError(
                        f"{deck}: generated NSET {name} must use start,end,step triplets"
                    )
                for offset in range(0, len(values), 3):
                    start, end, step = values[offset : offset + 3]
                    node_sets[name].extend(range(start, end + (1 if step > 0 else -1), step))
            else:
                node_sets[name].extend(values)

    if not block_nodes:
        raise ValueError(f"{deck}: no *NODE, NSET=BLOCK_NODES records")
    if not cells:
        raise ValueError(f"{deck}: no *ELEMENT, TYPE=C3D4, ELSET=BLOCK records")
    if not indenter_nodes:
        raise ValueError(f"{deck}: no *NODE, NSET=INDENTER_NODES records")
    if not indenter_triangles:
        raise ValueError(f"{deck}: no *ELEMENT, TYPE=S3, ELSET=INDENTER records")
    overlap = sorted(set(block_nodes) & set(indenter_nodes))
    if overlap:
        raise ValueError(f"{deck}: node labels occur in both bodies: {overlap[:5]}")
    block_node_labels = np.asarray(sorted(block_nodes), dtype=np.int64)
    indenter_node_labels = np.asarray(sorted(indenter_nodes), dtype=np.int64)
    node_labels = np.concatenate((block_node_labels, indenter_node_labels))
    label_to_index = {int(label): index for index, label in enumerate(node_labels)}
    block_label_to_index = {
        int(label): index for index, label in enumerate(block_node_labels)
    }
    element_labels = np.asarray(sorted(cells), dtype=np.int64)
    try:
        connectivity = np.asarray(
            [
                [block_label_to_index[node] for node in cells[int(label)]]
                for label in element_labels
            ],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"{deck}: BLOCK cell references a node outside BLOCK_NODES: {exc}"
        ) from exc
    try:
        shell_connectivity = np.asarray(
            [
                [label_to_index[node] for node in indenter_triangles[int(label)]]
                for label in sorted(indenter_triangles)
            ],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"{deck}: INDENTER triangle references a node outside INDENTER_NODES: {exc}"
        ) from exc
    return InputDeckMesh(
        node_labels=node_labels,
        nodes=np.asarray(
            [
                *(block_nodes[int(label)] for label in block_node_labels),
                *(indenter_nodes[int(label)] for label in indenter_node_labels),
            ],
            dtype=np.float64,
        ),
        block_node_labels=block_node_labels,
        indenter_node_labels=indenter_node_labels,
        element_labels=element_labels,
        cells=connectivity,
        indenter_triangles=shell_connectivity,
        bottom_node_labels=np.asarray(sorted(set(node_sets["BOTTOM"])), dtype=np.int64),
        top_node_labels=np.asarray(sorted(set(node_sets["BLOCK_TOP"])), dtype=np.int64),
    )


def _frd_dataset_time(header: str) -> float | None:
    exponentials = _EXPONENTIAL.findall(header)
    if exponentials:
        return float(exponentials[0].replace("D", "E").replace("d", "e"))
    # The official fixed-format header puts the dataset value in columns 13--24.
    if len(header) >= 24:
        candidate = header[12:24].strip()
        try:
            return float(candidate.replace("D", "E").replace("d", "e"))
        except ValueError:
            pass
    return None


def parse_ascii_frd(path: str | Path) -> list[FrdDataset]:
    """Parse all ASCII nodal result datasets from a CalculiX FRD file."""

    frd = Path(path)
    raw = frd.read_bytes()
    if b"\x00" in raw[:4096]:
        raise ValueError(
            f"{frd}: binary FRD is unsupported; generate ASCII output with *NODE FILE/*EL FILE"
        )
    lines = raw.decode("ascii", errors="strict").splitlines()
    datasets: list[FrdDataset] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.lstrip().upper().startswith("100C"):
            index += 1
            continue
        dataset_time = _frd_dataset_time(line)
        index += 1
        while index < len(lines) and _record_code(lines[index]) != -4:
            if lines[index].lstrip().upper().startswith("100C"):
                break
            index += 1
        if index >= len(lines) or _record_code(lines[index]) != -4:
            continue
        header_fields = lines[index].split()
        if len(header_fields) < 2:
            raise ValueError(f"{frd}: malformed FRD dataset header at line {index + 1}")
        name = header_fields[1].strip().upper()
        component_count = None
        for field in header_fields[2:]:
            try:
                component_count = int(field)
                break
            except ValueError:
                continue
        index += 1
        components: list[str] = []
        while index < len(lines) and _record_code(lines[index]) == -5:
            fields = lines[index].split()
            if len(fields) >= 2:
                components.append(fields[1].strip().upper())
            index += 1
        if component_count is None:
            component_count = len(components)
        if component_count <= 0:
            raise ValueError(f"{frd}: dataset {name} has no components")
        if len(components) < component_count:
            components.extend(
                f"C{offset + 1}" for offset in range(len(components), component_count)
            )

        values: dict[int, np.ndarray] = {}
        current_label: int | None = None
        current_values: list[float] = []

        def finish_record() -> None:
            if current_label is None:
                return
            if not current_values:
                raise ValueError(
                    f"{frd}: dataset {name} node {current_label} has no raw components"
                )
            # FRD's header count includes derived display entities such as
            # displacement magnitude ``ALL``. Those have a -5 definition but
            # no value in each -1 nodal record; retain the actual stored width.
            values[current_label] = np.asarray(current_values, dtype=np.float64)

        while index < len(lines):
            code = _record_code(lines[index])
            if code == -3:
                finish_record()
                index += 1
                break
            if code == -1:
                finish_record()
                numeric = _numbers(lines[index])
                if len(numeric) < 2:
                    raise ValueError(f"{frd}: malformed nodal record at line {index + 1}")
                current_label = int(numeric[1])
                current_values = numeric[2:]
            elif code == -2 and current_label is not None:
                current_values.extend(_numbers(lines[index])[1:])
            elif lines[index].lstrip().upper().startswith("100C"):
                finish_record()
                break
            index += 1
        datasets.append(
            FrdDataset(
                name=name,
                time=dataset_time,
                components=tuple(components[:component_count]),
                values=values,
            )
        )
    if not datasets:
        raise ValueError(f"{frd}: no ASCII 100C nodal result datasets found")
    return datasets


def parse_dat_stress(path: str | Path) -> list[DatStressFrame]:
    """Parse integration-point Cauchy stress tables written by ``*EL PRINT``."""

    dat = Path(path)
    lines = dat.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: list[DatStressFrame] = []
    index = 0
    while index < len(lines):
        lower = lines[index].lower()
        is_stress_header = "stress" in lower and "elem" in lower and "integ" in lower
        if not is_stress_header:
            index += 1
            continue
        match = _TIME.search(lines[index])
        if match is None:
            raise ValueError(f"{dat}: stress table at line {index + 1} has no parseable time")
        frame_time = float(match.group(1).replace("D", "E").replace("d", "e"))
        index += 1
        rows: dict[int, list[tuple[int, np.ndarray]]] = {}
        started = False
        while index < len(lines):
            line = lines[index]
            numeric = _numbers(line)
            if len(numeric) >= 8:
                element = int(numeric[0])
                integration_point = int(numeric[1])
                tensor = np.asarray(numeric[2:8], dtype=np.float64)
                if not np.isfinite(tensor).all():
                    raise ValueError(f"{dat}: non-finite stress at line {index + 1}")
                rows.setdefault(element, []).append((integration_point, tensor))
                started = True
                index += 1
                continue
            if started and line.strip():
                break
            index += 1
        if not rows:
            raise ValueError(f"{dat}: stress table at time {frame_time:g} contains no rows")
        packed: dict[int, np.ndarray] = {}
        for element, records in rows.items():
            point_ids = [point for point, _ in records]
            if len(point_ids) != len(set(point_ids)):
                raise ValueError(
                    f"{dat}: duplicate integration-point ids for element {element} "
                    f"at time {frame_time:g}"
                )
            packed[element] = np.stack([tensor for _, tensor in sorted(records)], axis=0)
        frames.append(DatStressFrame(time=frame_time, values=packed))
    return frames


def _component_indices(components: Sequence[str], kind: str) -> list[int]:
    normalized = [name.upper().replace("_", "") for name in components]
    if kind == "displacement":
        aliases = (("D1", "U1", "UX"), ("D2", "U2", "UY"), ("D3", "U3", "UZ"))
    elif kind == "stress":
        # ValveCase canonical tensor order is 11,22,33,12,13,23.  FRD normally
        # writes SXX,SYY,SZZ,SXY,SYZ,SZX, so the final two require reordering.
        aliases = (
            ("SXX", "S11"),
            ("SYY", "S22"),
            ("SZZ", "S33"),
            ("SXY", "S12"),
            ("SZX", "SXZ", "S13"),
            ("SYZ", "S23"),
        )
    else:
        raise ValueError(f"unknown component kind: {kind}")
    indices: list[int] = []
    for choices in aliases:
        index = next((normalized.index(alias) for alias in choices if alias in normalized), None)
        if index is None:
            raise ValueError(f"missing {kind} component {choices[0]} in {components}")
        indices.append(index)
    return indices


def _nodal_history(
    datasets: Sequence[FrdDataset],
    node_labels: np.ndarray,
    *,
    names: set[str],
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [dataset for dataset in datasets if dataset.name in names]
    if not selected:
        found = sorted({dataset.name for dataset in datasets})
        raise ValueError(f"FRD contains no {kind} dataset; found {found}")
    histories: list[np.ndarray] = []
    times: list[float] = []
    for ordinal, dataset in enumerate(selected):
        indices = _component_indices(dataset.components, kind)
        missing = [int(label) for label in node_labels if int(label) not in dataset.values]
        if missing:
            raise ValueError(
                f"FRD {dataset.name} at frame {ordinal} misses block node labels {missing[:5]}"
            )
        too_narrow = [
            int(label)
            for label in node_labels
            if max(indices) >= dataset.values[int(label)].shape[0]
        ]
        if too_narrow:
            raise ValueError(
                f"FRD {dataset.name} lacks raw {kind} components for node labels "
                f"{too_narrow[:5]}"
            )
        frame = np.stack(
            [dataset.values[int(label)][indices] for label in node_labels], axis=0
        )
        if not np.isfinite(frame).all():
            raise ValueError(f"FRD {dataset.name} frame {ordinal} contains non-finite values")
        histories.append(frame)
        times.append(float(dataset.time) if dataset.time is not None else float(ordinal + 1))
    time_array = np.asarray(times, dtype=np.float64)
    if time_array.size > 1 and np.any(np.diff(time_array) <= 0.0):
        raise ValueError(f"FRD {kind} dataset times must be strictly increasing: {times}")
    return time_array, np.stack(histories, axis=0)


def _match_frame_indices(
    target_times: np.ndarray,
    source_times: np.ndarray,
    name: str,
) -> np.ndarray:
    if target_times.shape == source_times.shape:
        scale = max(1.0, float(np.max(np.abs(target_times))))
        if np.allclose(target_times, source_times, rtol=1.0e-5, atol=1.0e-8 * scale):
            return np.arange(target_times.size, dtype=np.int64)
    tolerance = max(1.0e-8, 1.0e-5 * max(1.0, float(np.ptp(target_times))))
    indices = np.asarray([int(np.argmin(np.abs(source_times - time))) for time in target_times])
    error = np.abs(source_times[indices] - target_times)
    if np.any(error > tolerance) or len(set(indices.tolist())) != target_times.size:
        raise ValueError(
            f"cannot align {name} frames to displacement times; maximum time error={error.max():g}"
        )
    return indices


def _dat_cell_history(
    frames: Sequence[DatStressFrame],
    times: np.ndarray,
    element_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError("DAT contains no integration-point stress tables")
    source_times = np.asarray([frame.time for frame in frames], dtype=np.float64)
    indices = _match_frame_indices(times, source_times, "DAT stress")
    selected = [frames[int(index)] for index in indices]
    max_points = 0
    for frame in selected:
        missing = [int(label) for label in element_labels if int(label) not in frame.values]
        if missing:
            raise ValueError(
                f"DAT stress at time {frame.time:g} misses BLOCK elements {missing[:5]}"
            )
        point_counts = (frame.values[int(label)].shape[0] for label in element_labels)
        max_points = max(max_points, max(point_counts))
    integration = np.zeros((len(selected), len(element_labels), max_points, 6), dtype=np.float64)
    mask = np.zeros((len(selected), len(element_labels), max_points), dtype=bool)
    for time_index, frame in enumerate(selected):
        for cell_index, label in enumerate(element_labels):
            values = frame.values[int(label)]
            integration[time_index, cell_index, : values.shape[0]] = values
            mask[time_index, cell_index, : values.shape[0]] = True
    cell = np.divide(
        (integration * mask[..., None]).sum(axis=2),
        np.maximum(mask.sum(axis=2, keepdims=True), 1),
    )
    return cell, integration, mask


def _project_cell_to_nodes(
    cell_values: np.ndarray,
    cells: np.ndarray,
    volume: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    output = np.zeros((cell_values.shape[0], num_nodes, cell_values.shape[-1]), dtype=np.float64)
    weights = np.zeros((num_nodes,), dtype=np.float64)
    for local in range(4):
        np.add.at(weights, cells[:, local], volume[:, 0])
        for frame in range(cell_values.shape[0]):
            np.add.at(output[frame], cells[:, local], cell_values[frame] * volume)
    return np.divide(output, np.maximum(weights[None, :, None], np.finfo(np.float64).eps))


def _von_mises_tensor6(tensor: np.ndarray) -> np.ndarray:
    s11, s22, s33, s12, s13, s23 = np.moveaxis(tensor, -1, 0)
    equivalent_squared = 0.5 * (
        (s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
    ) + 3.0 * (s12**2 + s13**2 + s23**2)
    return np.sqrt(np.maximum(equivalent_squared, 0.0))


def _tetra_geometry(
    nodes: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = nodes[cells]
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
        raise ValueError(f"reference mesh contains degenerate tetrahedra at indices {bad}")
    inverse = np.linalg.inv(dm)
    gradients = np.empty((cells.shape[0], 4, 3), dtype=np.float64)
    gradients[:, 1:] = inverse
    gradients[:, 0] = -inverse.sum(axis=1)
    return inverse, (np.abs(determinant) / 6.0)[:, None], gradients


def _tetra_edges(cells: np.ndarray) -> np.ndarray:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edges = np.concatenate([cells[:, pair] for pair in pairs], axis=0)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _triangle_edges(triangles: np.ndarray) -> np.ndarray:
    pairs = ((0, 1), (1, 2), (2, 0))
    edges = np.concatenate([triangles[:, pair] for pair in pairs], axis=0)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _bidirectional_edges(*edge_groups: np.ndarray) -> np.ndarray:
    undirected = np.unique(np.concatenate(edge_groups, axis=0), axis=0)
    directed = np.concatenate((undirected, undirected[:, ::-1]), axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    return directed[order]


def _shell_lumped_mass(
    nodes: np.ndarray,
    triangles: np.ndarray,
    density: float,
    thickness: float,
) -> np.ndarray:
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("indenter density must be finite and positive")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("indenter shell thickness must be finite and positive")
    vertices = nodes[triangles]
    twice_area = np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    if np.any(twice_area <= np.finfo(np.float64).eps):
        bad = np.flatnonzero(twice_area <= np.finfo(np.float64).eps)[:5].tolist()
        raise ValueError(f"indenter contains degenerate shell triangles at indices {bad}")
    nodal_mass = np.zeros((nodes.shape[0], 1), dtype=np.float64)
    triangle_mass_third = 0.5 * twice_area * float(density) * float(thickness) / 3.0
    for local in range(3):
        np.add.at(nodal_mass[:, 0], triangles[:, local], triangle_mass_third)
    return nodal_mass


def _safe_case_destination(output: Path, case_id: str) -> Path:
    if (
        _CASE_ID.fullmatch(case_id) is None
        or case_id in {".", ".."}
        or Path(case_id).is_absolute()
        or len(Path(case_id).parts) != 1
        or "/" in case_id
        or "\\" in case_id
    ):
        raise ValueError(f"unsafe HyperContact case_id: {case_id!r}")
    destination = (output / case_id).resolve()
    try:
        destination.relative_to(output)
    except ValueError as exc:
        raise ValueError(f"unsafe HyperContact case_id: {case_id!r}") from exc
    return destination


def _finite_difference(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, times, axis=0, edge_order=1)


def _prepend_reference(
    times: np.ndarray,
    displacement: np.ndarray,
    nodal_stress: np.ndarray | None,
    cell_stress: np.ndarray | None,
    integration_stress: np.ndarray | None,
    integration_mask: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    bool,
]:
    if times.size == 0 or math.isclose(float(times[0]), 0.0, abs_tol=1.0e-12):
        return (
            times,
            displacement,
            nodal_stress,
            cell_stress,
            integration_stress,
            integration_mask,
            False,
        )
    times = np.concatenate(([0.0], times))
    displacement = np.concatenate((np.zeros_like(displacement[:1]), displacement), axis=0)
    if nodal_stress is not None:
        nodal_stress = np.concatenate((np.zeros_like(nodal_stress[:1]), nodal_stress), axis=0)
    if cell_stress is not None:
        cell_stress = np.concatenate((np.zeros_like(cell_stress[:1]), cell_stress), axis=0)
    if integration_stress is not None:
        integration_stress = np.concatenate(
            (np.zeros_like(integration_stress[:1]), integration_stress), axis=0
        )
    if integration_mask is not None:
        integration_mask = np.concatenate((integration_mask[:1], integration_mask), axis=0)
    return (
        times,
        displacement,
        nodal_stress,
        cell_stress,
        integration_stress,
        integration_mask,
        True,
    )


def _write_case_arrays(destination: Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in arrays.items():
        np.save(destination / name, value)


def convert_case(
    benchmark_root: str | Path,
    entry: Mapping[str, Any],
    output_directory: str | Path,
    *,
    require_dat_stress: bool = True,
) -> dict[str, Any]:
    """Convert one solved generated case to the repository's ValveCase schema."""

    root = Path(benchmark_root).resolve()
    deck = (root / str(entry["deck"])).resolve()
    try:
        deck.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"case deck escapes benchmark root: {entry['deck']!r}") from exc
    mesh = parse_hypercontact_deck(deck)
    frd_path = deck.with_suffix(".frd")
    dat_path = deck.with_suffix(".dat")
    datasets = parse_ascii_frd(frd_path)
    times, block_displacement = _nodal_history(
        datasets,
        mesh.block_node_labels,
        names={"DISP", "DISPLACEMENT"},
        kind="displacement",
    )
    dm_inv, volume, gradients = _tetra_geometry(mesh.nodes, mesh.cells)

    nodal_stress: np.ndarray | None = None
    try:
        stress_times, raw_nodal_stress = _nodal_history(
            datasets,
            mesh.block_node_labels,
            names={"STRESS", "STRES"},
            kind="stress",
        )
        nodal_stress = raw_nodal_stress[
            _match_frame_indices(times, stress_times, "FRD nodal stress")
        ]
    except ValueError:
        if not dat_path.is_file():
            raise

    dat_frames = parse_dat_stress(dat_path) if dat_path.is_file() else []
    stress_source = "CalculiX DAT integration-point Cauchy stress"
    integration_stress: np.ndarray | None = None
    integration_mask: np.ndarray | None = None
    try:
        cell_stress, integration_stress, integration_mask = _dat_cell_history(
            dat_frames, times, mesh.element_labels
        )
    except ValueError:
        if require_dat_stress:
            raise
        if nodal_stress is None:
            raise ValueError("neither DAT integration-point nor FRD nodal stress is available")
        cell_stress = nodal_stress[:, mesh.cells].mean(axis=2)
        stress_source = "mean of extrapolated FRD nodal Cauchy stress (fallback)"

    if nodal_stress is None:
        nodal_stress = _project_cell_to_nodes(
            cell_stress, mesh.cells, volume, mesh.block_node_labels.shape[0]
        )

    block_node_count = int(mesh.block_node_labels.shape[0])
    indenter_node_count = int(mesh.indenter_node_labels.shape[0])
    derived_parameters = dict(entry.get("derived", {}))
    if "imposed_indenter_displacement_m" not in derived_parameters:
        raise ValueError(
            f"{entry['case_id']}: manifest is missing imposed_indenter_displacement_m"
        )
    imposed_indenter_displacement = float(
        derived_parameters["imposed_indenter_displacement_m"]
    )
    step_duration = float(derived_parameters.get("step_duration", 1.0))
    if not math.isfinite(step_duration) or step_duration <= 0.0:
        raise ValueError("HyperContact step_duration must be finite and positive")
    normalized_time = times / step_duration
    if np.any(normalized_time < -1.0e-8) or np.any(normalized_time > 1.0 + 1.0e-6):
        raise ValueError(
            "FRD displacement times lie outside the generated rigid-motion step"
        )
    rigid_displacement = np.zeros(
        (times.shape[0], indenter_node_count, 3), dtype=np.float64
    )
    rigid_displacement[:, :, 2] = (
        imposed_indenter_displacement * normalized_time[:, None]
    )
    displacement = np.concatenate((block_displacement, rigid_displacement), axis=1)
    full_nodal_stress = np.zeros(
        (times.shape[0], mesh.nodes.shape[0], 7), dtype=np.float64
    )
    full_nodal_stress[:, :block_node_count, 0] = _von_mises_tensor6(nodal_stress)
    full_nodal_stress[:, :block_node_count, 1:] = nodal_stress
    nodal_stress = full_nodal_stress

    (
        times,
        displacement,
        nodal_stress,
        cell_stress,
        integration_stress,
        integration_mask,
        prepended_reference,
    ) = _prepend_reference(
        times,
        displacement,
        nodal_stress,
        cell_stress,
        integration_stress,
        integration_mask,
    )
    assert nodal_stress is not None and cell_stress is not None
    velocity = _finite_difference(displacement, times)
    acceleration = _finite_difference(velocity, times)

    label_to_index = {int(label): index for index, label in enumerate(mesh.node_labels)}
    fixed = np.zeros(mesh.nodes.shape[0], dtype=bool)
    prescribed = np.zeros(mesh.nodes.shape[0], dtype=bool)
    contact = np.zeros(mesh.nodes.shape[0], dtype=bool)
    for label in mesh.bottom_node_labels:
        if int(label) in label_to_index:
            fixed[label_to_index[int(label)]] = True
    for label in mesh.top_node_labels:
        if int(label) in label_to_index:
            contact[label_to_index[int(label)]] = True
    prescribed[block_node_count:] = True
    contact[block_node_count:] = True

    parameters = dict(entry.get("parameters", {}))
    material_parameters = dict(parameters.get("material", {}))
    density_value = float(material_parameters.get("density_kg_m3", 1.0))
    density = np.full((mesh.cells.shape[0], 1), density_value, dtype=np.float64)
    nodal_mass = np.zeros((mesh.nodes.shape[0], 1), dtype=np.float64)
    contribution = volume[:, 0] * density[:, 0] / 4.0
    for local in range(4):
        np.add.at(nodal_mass[:, 0], mesh.cells[:, local], contribution)
    indenter_density = float(
        derived_parameters.get("indenter_density_kg_m3", 7800.0)
    )
    indenter_thickness = float(
        derived_parameters.get("indenter_shell_thickness_m", 1.0e-4)
    )
    rigid_mass = _shell_lumped_mass(
        mesh.nodes,
        mesh.indenter_triangles,
        indenter_density,
        indenter_thickness,
    )
    nodal_mass += rigid_mass
    if np.any(nodal_mass <= 0.0):
        bad = np.flatnonzero(nodal_mass[:, 0] <= 0.0)[:5].tolist()
        raise ValueError(f"converted full-node mass is non-positive at indices {bad}")
    feature_values = np.asarray(
        [float(material_parameters.get(name, 0.0)) for name in MATERIAL_FEATURE_NAMES],
        dtype=np.float64,
    )
    material_features = np.broadcast_to(feature_values, (mesh.cells.shape[0], 3)).copy()
    edge_elements = _bidirectional_edges(
        _tetra_edges(mesh.cells),
        _triangle_edges(mesh.indenter_triangles),
    )
    leaflet_id = np.where(prescribed, 1, 0).astype(np.int64)
    node_type = np.zeros(mesh.nodes.shape[0], dtype=np.int64)
    node_type[fixed] = 3
    node_type[prescribed] = 1
    nodal_thickness = np.ones(mesh.nodes.shape[0], dtype=np.float32)
    nodal_thickness[prescribed] = indenter_thickness
    arrays: dict[str, np.ndarray] = {
        "nodes.npy": mesh.nodes.astype(np.float32),
        "elements.npy": edge_elements.astype(np.int64),
        "cells.npy": mesh.cells.astype(np.int64),
        "indenter_triangles.npy": mesh.indenter_triangles.astype(np.int64),
        "indenter_node_mask.npy": prescribed,
        "cell_element_labels.npy": mesh.element_labels.astype(np.int64),
        "node_labels.npy": mesh.node_labels.astype(np.int64),
        "times.npy": times.astype(np.float32),
        "pressure.npy": np.zeros(times.shape, dtype=np.float32),
        "U.npy": displacement.astype(np.float32),
        "V.npy": velocity.astype(np.float32),
        "A.npy": acceleration.astype(np.float32),
        "S.npy": nodal_stress.astype(np.float32),
        "S_cell.npy": cell_stress.astype(np.float32),
        "fixed_mask.npy": fixed,
        "prescribed_mask.npy": prescribed,
        "pressure_mask.npy": np.zeros_like(fixed),
        "contact_surface_mask.npy": contact,
        "leaflet_id.npy": leaflet_id,
        "node_type.npy": node_type,
        "thickness.npy": nodal_thickness,
        "Dm_inv.npy": dm_inv.astype(np.float32),
        "reference_volume.npy": volume.astype(np.float32),
        "shape_gradients.npy": gradients.astype(np.float32),
        "lumped_mass.npy": nodal_mass.astype(np.float32),
        "density.npy": density.astype(np.float32),
        "material_features.npy": material_features.astype(np.float32),
        "fiber_direction.npy": np.zeros((mesh.cells.shape[0], 3), dtype=np.float32),
    }
    if integration_stress is not None and integration_mask is not None:
        arrays["S_integration_point.npy"] = integration_stress.astype(np.float32)
        arrays["integration_point_mask.npy"] = integration_mask

    destination = Path(output_directory)
    _write_case_arrays(destination, arrays)
    material = {
        "material_feature_names": list(MATERIAL_FEATURE_NAMES),
        "model": "compressible_neo_hookean",
        "source": "HyperContact-3D generation manifest",
        "units": "SI",
        **material_parameters,
    }
    (destination / "material.json").write_text(
        json.dumps(material, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata = {
        "case_id": str(entry["case_id"]),
        "cell_representation": "four-node C3D4 linear tetrahedra",
        "contact_surface_mask": "BLOCK_TOP plus all prescribed INDENTER_NODES",
        "derived_kinematics": "V and A are finite differences with respect to static step time",
        "element_representation": (
            "directed two-node mesh edges derived from tetrahedral cells and "
            "indenter triangles"
        ),
        "frd": str(frd_path),
        "dat": str(dat_path) if dat_path.is_file() else None,
        "num_cells": int(mesh.cells.shape[0]),
        "num_block_nodes": block_node_count,
        "num_frames": int(times.shape[0]),
        "num_indenter_nodes": indenter_node_count,
        "num_indenter_triangles": int(mesh.indenter_triangles.shape[0]),
        "num_nodes": int(mesh.nodes.shape[0]),
        "material_feature_names": list(MATERIAL_FEATURE_NAMES),
        "parameters": parameters,
        "prepended_zero_reference_frame": prepended_reference,
        "schema_version": 2,
        "source": "HyperContact-3D / CalculiX",
        "split": str(entry.get("split", "unknown")),
        "stress_source": stress_source,
        "rigid_nodal_stress": "zero; prescribed rigid surface has no constitutive cells",
        "nodal_stress_components": [
            "von_mises",
            "S11",
            "S22",
            "S33",
            "S12",
            "S13",
            "S23",
        ],
        "rigid_indenter": {
            "density_kg_m3": indenter_density,
            "lumped_mass": "triangulated shell area times density and thickness",
            "prescribed_displacement": "linear ramp in z over normalized static step time",
            "shell_thickness_m": indenter_thickness,
        },
        "stress_tensor_components": ["S11", "S22", "S33", "S12", "S13", "S23"],
        "time_semantics": "quasi-static normalized step time, not physical transient time",
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def convert_benchmark(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    splits: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
    require_dat_stress: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Convert selected solved cases and write processed split metadata."""

    benchmark_root, manifest = load_manifest(manifest_path)
    entries = select_cases(manifest, splits=splits, case_ids=case_ids)
    if not entries:
        raise ValueError("case selection is empty")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    split_map: dict[str, list[str]] = {}
    for entry in entries:
        case_id = str(entry["case_id"])
        destination = _safe_case_destination(output, case_id)
        if destination.exists():
            if not force:
                raise FileExistsError(f"refusing to replace converted case: {destination}")
            shutil.rmtree(destination)
        record = convert_case(
            benchmark_root,
            entry,
            destination,
            require_dat_stress=require_dat_stress,
        )
        records.append(record)
        split_map.setdefault(str(entry.get("split", "unknown")), []).append(case_id)
    for case_list in split_map.values():
        case_list.sort()
    summary = {
        "case_count": len(records),
        "cases": records,
        "manifest": str(Path(manifest_path).resolve()),
        "require_dat_stress": bool(require_dat_stress),
        "schema_version": 2,
        "split_counts": {name: len(values) for name, values in sorted(split_map.items())},
    }
    (output / "splits.json").write_text(
        json.dumps(dict(sorted(split_map.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "conversion_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary
