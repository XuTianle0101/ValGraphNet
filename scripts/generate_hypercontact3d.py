#!/usr/bin/env python3
"""Generate the open HyperContact-3D CalculiX benchmark.

The benchmark is a compressible Neo-Hookean block indented by a triangulated
rigid sphere.  This module deliberately only writes solver inputs and metadata:
CalculiX is not required to generate, inspect, or test the benchmark.

Example
-------
python scripts/generate_hypercontact3d.py \
    --config configs/hypercontact3d.yaml \
    --output data/hypercontact3d_raw
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


GENERATOR_VERSION = "1.9"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BlockMesh:
    """A structured tetrahedral block with zero-based connectivity."""

    nodes: np.ndarray
    tetrahedra: np.ndarray
    bottom_nodes: np.ndarray
    top_nodes: np.ndarray
    top_faces: np.ndarray
    divisions: tuple[int, int, int]


@dataclass(frozen=True)
class SphereMesh:
    """A triangulated lower spherical cap with outward-facing triangles."""

    nodes: np.ndarray
    triangles: np.ndarray
    center: np.ndarray


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    split: str
    material: Mapping[str, float]
    load: Mapping[str, float]
    mesh: Mapping[str, int]

    def canonical_parameters(self) -> dict[str, Any]:
        return {
            "load": dict(self.load),
            "material": dict(self.material),
            "mesh": dict(self.mesh),
        }


def _require_finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def _require_int_at_least(value: Any, minimum: int, name: str) -> int:
    result = int(value)
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _case_id(parameters: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(parameters).encode("utf-8")).hexdigest()[:12]
    return f"hc3d-{digest}"


def generate_block_mesh(
    size: Sequence[float], divisions: Sequence[int]
) -> BlockMesh:
    """Create a conforming six-tetrahedra-per-voxel block mesh.

    The block occupies ``[-Lx/2,Lx/2] x [-Ly/2,Ly/2] x [0,Lz]``.
    Tetrahedra are reoriented to have strictly positive reference volume.
    """

    if len(size) != 3 or len(divisions) != 3:
        raise ValueError("size and divisions must each contain exactly three entries")
    lx, ly, lz = (_require_finite_positive(v, f"size[{i}]") for i, v in enumerate(size))
    nx, ny, nz = (
        _require_int_at_least(v, 1, f"divisions[{i}]") for i, v in enumerate(divisions)
    )

    xs = np.linspace(-0.5 * lx, 0.5 * lx, nx + 1, dtype=np.float64)
    ys = np.linspace(-0.5 * ly, 0.5 * ly, ny + 1, dtype=np.float64)
    zs = np.linspace(0.0, lz, nz + 1, dtype=np.float64)
    nodes = np.asarray(
        [(x, y, z) for z in zs for y in ys for x in xs], dtype=np.float64
    )

    def node_id(i: int, j: int, k: int) -> int:
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    tetrahedra: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n000 = node_id(i, j, k)
                n100 = node_id(i + 1, j, k)
                n010 = node_id(i, j + 1, k)
                n110 = node_id(i + 1, j + 1, k)
                n001 = node_id(i, j, k + 1)
                n101 = node_id(i + 1, j, k + 1)
                n011 = node_id(i, j + 1, k + 1)
                n111 = node_id(i + 1, j + 1, k + 1)
                voxel_tets = [
                    [n000, n100, n110, n111],
                    [n000, n110, n010, n111],
                    [n000, n010, n011, n111],
                    [n000, n011, n001, n111],
                    [n000, n001, n101, n111],
                    [n000, n101, n100, n111],
                ]
                for tet in voxel_tets:
                    x = nodes[np.asarray(tet)]
                    signed_six_volume = np.linalg.det(
                        np.stack((x[1] - x[0], x[2] - x[0], x[3] - x[0]), axis=1)
                    )
                    if signed_six_volume < 0.0:
                        tet[2], tet[3] = tet[3], tet[2]
                        signed_six_volume = -signed_six_volume
                    if signed_six_volume <= np.finfo(np.float64).eps:
                        raise RuntimeError("generated a degenerate tetrahedron")
                    tetrahedra.append(tet)

    bottom = np.flatnonzero(np.isclose(nodes[:, 2], 0.0)).astype(np.int64)
    top = np.flatnonzero(np.isclose(nodes[:, 2], lz)).astype(np.int64)
    top_set = set(int(node) for node in top)
    # CalculiX C3D4 face labels: S1=(1,2,3), S2=(1,4,2),
    # S3=(2,4,3), S4=(3,4,1). Store one-based element and face labels.
    local_faces = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
    top_faces: list[tuple[int, int]] = []
    for element_index, tet in enumerate(tetrahedra, start=1):
        for face_label, local_nodes in enumerate(local_faces, start=1):
            if all(int(tet[local]) in top_set for local in local_nodes):
                top_faces.append((element_index, face_label))
    expected_top_faces = 2 * nx * ny
    if len(top_faces) != expected_top_faces:
        raise RuntimeError(
            f"expected {expected_top_faces} top faces, found {len(top_faces)}"
        )
    return BlockMesh(
        nodes=nodes,
        tetrahedra=np.asarray(tetrahedra, dtype=np.int64),
        bottom_nodes=bottom,
        top_nodes=top,
        top_faces=np.asarray(top_faces, dtype=np.int64),
        divisions=(nx, ny, nz),
    )


def generate_spherical_cap(
    center: Sequence[float],
    radius: float,
    rings: int,
    segments: int,
    cap_angle_degrees: float,
) -> SphereMesh:
    """Triangulate the lower cap of a sphere.

    Triangle winding is chosen so the positive shell normal points away from
    the sphere center.  Consequently the cap's contact-facing pole has a
    downward normal.
    """

    center_array = np.asarray(center, dtype=np.float64)
    if center_array.shape != (3,) or not np.isfinite(center_array).all():
        raise ValueError("center must contain three finite coordinates")
    radius = _require_finite_positive(radius, "radius")
    rings = _require_int_at_least(rings, 1, "rings")
    segments = _require_int_at_least(segments, 3, "segments")
    cap_angle_degrees = _require_finite_positive(cap_angle_degrees, "cap_angle_degrees")
    if cap_angle_degrees >= 90.0:
        raise ValueError("cap_angle_degrees must be less than 90 degrees")

    points: list[np.ndarray] = [center_array + np.asarray([0.0, 0.0, -radius])]
    max_angle = math.radians(cap_angle_degrees)
    for ring in range(1, rings + 1):
        polar = max_angle * ring / rings
        radial = radius * math.sin(polar)
        z = center_array[2] - radius * math.cos(polar)
        for segment in range(segments):
            azimuth = 2.0 * math.pi * segment / segments
            points.append(
                np.asarray(
                    [
                        center_array[0] + radial * math.cos(azimuth),
                        center_array[1] + radial * math.sin(azimuth),
                        z,
                    ],
                    dtype=np.float64,
                )
            )
    nodes = np.asarray(points, dtype=np.float64)

    def ring_node(ring: int, segment: int) -> int:
        if ring == 0:
            return 0
        return 1 + (ring - 1) * segments + segment % segments

    triangles: list[list[int]] = []
    for segment in range(segments):
        triangles.append([0, ring_node(1, segment), ring_node(1, segment + 1)])
    for ring in range(1, rings):
        for segment in range(segments):
            a = ring_node(ring, segment)
            b = ring_node(ring, segment + 1)
            c = ring_node(ring + 1, segment)
            d = ring_node(ring + 1, segment + 1)
            triangles.extend(([a, c, d], [a, d, b]))

    for triangle in triangles:
        p = nodes[np.asarray(triangle)]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        outward = p.mean(axis=0) - center_array
        if float(np.dot(normal, outward)) < 0.0:
            triangle[1], triangle[2] = triangle[2], triangle[1]
        elif np.linalg.norm(normal) <= np.finfo(np.float64).eps:
            raise RuntimeError("generated a degenerate indenter triangle")

    return SphereMesh(
        nodes=nodes,
        triangles=np.asarray(triangles, dtype=np.int64),
        center=center_array,
    )


def _format_ids(ids: Iterable[int], width: int = 16) -> list[str]:
    values = list(ids)
    return [", ".join(str(value) for value in values[i : i + width]) for i in range(0, len(values), width)]


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _material_d1(c10_pa: float, poisson_ratio: float) -> float:
    # Neo-Hookean small-strain equivalence: mu=2*C10 and D1=2/K.
    shear_modulus = 2.0 * c10_pa
    bulk_modulus = 2.0 * shear_modulus * (1.0 + poisson_ratio) / (
        3.0 * (1.0 - 2.0 * poisson_ratio)
    )
    return 2.0 / bulk_modulus


def render_calculix_deck(
    case: BenchmarkCase,
    config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Render one self-contained CalculiX input deck and its mesh metadata."""

    geometry = config["geometry"]
    solver = config["solver"]
    size = tuple(float(v) for v in geometry["block_size_m"])
    divisions = (int(case.mesh["nx"]), int(case.mesh["ny"]), int(case.mesh["nz"]))
    block = generate_block_mesh(size, divisions)

    radius = float(geometry["indenter_radius_m"])
    gap = float(geometry["initial_gap_m"])
    offset_x = float(case.load.get("offset_x_m", 0.0))
    offset_y = float(case.load.get("offset_y_m", 0.0))
    center = np.asarray([offset_x, offset_y, size[2] + gap + radius], dtype=np.float64)
    sphere = generate_spherical_cap(
        center=center,
        radius=radius,
        rings=int(geometry["indenter_rings"]),
        segments=int(geometry["indenter_segments"]),
        cap_angle_degrees=float(geometry["indenter_cap_angle_degrees"]),
    )

    c10 = float(case.material["c10_pa"])
    poisson = float(case.material["poisson_ratio"])
    density = float(case.material["density_kg_m3"])
    indenter_density = float(geometry.get("indenter_density_kg_m3", 7800.0))
    d1 = _material_d1(c10, poisson)
    min_spacing = min(
        size[0] / divisions[0], size[1] / divisions[1], size[2] / divisions[2]
    )
    # A linear pressure-overclosure slope has units pressure / length. Scale
    # the adjacent Young's modulus by the local inverse mesh length so the
    # input remains dimensionally correct in SI units.
    young_modulus = 4.0 * c10 * (1.0 + poisson)
    penalty_factor = float(solver["contact_penalty_factor"])
    penalty = penalty_factor * young_modulus / min_spacing
    indentation = float(case.load["indentation_m"])
    imposed_displacement = -(gap + indentation)

    block_node_count = block.nodes.shape[0]
    sphere_node_count = sphere.nodes.shape[0]
    solid_element_count = block.tetrahedra.shape[0]
    first_shell_element = solid_element_count + 1

    lines = [
        "*HEADING",
        f"HyperContact-3D {case.case_id}",
        f"** schema_version={SCHEMA_VERSION}",
        f"** generator_version={GENERATOR_VERSION}",
        f"** split={case.split}",
        f"** parameters={_canonical_json(case.canonical_parameters())}",
        "*NODE, NSET=BLOCK_NODES",
    ]
    for index, xyz in enumerate(block.nodes, start=1):
        lines.append(f"{index}, {_fmt(xyz[0])}, {_fmt(xyz[1])}, {_fmt(xyz[2])}")

    lines.append("*NODE, NSET=INDENTER_NODES")
    sphere_node_offset = block_node_count
    for local_index, xyz in enumerate(sphere.nodes, start=1):
        index = sphere_node_offset + local_index
        lines.append(f"{index}, {_fmt(xyz[0])}, {_fmt(xyz[1])}, {_fmt(xyz[2])}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=BLOCK")
    for index, tet in enumerate(block.tetrahedra, start=1):
        connectivity = ", ".join(str(int(node) + 1) for node in tet)
        lines.append(f"{index}, {connectivity}")

    lines.append("*ELEMENT, TYPE=S3, ELSET=INDENTER")
    for local_index, triangle in enumerate(sphere.triangles):
        index = first_shell_element + local_index
        connectivity = ", ".join(
            str(sphere_node_offset + int(node) + 1) for node in triangle
        )
        lines.append(f"{index}, {connectivity}")

    lines.append("*NSET, NSET=BOTTOM")
    lines.extend(_format_ids(int(node) + 1 for node in block.bottom_nodes))
    lines.append("*NSET, NSET=BLOCK_TOP")
    lines.extend(_format_ids(int(node) + 1 for node in block.top_nodes))
    lines.append("*SURFACE, NAME=BLOCK_CONTACT, TYPE=ELEMENT")
    lines.extend(
        f"{int(element)}, S{int(face)}" for element, face in block.top_faces
    )
    lines.extend(
        [
            "*MATERIAL, NAME=BLOCK_MAT",
            "*DENSITY",
            _fmt(density),
            "*HYPERELASTIC, NEO HOOKE",
            f"{_fmt(c10)}, {_fmt(d1)}",
            "*SOLID SECTION, ELSET=BLOCK, MATERIAL=BLOCK_MAT",
            "*MATERIAL, NAME=INDENTER_MAT",
            "*DENSITY",
            _fmt(indenter_density),
            "*ELASTIC",
            "2.1e11, 0.3",
            "*SHELL SECTION, ELSET=INDENTER, MATERIAL=INDENTER_MAT",
            _fmt(float(geometry["indenter_shell_thickness_m"])),
            "*SURFACE, NAME=INDENTER_CONTACT, TYPE=ELEMENT",
            "INDENTER, SPOS",
            "*SURFACE INTERACTION, NAME=CONTACT_PROPERTY",
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
            _fmt(penalty),
        ]
    )
    friction = float(solver.get("friction_coefficient", 0.0))
    if friction > 0.0:
        lines.extend(["*FRICTION", _fmt(friction)])
    lines.extend(
        [
            "*CONTACT PAIR, INTERACTION=CONTACT_PROPERTY, TYPE=SURFACE TO SURFACE",
            "INDENTER_CONTACT, BLOCK_CONTACT",
            "*BOUNDARY",
            "BOTTOM, 1, 3, 0.",
            "INDENTER_NODES, 1, 2, 0.",
            "*STEP, NLGEOM",
            "*STATIC",
            (
                f"{_fmt(1.0 / int(solver['target_increments']))}, 1., "
                f"{_fmt(float(solver['minimum_increment']))}, "
                f"{_fmt(1.0 / int(solver['target_increments']))}"
            ),
            "*BOUNDARY",
            f"INDENTER_NODES, 3, 3, {_fmt(imposed_displacement)}",
            "*NODE FILE, FREQUENCY=1, GLOBAL=YES",
            "U, RF",
            "*EL FILE, FREQUENCY=1",
            "S, E, ENER",
            "*EL PRINT, ELSET=BLOCK, FREQUENCY=1",
            "S",
            "*CONTACT FILE, FREQUENCY=1",
            "CDIS, CSTR",
            "*END STEP",
            "",
        ]
    )

    mesh_metadata = {
        "block_nodes": int(block_node_count),
        "block_tetrahedra": int(solid_element_count),
        "bottom_nodes": int(block.bottom_nodes.size),
        "top_nodes": int(block.top_nodes.size),
        "top_contact_faces": int(block.top_faces.shape[0]),
        "indenter_nodes": int(sphere_node_count),
        "indenter_triangles": int(sphere.triangles.shape[0]),
        "indenter_kinematics": "uniform prescribed translation on all surface nodes",
        "minimum_spacing_m": float(min_spacing),
    }
    derived = {
        "contact_formulation": "surface_to_surface",
        "contact_characteristic_length_m": float(min_spacing),
        "contact_penalty_factor": float(penalty_factor),
        "contact_penalty_stiffness_pa_per_m": float(penalty),
        "d1_pa_inverse": float(d1),
        "imposed_indenter_displacement_m": float(imposed_displacement),
        "indenter_density_kg_m3": float(indenter_density),
        "indenter_shell_thickness_m": float(
            geometry["indenter_shell_thickness_m"]
        ),
        "step_duration": 1.0,
    }
    return "\n".join(lines), {"derived": derived, "mesh_statistics": mesh_metadata}


def _validate_axis_entries(config: Mapping[str, Any], axis: str) -> None:
    required_categories = ("train", "interpolation", "ood")
    entries = config["parameter_grid"].get(axis)
    if not isinstance(entries, Mapping):
        raise ValueError(f"parameter_grid.{axis} must be a mapping")
    seen: set[str] = set()
    for category in required_categories:
        values = entries.get(category)
        if not isinstance(values, list) or not values:
            raise ValueError(f"parameter_grid.{axis}.{category} must be a non-empty list")
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError(f"each {axis} entry must be a mapping")
            canonical = _canonical_json(_canonical_axis_parameters(axis, value))
            if canonical in seen:
                raise ValueError(f"duplicate {axis} value across grid categories: {value}")
            seen.add(canonical)


def _canonical_axis_parameters(
    axis: str,
    value: Mapping[str, Any],
) -> dict[str, float | int]:
    if axis in {"material", "load"}:
        return {str(key): float(item) for key, item in value.items()}
    if axis == "mesh":
        return {str(key): int(item) for key, item in value.items()}
    raise ValueError(f"unknown HyperContact parameter axis: {axis}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate all inputs that influence deck validity or split semantics."""

    for section in ("benchmark", "geometry", "solver", "parameter_grid"):
        if section not in config:
            raise ValueError(f"missing required config section: {section}")
    geometry = config["geometry"]
    if len(geometry.get("block_size_m", [])) != 3:
        raise ValueError("geometry.block_size_m must contain three values")
    for index, value in enumerate(geometry["block_size_m"]):
        _require_finite_positive(value, f"geometry.block_size_m[{index}]")
    _require_finite_positive(geometry["indenter_radius_m"], "geometry.indenter_radius_m")
    _require_finite_positive(geometry["initial_gap_m"], "geometry.initial_gap_m")
    _require_finite_positive(
        geometry["indenter_shell_thickness_m"], "geometry.indenter_shell_thickness_m"
    )
    _require_finite_positive(
        geometry.get("indenter_density_kg_m3", 7800.0),
        "geometry.indenter_density_kg_m3",
    )
    _require_int_at_least(geometry["indenter_rings"], 1, "geometry.indenter_rings")
    _require_int_at_least(geometry["indenter_segments"], 3, "geometry.indenter_segments")
    cap_angle = float(geometry["indenter_cap_angle_degrees"])
    if not 0.0 < cap_angle < 90.0:
        raise ValueError("geometry.indenter_cap_angle_degrees must lie in (0, 90)")

    solver = config["solver"]
    _require_finite_positive(solver["contact_penalty_factor"], "solver.contact_penalty_factor")
    penalty_factor = float(solver["contact_penalty_factor"])
    if not 5.0 <= penalty_factor <= 50.0:
        raise ValueError("solver.contact_penalty_factor must lie in the recommended [5, 50]")
    if solver.get("contact_formulation") != "surface_to_surface":
        raise ValueError(
            "solver.contact_formulation must be surface_to_surface to avoid "
            "mesh-dependent missed contact"
        )
    _require_int_at_least(solver["target_increments"], 1, "solver.target_increments")
    _require_finite_positive(solver["minimum_increment"], "solver.minimum_increment")
    friction = float(solver.get("friction_coefficient", 0.0))
    if not 0.0 <= friction < 1.0:
        raise ValueError("solver.friction_coefficient must lie in [0, 1)")
    holdout = float(config["benchmark"].get("id_test_fraction", 0.2))
    if not 0.0 <= holdout < 1.0:
        raise ValueError("benchmark.id_test_fraction must lie in [0, 1)")

    for axis in ("material", "load", "mesh"):
        _validate_axis_entries(config, axis)
    for category in ("train", "interpolation", "ood"):
        for material in config["parameter_grid"]["material"][category]:
            _require_finite_positive(material["c10_pa"], "material.c10_pa")
            _require_finite_positive(material["density_kg_m3"], "material.density_kg_m3")
            poisson = float(material["poisson_ratio"])
            if not -0.99 < poisson < 0.499:
                raise ValueError("material.poisson_ratio must lie in (-0.99, 0.499)")
        for load in config["parameter_grid"]["load"][category]:
            indentation = _require_finite_positive(load["indentation_m"], "load.indentation_m")
            if indentation >= float(geometry["indenter_radius_m"]):
                raise ValueError("load.indentation_m must be smaller than the indenter radius")
            for coordinate in ("offset_x_m", "offset_y_m"):
                value = float(load.get(coordinate, 0.0))
                if not math.isfinite(value):
                    raise ValueError(f"load.{coordinate} must be finite")
        for mesh in config["parameter_grid"]["mesh"][category]:
            for coordinate in ("nx", "ny", "nz"):
                _require_int_at_least(mesh[coordinate], 1, f"mesh.{coordinate}")


def _split_for_categories(categories: Mapping[str, str]) -> str:
    ood_axes = sorted(axis for axis, category in categories.items() if category == "ood")
    if len(ood_axes) > 1:
        return "test_ood_combined"
    if len(ood_axes) == 1:
        return f"test_ood_{ood_axes[0]}"
    if any(category == "interpolation" for category in categories.values()):
        return "validation"
    return "train"


def enumerate_cases(config: Mapping[str, Any]) -> list[BenchmarkCase]:
    """Expand and deterministically split the Cartesian parameter grid."""

    validate_config(config)
    grid = config["parameter_grid"]
    labelled: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for axis in ("material", "load", "mesh"):
        labelled[axis] = [
            (category, value)
            for category in ("train", "interpolation", "ood")
            for value in grid[axis][category]
        ]

    provisional: list[BenchmarkCase] = []
    for material_entry, load_entry, mesh_entry in itertools.product(
        labelled["material"], labelled["load"], labelled["mesh"]
    ):
        categories = {
            "material": material_entry[0],
            "load": load_entry[0],
            "mesh": mesh_entry[0],
        }
        parameters = {
            "material": _canonical_axis_parameters("material", material_entry[1]),
            "load": _canonical_axis_parameters("load", load_entry[1]),
            "mesh": _canonical_axis_parameters("mesh", mesh_entry[1]),
        }
        provisional.append(
            BenchmarkCase(
                case_id=_case_id(parameters),
                split=_split_for_categories(categories),
                material=parameters["material"],
                load=parameters["load"],
                mesh=parameters["mesh"],
            )
        )

    # Exact-size, hash-ranked ID holdout avoids RNG/library-version dependence.
    train_indices = [index for index, case in enumerate(provisional) if case.split == "train"]
    fraction = float(config["benchmark"].get("id_test_fraction", 0.2))
    holdout_count = min(max(0, round(len(train_indices) * fraction)), max(0, len(train_indices) - 1))
    split_seed = str(config["benchmark"].get("split_seed", 0))
    ranked = sorted(
        train_indices,
        key=lambda index: hashlib.sha256(
            f"{split_seed}:{provisional[index].case_id}".encode("utf-8")
        ).hexdigest(),
    )
    for index in ranked[:holdout_count]:
        case = provisional[index]
        provisional[index] = BenchmarkCase(
            case_id=case.case_id,
            split="test_id",
            material=case.material,
            load=case.load,
            mesh=case.mesh,
        )
    return sorted(provisional, key=lambda case: case.case_id)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("benchmark config must contain a YAML mapping")
    validate_config(config)
    return config


def generate_benchmark(
    config: Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Write all decks plus canonical ``manifest.json`` and ``splits.json``."""

    config_dict = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    validate_config(config_dict)
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is not empty: {output_path}; pass force=True to replace generated files"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    cases = enumerate_cases(config_dict)
    manifest_cases: list[dict[str, Any]] = []
    splits: dict[str, list[str]] = {}

    for case in cases:
        case_directory = output_path / "cases" / case.case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        deck_path = case_directory / "model.inp"
        if deck_path.exists() and not force:
            raise FileExistsError(f"refusing to replace existing deck: {deck_path}")
        deck, metadata = render_calculix_deck(case, config_dict)
        deck_path.write_text(deck, encoding="utf-8", newline="\n")
        deck_sha256 = hashlib.sha256(deck.encode("utf-8")).hexdigest()
        relative_deck = deck_path.relative_to(output_path).as_posix()
        entry = {
            "case_id": case.case_id,
            "deck": relative_deck,
            "deck_sha256": deck_sha256,
            "derived": metadata["derived"],
            "expected_outputs": [
                f"cases/{case.case_id}/model.frd",
                f"cases/{case.case_id}/model.dat",
                f"cases/{case.case_id}/model.sta",
            ],
            "mesh_statistics": metadata["mesh_statistics"],
            "parameters": case.canonical_parameters(),
            "run_command": f"ccx -i cases/{case.case_id}/model",
            "split": case.split,
        }
        (case_directory / "case.json").write_text(
            json.dumps(entry, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest_cases.append(entry)
        splits.setdefault(case.split, []).append(case.case_id)

    for case_ids in splits.values():
        case_ids.sort()
    config_sha256 = hashlib.sha256(_canonical_json(config_dict).encode("utf-8")).hexdigest()
    manifest = {
        "benchmark": str(config_dict["benchmark"].get("name", "HyperContact-3D")),
        "case_count": len(manifest_cases),
        "cases": manifest_cases,
        "config_sha256": config_sha256,
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "solver": {
            "minimum_version": str(config_dict["solver"].get("minimum_version", "2.18")),
            "name": "CalculiX CrunchiX",
        },
        "split_counts": {key: len(value) for key, value in sorted(splits.items())},
        "units": "SI",
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_path / "splits.json").write_text(
        json.dumps(dict(sorted(splits.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_path / "generation_config.yaml").write_text(
        yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="HyperContact-3D YAML config")
    parser.add_argument("--output", type=Path, required=True, help="output benchmark directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace files generated for the same deterministic cases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_benchmark(args.config, args.output, force=args.force)
    split_summary = ", ".join(
        f"{name}={count}" for name, count in manifest["split_counts"].items()
    )
    print(f"Generated {manifest['case_count']} HyperContact-3D cases ({split_summary})")


if __name__ == "__main__":
    main()
