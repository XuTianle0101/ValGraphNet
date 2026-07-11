import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.generate_hypercontact3d import (
    enumerate_cases,
    generate_benchmark,
    generate_block_mesh,
    generate_spherical_cap,
    render_calculix_deck,
    validate_config,
)


def _tiny_config():
    return {
        "benchmark": {
            "name": "HyperContact-3D-test",
            "split_seed": 17,
            "id_test_fraction": 0.25,
        },
        "geometry": {
            "block_size_m": [0.04, 0.04, 0.012],
            "indenter_radius_m": 0.008,
            "initial_gap_m": 0.0002,
            "indenter_rings": 2,
            "indenter_segments": 8,
            "indenter_cap_angle_degrees": 65.0,
            "indenter_shell_thickness_m": 0.0001,
        },
        "solver": {
            "minimum_version": "2.18",
            "target_increments": 20,
            "minimum_increment": 1e-7,
            "contact_penalty_factor": 50.0,
            "contact_tension_fraction": 0.0025,
            "contact_search_factor": 0.05,
            "friction_coefficient": 0.0,
        },
        "parameter_grid": {
            "material": {
                "train": [
                    {"c10_pa": 2e5, "poisson_ratio": 0.45, "density_kg_m3": 1000.0},
                    {"c10_pa": 4e5, "poisson_ratio": 0.45, "density_kg_m3": 1000.0},
                ],
                "interpolation": [
                    {"c10_pa": 3e5, "poisson_ratio": 0.45, "density_kg_m3": 1000.0}
                ],
                "ood": [
                    {"c10_pa": 8e5, "poisson_ratio": 0.40, "density_kg_m3": 1100.0}
                ],
            },
            "load": {
                "train": [
                    {"indentation_m": 0.001, "offset_x_m": 0.0, "offset_y_m": 0.0},
                    {"indentation_m": 0.002, "offset_x_m": 0.0, "offset_y_m": 0.0},
                ],
                "interpolation": [
                    {"indentation_m": 0.0015, "offset_x_m": 0.001, "offset_y_m": 0.0}
                ],
                "ood": [
                    {"indentation_m": 0.003, "offset_x_m": 0.0, "offset_y_m": 0.0}
                ],
            },
            "mesh": {
                "train": [{"nx": 2, "ny": 2, "nz": 1}],
                "interpolation": [{"nx": 3, "ny": 3, "nz": 2}],
                "ood": [{"nx": 4, "ny": 4, "nz": 2}],
            },
        },
    }


def test_block_mesh_has_expected_counts_and_positive_orientations():
    mesh = generate_block_mesh((2.0, 1.0, 0.5), (2, 3, 1))

    assert mesh.nodes.shape == ((2 + 1) * (3 + 1) * (1 + 1), 3)
    assert mesh.tetrahedra.shape == (2 * 3 * 1 * 6, 4)
    assert mesh.bottom_nodes.size == (2 + 1) * (3 + 1)
    assert mesh.top_nodes.size == (2 + 1) * (3 + 1)
    coordinates = mesh.nodes[mesh.tetrahedra]
    signed_six_volume = np.linalg.det(
        np.stack(
            (
                coordinates[:, 1] - coordinates[:, 0],
                coordinates[:, 2] - coordinates[:, 0],
                coordinates[:, 3] - coordinates[:, 0],
            ),
            axis=2,
        )
    )
    assert np.all(signed_six_volume > 0.0)
    np.testing.assert_allclose(signed_six_volume.sum() / 6.0, 2.0 * 1.0 * 0.5)


def test_spherical_cap_normals_are_outward_and_pole_normal_is_downward():
    sphere = generate_spherical_cap(
        center=(0.1, -0.2, 1.0),
        radius=0.5,
        rings=3,
        segments=12,
        cap_angle_degrees=70.0,
    )

    assert sphere.nodes.shape == (1 + 3 * 12, 3)
    assert sphere.triangles.shape == (12 + 2 * 12 * (3 - 1), 3)
    coordinates = sphere.nodes[sphere.triangles]
    normals = np.cross(
        coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]
    )
    outward = coordinates.mean(axis=1) - sphere.center
    assert np.all(np.einsum("ij,ij->i", normals, outward) > 0.0)
    assert normals[0, 2] < 0.0


def test_deck_contains_hyperelastic_contact_rigid_body_and_full_outputs():
    config = _tiny_config()
    case = enumerate_cases(config)[0]

    deck, metadata = render_calculix_deck(case, config)

    required_cards = (
        "*ELEMENT, TYPE=C3D4, ELSET=BLOCK",
        "*HYPERELASTIC, NEO HOOKE",
        "*RIGID BODY, ELSET=INDENTER",
        "*SURFACE, NAME=BLOCK_CONTACT, TYPE=NODE",
        "*SURFACE, NAME=INDENTER_CONTACT, TYPE=ELEMENT",
        "*CONTACT PAIR, INTERACTION=CONTACT_PROPERTY, TYPE=NODE TO SURFACE",
        "*STEP, NLGEOM",
        "*NODE FILE, FREQUENCY=1, GLOBAL=YES",
        "*EL FILE, FREQUENCY=1",
        "S, E, ENER",
        "*CONTACT FILE, FREQUENCY=1",
    )
    for card in required_cards:
        assert card in deck
    assert metadata["mesh_statistics"]["block_tetrahedra"] == 6 * 2 * 2 * 1
    assert metadata["derived"]["d1_pa_inverse"] > 0.0
    assert metadata["derived"]["contact_tension_cutoff_pa"] > 0.0
    assert metadata["derived"]["imposed_indenter_displacement_m"] < 0.0
    assert "BLOCK_CONTACT, INDENTER_CONTACT" in deck
    in_element_block = False
    for line in deck.splitlines():
        if line.startswith("*ELEMENT"):
            in_element_block = True
            continue
        if line.startswith("*"):
            in_element_block = False
        elif in_element_block:
            assert all(int(value.strip()) > 0 for value in line.split(","))


def test_grid_has_explicit_id_interpolation_and_each_ood_split():
    cases = enumerate_cases(_tiny_config())
    split_names = {case.split for case in cases}

    assert split_names == {
        "train",
        "validation",
        "test_id",
        "test_ood_material",
        "test_ood_load",
        "test_ood_mesh",
        "test_ood_combined",
    }
    assert len({case.case_id for case in cases}) == len(cases)


def test_generation_is_deterministic_and_manifest_decks_are_self_consistent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest_first = generate_benchmark(_tiny_config(), first)
    manifest_second = generate_benchmark(_tiny_config(), second)

    assert manifest_first == manifest_second
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "splits.json").read_bytes() == (second / "splits.json").read_bytes()
    splits = json.loads((first / "splits.json").read_text(encoding="utf-8"))
    assert sum(map(len, splits.values())) == manifest_first["case_count"]
    for entry in manifest_first["cases"]:
        deck = first / entry["deck"]
        case_metadata = deck.parent / "case.json"
        assert deck.is_file()
        assert case_metadata.is_file()
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == entry["deck_sha256"]
        assert json.loads(case_metadata.read_text(encoding="utf-8"))["case_id"] == entry["case_id"]
        assert entry["case_id"] in splits[entry["split"]]


def test_nonempty_output_requires_explicit_force(tmp_path):
    output = tmp_path / "benchmark"
    generate_benchmark(_tiny_config(), output)

    with pytest.raises(FileExistsError, match="not empty"):
        generate_benchmark(_tiny_config(), output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("geometry", "indenter_radius_m", -1.0), "indenter_radius_m"),
        (("solver", "friction_coefficient", 1.2), "friction_coefficient"),
    ],
)
def test_invalid_global_config_is_rejected(mutation, message):
    config = _tiny_config()
    section, key, value = mutation
    config[section][key] = value

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_duplicate_axis_value_across_categories_is_rejected():
    config = _tiny_config()
    config["parameter_grid"]["mesh"]["ood"] = copy.deepcopy(
        config["parameter_grid"]["mesh"]["train"]
    )

    with pytest.raises(ValueError, match="duplicate mesh value"):
        validate_config(config)
