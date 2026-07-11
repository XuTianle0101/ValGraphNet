import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from valgraphnet.calculix_results import (
    convert_benchmark,
    convert_case,
    parse_ascii_frd,
    parse_dat_stress,
    parse_hypercontact_deck,
)
from valgraphnet.chp_model import build_chp_static, radius_contact_pairs
from valgraphnet.data.case import load_case
from valgraphnet.data.valve_ood import validate_case_requirements
from valgraphnet.hypercontact_solver import run_benchmark
from valgraphnet.mechanics import semi_implicit_step


FIXTURES = Path(__file__).parent / "fixtures" / "hypercontact"


def _raw_case(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "raw"
    case = root / "cases" / "hc3d-fixture"
    case.mkdir(parents=True)
    for name in ("model.inp", "model.frd"):
        shutil.copyfile(FIXTURES / name, case / name)
    shutil.copyfile(FIXTURES / "model.dat.txt", case / "model.dat")
    (case / "model.sta").write_text("analysis completed\n", encoding="utf-8")
    deck = case / "model.inp"
    entry = {
        "case_id": "hc3d-fixture",
        "deck": "cases/hc3d-fixture/model.inp",
        "deck_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
        "expected_outputs": [
            "cases/hc3d-fixture/model.frd",
            "cases/hc3d-fixture/model.dat",
            "cases/hc3d-fixture/model.sta",
        ],
        "parameters": {
            "material": {
                "c10_pa": 200000.0,
                "poisson_ratio": 0.45,
                "density_kg_m3": 1200.0,
            },
            "load": {"indentation_m": 0.001},
            "mesh": {"nx": 1, "ny": 1, "nz": 1},
        },
        "derived": {
            "imposed_indenter_displacement_m": -0.2,
            "indenter_density_kg_m3": 1000.0,
            "indenter_shell_thickness_m": 0.01,
            "step_duration": 1.0,
        },
        "split": "validation",
    }
    return root, entry


def test_ascii_frd_and_dat_parsers_preserve_tensor_components():
    mesh = parse_hypercontact_deck(FIXTURES / "model.inp")
    datasets = parse_ascii_frd(FIXTURES / "model.frd")
    stress = [dataset for dataset in datasets if dataset.name == "STRESS"]
    frames = parse_dat_stress(FIXTURES / "model.dat.txt")

    assert mesh.nodes.shape == (7, 3)
    np.testing.assert_array_equal(mesh.cells, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(mesh.indenter_triangles, [[4, 5, 6]])
    assert [dataset.time for dataset in stress] == [0.5, 1.0]
    displacement = [dataset for dataset in datasets if dataset.name == "DISP"]
    assert displacement[0].components == ("D1", "D2", "D3", "ALL")
    assert displacement[0].values[1].shape == (3,)
    contact = [dataset for dataset in datasets if dataset.name == "CONTACT"]
    assert len(contact) == 1
    assert contact[0].values == {}
    assert stress[0].components == ("SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX")
    np.testing.assert_allclose(frames[0].values[1], [[10, 20, 30, 40, 50, 60]])


def test_conversion_writes_loadable_constitutive_case(tmp_path):
    raw, entry = _raw_case(tmp_path)
    output = tmp_path / "processed" / entry["case_id"]

    metadata = convert_case(raw, entry, output)
    case = load_case(output)

    assert case.num_steps == 3
    assert case.num_nodes == 7
    assert case.num_cells == 1
    assert case.has_constitutive_data
    np.testing.assert_allclose(case.times, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(case.displacement[1, :4, 2], [0.0, 0.0, 0.0, -0.05])
    np.testing.assert_allclose(case.displacement[:, 4:, 2], [[0.0] * 3, [-0.1] * 3, [-0.2] * 3])
    np.testing.assert_allclose(
        case.stress[1, 0], [np.sqrt(234.0), 1, 2, 3, 4, 5, 6]
    )
    np.testing.assert_allclose(case.stress[:, 4:], 0.0)
    np.testing.assert_allclose(
        case.cell_stress[:, 0],
        [[0] * 6, [10, 20, 30, 40, 50, 60], [20, 40, 60, 80, 100, 120]],
    )
    np.testing.assert_allclose(case.integration_point_stress[1, 0, 0], [10, 20, 30, 40, 50, 60])
    np.testing.assert_allclose(case.density, [[1200.0]])
    np.testing.assert_allclose(case.lumped_mass[:4], np.full((4, 1), 50.0))
    np.testing.assert_allclose(case.lumped_mass[4:], np.full((3, 1), 1.0 / 60.0))
    np.testing.assert_array_equal(
        case.fixed_mask, [True, True, True, False, False, False, False]
    )
    np.testing.assert_array_equal(
        case.prescribed_mask, [False, False, False, False, True, True, True]
    )
    np.testing.assert_array_equal(case.pressure_mask, np.zeros(7, dtype=bool))
    np.testing.assert_array_equal(
        case.contact_surface_mask, [False, False, False, True, True, True, True]
    )
    assert case.mesh_edge_index.shape == (2, 18)
    edge_set = {tuple(edge) for edge in case.mesh_edge_index.T.tolist()}
    assert {(4, 5), (5, 4), (5, 6), (6, 5), (4, 6), (6, 4)} <= edge_set
    assert metadata["material_feature_names"] == [
        "c10_pa",
        "poisson_ratio",
        "density_kg_m3",
    ]
    material = json.loads((output / "material.json").read_text(encoding="utf-8"))
    assert material["material_feature_names"] == metadata["material_feature_names"]
    assert isinstance(material["c10_pa"], float)
    assert metadata["stress_source"].startswith("CalculiX DAT")
    assert metadata["prepended_zero_reference_frame"]
    assert "quasi-static" in metadata["time_semantics"]
    assert metadata["deck_sha256"] == entry["deck_sha256"]
    assert metadata["derived_solver_parameters"]["step_duration"] == 1.0


def test_explicit_contact_surface_and_prescribed_state_are_exact(tmp_path):
    raw, entry = _raw_case(tmp_path)
    output = tmp_path / "processed" / entry["case_id"]
    convert_case(raw, entry, output)
    case = load_case(output)
    static = build_chp_static(case, "cpu")
    displacement = torch.from_numpy(np.array(case.displacement, copy=True))
    velocity = torch.from_numpy(np.array(case.velocity, copy=True))
    position = static.reference_position + displacement[1]
    target_position = static.reference_position + displacement[2]

    result = semi_implicit_step(
        position,
        velocity[1],
        torch.ones_like(position),
        static.lumped_mass,
        float(case.times[2] - case.times[1]),
        fixed_mask=static.fixed_mask,
        prescribed_mask=static.prescribed_mask,
        prescribed_position=target_position,
        prescribed_velocity=velocity[2],
    )

    torch.testing.assert_close(
        static.contact_surface_mask,
        torch.tensor([False, False, False, True, True, True, True]),
    )
    torch.testing.assert_close(
        result.position[static.prescribed_mask],
        target_position[static.prescribed_mask],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.velocity[static.prescribed_mask],
        velocity[2][static.prescribed_mask],
        rtol=0.0,
        atol=0.0,
    )
    validate_case_requirements(
        case,
        {
            "data": {
                "requirements": {
                    "num_frames": 3,
                    "explicit_contact_surface_mask": True,
                    "prescribed_contact_surface": True,
                    "zero_pressure_mask": True,
                    "time_semantics": (
                        "quasi-static normalized step time, not physical transient time"
                    ),
                }
            }
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_converted_contact_search_keeps_prescribed_free_surface_pairs(tmp_path):
    raw, entry = _raw_case(tmp_path)
    output = tmp_path / "processed" / entry["case_id"]
    convert_case(raw, entry, output)
    case = load_case(output)
    static = build_chp_static(case, "cuda")
    displacement = torch.from_numpy(np.array(case.displacement[1], copy=True)).cuda()
    position = static.reference_position + displacement

    pairs = radius_contact_pairs(
        position,
        static.mesh_edge_index,
        static.fixed_mask,
        0.2,
        max_neighbors=7,
        prescribed_mask=static.prescribed_mask,
        surface_mask=static.contact_surface_mask,
    )

    assert pairs.shape[1] > 0
    assert torch.all(static.prescribed_mask[pairs[0]] ^ static.prescribed_mask[pairs[1]])
    assert torch.all(
        static.contact_surface_mask[pairs[0]] & static.contact_surface_mask[pairs[1]]
    )


def test_strict_conversion_rejects_missing_dat_stress(tmp_path):
    raw, entry = _raw_case(tmp_path)
    (raw / "cases" / entry["case_id"] / "model.dat").write_text(
        "no integration point output\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no integration-point stress"):
        convert_case(raw, entry, tmp_path / "strict")

    metadata = convert_case(
        raw,
        entry,
        tmp_path / "fallback",
        require_dat_stress=False,
    )
    assert metadata["stress_source"].startswith("mean of extrapolated FRD")


def test_batch_runner_records_success_and_resumes(tmp_path):
    root = tmp_path / "benchmark"
    case = root / "cases" / "case-a"
    case.mkdir(parents=True)
    deck = case / "model.inp"
    deck.write_text("*HEADING\nfixture\n", encoding="utf-8")
    fake_solver = tmp_path / "fake_ccx.py"
    fake_solver.write_text(
        """from pathlib import Path
import os
import sys
base = sys.argv[sys.argv.index('-i') + 1]
for suffix in ('.frd', '.dat'):
    Path(base + suffix).write_text('solver output\\n', encoding='utf-8')
Path(base + '.sta').write_text(
    'SUMMARY OF JOB INFORMATION\\n'
    '  STEP INC ATT ITRS TOT TIME STEP TIME INC TIME\\n'
    '  1 100 1 2 0.100000E+01 0.100000E+01 0.100000E-01\\n',
    encoding='utf-8',
)
print('threads=' + os.environ['OMP_NUM_THREADS'])
print('Job finished')
""",
        encoding="utf-8",
    )
    entry = {
        "case_id": "case-a",
        "deck": "cases/case-a/model.inp",
        "deck_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
        "expected_outputs": [
            "cases/case-a/model.frd",
            "cases/case-a/model.dat",
            "cases/case-a/model.sta",
        ],
        "split": "train",
    }
    manifest = {"cases": [entry]}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    first = run_benchmark(
        manifest_path,
        [sys.executable, str(fake_solver)],
        workers=2,
        solver_threads=3,
    )
    second = run_benchmark(
        manifest_path,
        [sys.executable, str(fake_solver)],
        solver_threads=3,
    )

    assert first["counts"] == {"succeeded": 1}
    assert second["counts"] == {"skipped": 1}
    stdout = (case / "solver.stdout.log").read_text(encoding="utf-8")
    assert "threads=3" in stdout
    assert "Job finished" in stdout
    status = json.loads((case / "solver_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "succeeded"
    assert status["solver_threads"] == 3
    assert status["convergence"]["validated"]
    assert status["convergence"]["sta_step_time"] == 1.0


def test_batch_runner_rejects_rc_zero_partial_step(tmp_path):
    root = tmp_path / "benchmark"
    case = root / "cases" / "partial"
    case.mkdir(parents=True)
    deck = case / "model.inp"
    deck.write_text("*HEADING\npartial\n", encoding="utf-8")
    fake_solver = tmp_path / "partial_ccx.py"
    fake_solver.write_text(
        """from pathlib import Path
import sys
base = sys.argv[sys.argv.index('-i') + 1]
Path(base + '.frd').write_text('partial frd\\n', encoding='utf-8')
Path(base + '.dat').write_text('partial dat\\n', encoding='utf-8')
Path(base + '.sta').write_text(
    'SUMMARY OF JOB INFORMATION\\n'
    '1 50 1 2 0.500000E+00 0.500000E+00 0.100000E-01\\n',
    encoding='utf-8',
)
""",
        encoding="utf-8",
    )
    entry = {
        "case_id": "partial",
        "deck": "cases/partial/model.inp",
        "deck_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
        "derived": {"step_duration": 1.0},
        "expected_outputs": [
            "cases/partial/model.frd",
            "cases/partial/model.dat",
            "cases/partial/model.sta",
        ],
        "split": "train",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"cases": [entry]}), encoding="utf-8")

    summary = run_benchmark(manifest_path, [sys.executable, str(fake_solver)])

    assert summary["counts"] == {"failed": 1}
    result = summary["results"][0]
    assert "convergence evidence" in result["message"]
    assert result["convergence"]["sta_step_time"] == 0.5
    assert not result["convergence"]["validated"]


def test_batch_runner_rejects_modified_deck(tmp_path):
    root, entry = _raw_case(tmp_path)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"cases": [entry]}), encoding="utf-8")
    (root / entry["deck"]).write_text("*HEADING\nmodified\n", encoding="utf-8")

    summary = run_benchmark(manifest_path, [sys.executable], case_ids=[entry["case_id"]])

    assert summary["counts"] == {"failed": 1}
    assert "hash mismatch" in summary["results"][0]["message"]


def test_conversion_rejects_case_id_path_traversal_before_force_removal(tmp_path):
    raw, entry = _raw_case(tmp_path)
    entry["case_id"] = "../victim"
    manifest_path = raw / "manifest.json"
    manifest_path.write_text(json.dumps({"cases": [entry]}), encoding="utf-8")
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("owned by user", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe HyperContact case_id"):
        convert_benchmark(manifest_path, tmp_path / "processed", force=True)

    assert marker.read_text(encoding="utf-8") == "owned by user"
