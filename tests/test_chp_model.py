import numpy as np
import pytest
import torch

from valgraphnet.chp_model import (
    CHPGNS,
    CHPState,
    PairForceHeads,
    backtrack_tetrahedral_update,
    build_chp_static,
    radius_contact_pairs,
    tetrahedral_domain_penalty,
    tetra_surface_node_mask,
    unique_undirected_pairs,
)
from valgraphnet.data.case import ValveCase
from valgraphnet.mechanics import deformation_gradient


def _case() -> ValveCase:
    nodes = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    dm_inv = np.eye(3, dtype=np.float32)[None]
    gradients = np.asarray([[[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32)
    mesh = np.asarray(
        [[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]],
        dtype=np.int64,
    )
    zeros = np.zeros((2, 4, 3), dtype=np.float32)
    return ValveCase(
        case_id="unit",
        root=None,
        metadata={},
        nodes=nodes,
        elements=cells,
        times=np.asarray([0, 1], dtype=np.float32),
        pressure=np.zeros(2, dtype=np.float32),
        displacement=zeros,
        velocity=zeros,
        acceleration=zeros,
        stress=np.zeros((2, 4, 1), dtype=np.float32),
        fixed_mask=np.zeros(4, dtype=bool),
        prescribed_mask=np.zeros(4, dtype=bool),
        pressure_mask=np.zeros(4, dtype=bool),
        leaflet_id=np.zeros(4, dtype=np.int64),
        thickness=np.ones(4, dtype=np.float32),
        normals=np.zeros_like(nodes),
        nodal_area=np.ones(4, dtype=np.float32),
        mesh_edge_index=mesh,
        cells=cells,
        dm_inv=dm_inv,
        reference_volume=np.asarray([[1 / 6]], dtype=np.float32),
        shape_gradients=gradients,
        lumped_mass=np.full((4, 1), 1 / 24, dtype=np.float32),
        density=np.ones((1, 1), dtype=np.float32),
        material_features=np.zeros((1, 0), dtype=np.float32),
        fiber_direction=np.zeros((1, 3), dtype=np.float32),
    )


def _cfg():
    return {
        "contact": {"radius": 0.03},
        "model": {
            "scalar_dim": 16,
            "vector_dim": 4,
            "cell_dim": 8,
            "potential_order": 2,
            "contact_substeps": 2,
        },
    }


def test_reference_state_remains_static_and_has_zero_stress():
    static = build_chp_static(_case(), "cpu")
    model = CHPGNS(_cfg()).eval()
    state = CHPState(static.reference_position.clone(), torch.zeros_like(static.reference_position))
    output = model(static, state)

    torch.testing.assert_close(output.next_position, state.position, atol=1.0e-6, rtol=0)
    torch.testing.assert_close(output.next_velocity, state.velocity, atol=1.0e-6, rtol=0)
    torch.testing.assert_close(output.nodal_stress, torch.zeros_like(output.nodal_stress), atol=1.0e-6, rtol=0)
    assert output.legacy_predictions(state)["delta_u"].shape == (4, 3)


def test_residual_acceleration_preserves_force_contract():
    static = build_chp_static(_case(), "cpu")
    model = CHPGNS(_cfg()).eval()
    translated = static.reference_position + torch.tensor([0.01, 0.0, 0.0])
    state = CHPState(translated, torch.zeros_like(translated))
    output = model(static, state)
    effective_mass = (
        static.lumped_mass * model.log_mass_scale.detach().exp()
    )[:, None]
    recovered = output.residual_force / effective_mass
    torch.testing.assert_close(
        recovered.square().mean().sqrt(),
        output.energy_diagnostics["residual_norm"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.linalg.vector_norm(recovered, dim=1).max() <= 3.0e-3


def test_full_step_exports_are_kinematically_consistent_with_two_substeps():
    static = build_chp_static(_case(), "cpu")
    model = CHPGNS(_cfg()).eval()
    translated = static.reference_position + torch.tensor([0.01, 0.0, 0.0])
    velocity = torch.full_like(translated, 1.0e-3)
    state = CHPState(translated, velocity)
    output = model(static, state, dt=0.5)
    torch.testing.assert_close(
        output.next_position - state.position,
        0.5 * output.next_velocity,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        output.next_velocity - state.velocity,
        0.5 * output.acceleration,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_pair_force_scatter_has_exact_zero_resultant():
    force = torch.randn(3, 3)
    pairs = torch.tensor([[0, 0, 1], [1, 2, 3]])
    assembled = PairForceHeads.scatter_pair(force, pairs, 4)
    torch.testing.assert_close(assembled.sum(0), torch.zeros(3), atol=1.0e-6, rtol=0)


def test_reference_contact_gap_has_zero_force_and_dissipation():
    heads = PairForceHeads(4)
    reference = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    scalar = torch.zeros(2, 4)
    velocity = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    pairs = torch.tensor([[0], [1]])
    force, penetration, dissipation = heads.contact_force(
        scalar,
        reference,
        velocity,
        pairs,
        radius=0.03,
        reference_position=reference,
    )
    torch.testing.assert_close(force, torch.zeros_like(force))
    torch.testing.assert_close(penetration, torch.zeros_like(penetration))
    torch.testing.assert_close(dissipation, torch.zeros_like(dissipation))


def test_unique_undirected_pairs_removes_reverse_duplicates():
    edges = torch.tensor([[0, 1, 2, 3, 2], [1, 0, 3, 2, 2]])
    pairs = unique_undirected_pairs(edges, 4)
    assert torch.equal(pairs, torch.tensor([[0, 2], [1, 3]]))


def test_tetra_surface_mask_excludes_interior_vertex():
    cells = torch.tensor(
        [[4, 1, 2, 3], [0, 4, 2, 3], [0, 1, 4, 3], [0, 1, 2, 4]]
    )
    assert torch.equal(
        tetra_surface_node_mask(cells, 5),
        torch.tensor([True, True, True, True, False]),
    )


def test_tetrahedral_update_backtracks_inversion_but_keeps_prescribed_node():
    case = _case()
    current = torch.from_numpy(case.nodes.copy())
    proposed = current.clone()
    proposed[1, 0] = -1.0
    proposed[2, 1] = 1.1
    accepted, scale, valid = backtrack_tetrahedral_update(
        current,
        proposed,
        torch.from_numpy(case.cells),
        torch.from_numpy(case.dm_inv),
        prescribed_mask=torch.tensor([False, False, True, False]),
        minimum_j=1.0e-4,
    )
    assert valid
    assert 0.0 < float(scale) < 1.0
    torch.testing.assert_close(accepted[2], proposed[2])
    determinant = torch.linalg.det(
        deformation_gradient(
            accepted, torch.from_numpy(case.cells), torch.from_numpy(case.dm_inv)
        )
    )
    assert determinant.min() >= 1.0e-4


def test_tetrahedral_update_reports_infeasible_prescribed_motion():
    case = _case()
    current = torch.from_numpy(case.nodes.copy())
    proposed = current.clone()
    proposed[1, 0] = -1.0
    accepted, scale, valid = backtrack_tetrahedral_update(
        current,
        proposed,
        torch.from_numpy(case.cells),
        torch.from_numpy(case.dm_inv),
        prescribed_mask=torch.ones(4, dtype=torch.bool),
        minimum_j=1.0e-4,
    )
    assert not valid
    assert float(scale) == 0.0
    torch.testing.assert_close(accepted, proposed)
    determinant = torch.linalg.det(
        deformation_gradient(
            accepted, torch.from_numpy(case.cells), torch.from_numpy(case.dm_inv)
        )
    )
    assert determinant.min() < 0.0


def test_raw_tetrahedral_proposal_barrier_has_finite_gradient():
    case = _case()
    proposed = torch.from_numpy(case.nodes.copy()).requires_grad_(True)
    inverted = proposed.clone()
    inverted[1, 0] = -1.0
    penalty = tetrahedral_domain_penalty(
        inverted,
        torch.from_numpy(case.cells),
        torch.from_numpy(case.dm_inv),
        minimum_j=1.0e-4,
    )
    penalty.backward()
    assert penalty > 0.0
    assert proposed.grad is not None
    assert torch.isfinite(proposed.grad).all()
    assert proposed.grad.abs().sum() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_chp_model_cuda_forward_uses_gpu():
    static = build_chp_static(_case(), "cuda")
    model = CHPGNS(_cfg()).cuda().eval()
    state = CHPState(static.reference_position.clone(), torch.zeros_like(static.reference_position))
    output = model(static, state)
    assert output.next_position.is_cuda
    assert output.cell_stress_tensor.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_contact_search_keeps_only_cross_body_pairs():
    position = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.005, 0.005, 0.0], [0.006, 0.006, 0.0]],
        device="cuda",
    )
    pairs = radius_contact_pairs(
        position,
        torch.zeros((2, 0), dtype=torch.long, device="cuda"),
        torch.zeros(4, dtype=torch.bool, device="cuda"),
        0.03,
        max_neighbors=4,
        prescribed_mask=torch.tensor([True, True, False, False], device="cuda"),
    )
    prescribed = torch.tensor([True, True, False, False], device="cuda")
    assert pairs.shape[1] == 4
    assert torch.all(prescribed[pairs[0]] ^ prescribed[pairs[1]])
