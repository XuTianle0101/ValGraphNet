import numpy as np
import pytest
import torch

from valgraphnet.chp_model import (
    CHPGNS,
    CHPState,
    CellNodeBlock,
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


def test_chp_threads_matched_flat_processor_switch():
    hierarchical_cfg = _cfg()
    flat_cfg = _cfg()
    flat_cfg["model"]["use_topology_hierarchy"] = False
    hierarchical = CHPGNS(hierarchical_cfg)
    flat = CHPGNS(flat_cfg)

    assert hierarchical.processor.use_topology_hierarchy is True
    assert flat.processor.use_topology_hierarchy is False
    assert sum(parameter.numel() for parameter in flat.parameters()) == sum(
        parameter.numel() for parameter in hierarchical.parameters()
    )


def test_reference_state_remains_static_and_has_zero_stress():
    static = build_chp_static(_case(), "cpu")
    model = CHPGNS(_cfg()).eval()
    state = CHPState(static.reference_position.clone(), torch.zeros_like(static.reference_position))
    output = model(static, state)

    torch.testing.assert_close(output.next_position, state.position, atol=1.0e-6, rtol=0)
    torch.testing.assert_close(output.next_velocity, state.velocity, atol=1.0e-6, rtol=0)
    torch.testing.assert_close(output.nodal_stress, torch.zeros_like(output.nodal_stress), atol=1.0e-6, rtol=0)
    assert output.legacy_predictions(state)["delta_u"].shape == (4, 3)


def test_material_features_modulate_every_positive_potential_basis():
    cfg = _cfg()
    cfg["model"]["potential_ridge_terms"] = 5
    model = CHPGNS(cfg, material_dim=3)
    material = torch.tensor(
        [[2.0e5, 0.45, 1000.0], [9.0e5, 0.475, 1100.0]]
    )
    multipliers = model.material_coefficient_multipliers(material)

    assert set(multipliers) == {"i1", "i2", "j", "log_j", "ridge"}
    assert multipliers["i1"].shape == (2, model.potential.order)
    assert multipliers["i2"].shape == (2, model.potential.order)
    assert multipliers["j"].shape == (2, model.potential.order)
    assert multipliers["log_j"].shape == (2, 1)
    assert multipliers["ridge"].shape == (2, 5)
    assert all(torch.all(value > 0.0) for value in multipliers.values())
    for value in multipliers.values():
        torch.testing.assert_close(value, torch.ones_like(value), atol=2.0e-6, rtol=0.0)


def test_cell_node_incidence_block_is_rotation_equivariant():
    torch.manual_seed(7)
    block = CellNodeBlock(node_dim=8, vector_dim=3, cell_dim=5).eval()
    node = torch.randn(4, 8)
    vector = torch.randn(4, 3, 3)
    cell = torch.randn(1, 5)
    cells = torch.tensor([[0, 1, 2, 3]])
    position = torch.from_numpy(_case().nodes.copy())
    angle = torch.tensor(0.63)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        [
            torch.stack([cosine, -sine, torch.tensor(0.0)]),
            torch.stack([sine, cosine, torch.tensor(0.0)]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]
    )

    scalar_a, vector_a, cell_a = block(node, vector, cell, cells, position)
    scalar_b, vector_b, cell_b = block(
        node,
        vector @ rotation.T,
        cell,
        cells,
        position @ rotation.T + torch.tensor([2.0, -1.0, 0.5]),
    )

    torch.testing.assert_close(scalar_a, scalar_b, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(cell_a, cell_b, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        vector_a @ rotation.T, vector_b, atol=2.0e-5, rtol=2.0e-5
    )


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


def test_full_step_accumulates_two_semi_implicit_substeps():
    static = build_chp_static(_case(), "cpu")
    model = CHPGNS(_cfg()).eval()
    torch.nn.init.zeros_(model.residual_channel.weight)
    velocity = torch.zeros_like(static.reference_position)
    state = CHPState(static.reference_position.clone(), velocity)
    nodal_force = static.lumped_mass[:, None] * torch.tensor([2.0, 0.0, 0.0])
    output = model(static, state, dt=0.5, external_force=nodal_force)

    expected_velocity = torch.tensor([1.0, 0.0, 0.0]).expand_as(velocity)
    # For two symplectic-Euler substeps under constant acceleration a and
    # zero initial velocity: dx = (3/4) * dt^2 * a.
    expected_displacement = torch.tensor([0.375, 0.0, 0.0]).expand_as(velocity)
    torch.testing.assert_close(
        output.next_position - state.position, expected_displacement,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        output.next_velocity, expected_velocity,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        output.acceleration,
        torch.tensor([2.0, 0.0, 0.0]).expand_as(velocity),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_full_step_overwrites_fixed_and_prescribed_nodes_exactly():
    case = _case()
    case.fixed_mask[0] = True
    case.prescribed_mask[1] = True
    static = build_chp_static(case, "cpu")
    model = CHPGNS(_cfg()).eval()
    state = CHPState(
        static.reference_position.clone(), torch.zeros_like(static.reference_position)
    )
    target = static.reference_position.clone()
    target[1, 0] += 0.02
    output = model(static, state, dt=0.5, prescribed_position=target)

    torch.testing.assert_close(output.next_position[0], state.position[0], rtol=0, atol=0)
    torch.testing.assert_close(output.next_velocity[0], torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(output.next_position[1], target[1], rtol=0, atol=0)
    torch.testing.assert_close(
        output.next_velocity[1], (target[1] - state.position[1]) / 0.5, rtol=0, atol=0
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


def test_contact_tangent_is_dissipative_and_bounded_by_normal_force():
    heads = PairForceHeads(4, tangential_ratio_cap=0.5)
    for parameter in heads.parameters():
        torch.nn.init.zeros_(parameter)
    scalar = torch.zeros(2, 4)
    reference = torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    position = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    velocity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    pairs = torch.tensor([[0], [1]])

    force, penetration, dissipation = heads.contact_force(
        scalar,
        position,
        velocity,
        pairs,
        radius=0.03,
        reference_position=reference,
    )

    assert penetration > 0.0
    detached_force = force.detach()
    assert abs(float(detached_force[0, 1])) <= (
        0.5 * abs(float(detached_force[0, 0])) + 1.0e-7
    )
    torch.testing.assert_close(force.sum(0), torch.zeros(3), atol=1.0e-7, rtol=0.0)
    assert dissipation > 0.0
    mechanical_power = (force * velocity).sum()
    assert mechanical_power <= 0.0


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
def test_contact_pairs_are_refreshed_between_cuda_substeps(monkeypatch):
    static = build_chp_static(_case(), "cuda")
    model = CHPGNS(_cfg()).cuda().eval()
    state = CHPState(
        static.reference_position.clone(), torch.zeros_like(static.reference_position)
    )
    calls = []

    def fake_radius_search(position, *args, **kwargs):
        calls.append(position.detach().clone())
        return torch.zeros((2, 0), dtype=torch.long, device=position.device)

    monkeypatch.setattr("valgraphnet.chp_model.radius_contact_pairs", fake_radius_search)
    output = model(
        static,
        state,
        contact_pairs=torch.zeros((2, 0), dtype=torch.long, device="cuda"),
    )

    assert len(calls) == model.contact_substeps - 1
    assert output.energy_diagnostics["contact_pair_count"].item() == 0.0


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
