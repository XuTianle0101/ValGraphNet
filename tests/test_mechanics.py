import math

import pytest
import torch

from valgraphnet.mechanics import (
    AnalyticPotential,
    assemble_internal_force,
    deformation_gradient,
    invariants,
    negative_j_barrier,
    precompute_tetrahedra,
    project_cell_to_nodes,
    semi_implicit_step,
    von_mises,
)


def _unit_tetra(dtype=torch.float64, device="cpu"):
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
    return nodes, cells


def _rotation_z(angle, dtype=torch.float64, device="cpu"):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def test_reference_precomputation_and_identity_have_zero_stress_force():
    nodes, cells = _unit_tetra()
    reference = precompute_tetrahedra(nodes, cells, density=6.0)
    deformation = deformation_gradient(nodes, cells, reference.dm_inv)
    response = AnalyticPotential(fiber_order=1).double()(
        deformation, fiber_direction=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    force = assemble_internal_force(
        response.first_piola,
        cells,
        reference.volume,
        reference.shape_gradients,
        num_nodes=nodes.shape[0],
    )

    torch.testing.assert_close(deformation, torch.eye(3, dtype=nodes.dtype)[None])
    torch.testing.assert_close(
        reference.shape_gradients.sum(dim=1), torch.zeros(1, 3, dtype=nodes.dtype)
    )
    torch.testing.assert_close(reference.volume, torch.tensor([1.0 / 6.0], dtype=nodes.dtype))
    torch.testing.assert_close(reference.lumped_mass.sum(), torch.tensor(1.0, dtype=nodes.dtype))
    torch.testing.assert_close(
        response.energy_density,
        torch.zeros(1, dtype=nodes.dtype),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        response.first_piola,
        torch.zeros_like(response.first_piola),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(force, torch.zeros_like(force), atol=1.0e-12, rtol=0.0)


def test_uniform_stretch_produces_positive_isotropic_stress():
    potential = AnalyticPotential().double()
    stretch = 1.1 * torch.eye(3, dtype=torch.float64)[None]
    response = potential(stretch)
    diagonal = torch.diagonal(response.cauchy_stress[0])

    assert response.energy_density.item() > 0.0
    assert torch.all(diagonal > 0.0)
    torch.testing.assert_close(diagonal, diagonal[0].expand_as(diagonal), atol=1.0e-11, rtol=1.0e-11)
    off_diagonal = response.cauchy_stress[0] - torch.diag(diagonal)
    torch.testing.assert_close(off_diagonal, torch.zeros_like(off_diagonal), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        von_mises(response.cauchy_stress),
        torch.zeros(1, dtype=stretch.dtype),
        atol=1.0e-11,
        rtol=0.0,
    )


def test_isochoric_potential_has_nonzero_small_strain_tangent():
    potential = AnalyticPotential().double()
    shear = 1.0e-5
    deformation = torch.eye(3, dtype=torch.float64)[None]
    deformation[0, 0, 1] = shear
    response = potential(deformation)

    # Ibar-3 is already second order in infinitesimal strain.  A linear
    # invariant term is therefore required for a finite reference tangent;
    # starting at (Ibar-3)^2 would incorrectly make the shear modulus zero.
    tangent = response.cauchy_stress[0, 0, 1] / shear
    assert tangent.item() > 0.1
    assert response.energy_density.item() / shear**2 > 0.1


def test_rigid_rotation_preserves_energy_and_rotates_stress():
    potential = AnalyticPotential(
        fiber_order=1,
        ridge_terms=8,
        ridge_input_scales=[0.1, 0.1, 0.2],
    ).double()
    deformation = torch.tensor(
        [[[1.08, 0.08, 0.0], [0.02, 0.94, 0.04], [0.0, 0.03, 1.02]]],
        dtype=torch.float64,
    )
    rotation = _rotation_z(0.63)
    fiber = torch.tensor([0.8, 0.6, 0.0], dtype=torch.float64)
    base = potential(deformation, fiber)
    rotated = potential(rotation @ deformation, fiber)

    torch.testing.assert_close(rotated.energy_density, base.energy_density, atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(rotated.first_piola, rotation @ base.first_piola, atol=2.0e-11, rtol=2.0e-11)
    expected_cauchy = rotation @ base.cauchy_stress @ rotation.T
    torch.testing.assert_close(rotated.cauchy_stress, expected_cauchy, atol=2.0e-11, rtol=2.0e-11)
    torch.testing.assert_close(von_mises(rotated.cauchy_stress), von_mises(base.cauchy_stress), atol=2.0e-11, rtol=2.0e-11)


def test_von_mises_has_finite_zero_stress_gradient():
    stress = torch.zeros(2, 3, 3, requires_grad=True)
    value = von_mises(stress)
    torch.testing.assert_close(value, torch.zeros_like(value), atol=0.0, rtol=0.0)
    value.sum().backward()
    assert stress.grad is not None
    assert torch.isfinite(stress.grad).all()


def test_assembled_internal_force_has_zero_resultant_and_projection_is_weighted():
    nodes, cells = _unit_tetra()
    reference = precompute_tetrahedra(nodes, cells)
    potential = AnalyticPotential().double()
    deformation = torch.tensor(
        [[[1.05, 0.1, 0.0], [0.0, 0.97, 0.02], [0.0, 0.0, 1.03]]],
        dtype=torch.float64,
    )
    force = assemble_internal_force(
        potential(deformation).first_piola,
        cells,
        reference.volume,
        reference.shape_gradients,
        num_nodes=4,
    )
    projected = project_cell_to_nodes(
        torch.tensor([[2.0, 3.0]], dtype=torch.float64), cells, num_nodes=4, weights=reference.volume
    )

    assert torch.linalg.vector_norm(force.sum(dim=0)).item() < 1.0e-12
    torch.testing.assert_close(projected, torch.tensor([[2.0, 3.0]] * 4, dtype=torch.float64))


def test_closed_form_first_piola_matches_energy_finite_difference():
    potential = AnalyticPotential(
        order=2,
        fiber_order=1,
        ridge_terms=12,
        ridge_input_scales=[0.1, 0.1, 0.2],
        i1_init=[0.7, 0.04],
        i2_init=[0.2, 0.03],
        j_init=[1.3, 0.08],
    ).double()
    deformation = torch.tensor(
        [[[1.08, 0.06, -0.01], [0.02, 0.96, 0.04], [0.01, -0.02, 1.04]]],
        dtype=torch.float64,
    )
    fiber = torch.tensor([0.8, 0.5, 0.1], dtype=torch.float64)
    analytical = potential(deformation, fiber).first_piola
    finite_difference = torch.empty_like(analytical)
    epsilon = 2.0e-6
    for row in range(3):
        for column in range(3):
            perturbation = torch.zeros_like(deformation)
            perturbation[0, row, column] = epsilon
            energy_plus = potential(deformation + perturbation, fiber).energy_density
            energy_minus = potential(deformation - perturbation, fiber).energy_density
            finite_difference[0, row, column] = (energy_plus - energy_minus) / (2.0 * epsilon)

    torch.testing.assert_close(analytical, finite_difference, atol=2.0e-8, rtol=2.0e-6)


def test_convex_ridge_basis_has_zero_reference_energy_and_stress():
    potential = AnalyticPotential(
        ridge_terms=16,
        ridge_input_scales=[0.01, 0.01, 0.02],
    ).double()
    identity = torch.eye(3, dtype=torch.float64)[None]
    reference = potential(identity)
    torch.testing.assert_close(
        reference.energy_density, torch.zeros(1, dtype=torch.float64), atol=1.0e-12, rtol=0
    )
    torch.testing.assert_close(
        reference.first_piola, torch.zeros(1, 3, 3, dtype=torch.float64), atol=1.0e-11, rtol=0
    )
    deformed = torch.tensor(
        [[[1.04, 0.03, 0.0], [0.0, 0.98, 0.01], [0.0, 0.0, 1.02]]],
        dtype=torch.float64,
    )
    response = potential(deformed)
    assert response.energy_density.item() >= -1.0e-12
    assert torch.isfinite(response.first_piola).all()
    assert torch.all(potential.ridge_coefficients > 0.0)
    assert torch.all(potential.ridge_centers.abs() < potential.ridge_center_limit)

    reference_fp32 = potential.float()(torch.eye(3)[None])
    torch.testing.assert_close(
        reference_fp32.energy_density, torch.zeros(1), atol=2.0e-7, rtol=0
    )
    torch.testing.assert_close(
        reference_fp32.first_piola, torch.zeros(1, 3, 3), atol=2.0e-6, rtol=0
    )


def test_separable_ridge_has_term_invariant_curvature_budget():
    scales = [0.015, 0.018, 0.015]
    potential = AnalyticPotential(
        ridge_terms=8,
        ridge_init=1.0,
        ridge_input_scales=scales,
        ridge_curvature_normalization=True,
        ridge_mode="separable",
        ridge_train_directions=False,
        ridge_train_centers=False,
    )
    directions = potential.ridge_directions
    iso = directions[:, 2].abs() < 1.0e-7
    vol = ~iso

    assert int(iso.sum()) == 4
    assert int(vol.sum()) == 4
    assert torch.all(directions[iso, :2].abs() > 1.0e-4)
    torch.testing.assert_close(
        directions[vol, :2], torch.zeros_like(directions[vol, :2])
    )
    assert not potential.raw_ridge_directions.requires_grad
    assert not potential.raw_ridge_centers.requires_grad
    expected_iso_budget = 2.0 * scales[0] * scales[1]
    expected_vol_budget = scales[2] ** 2
    torch.testing.assert_close(
        potential._ridge_basis_scales[iso].sum(),
        torch.tensor(expected_iso_budget),
    )
    torch.testing.assert_close(
        potential._ridge_basis_scales[vol].sum(),
        torch.tensor(expected_vol_budget),
    )


def test_curvature_normalized_ridge_tangent_is_scale_invariant():
    common = {
        "ridge_terms": 8,
        "ridge_init": 1.0,
        "ridge_curvature_normalization": True,
        "ridge_mode": "separable",
        "ridge_train_directions": False,
        "ridge_train_centers": False,
    }
    first = AnalyticPotential(
        **common, ridge_input_scales=[0.01, 0.01, 0.02]
    ).double()
    second = AnalyticPotential(
        **common, ridge_input_scales=[0.02, 0.02, 0.04]
    ).double()
    strain = 1.0e-5
    deformation = (1.0 + strain) * torch.eye(3, dtype=torch.float64)[None]
    first_tangent = first(deformation).cauchy_stress[0, 0, 0] / strain
    second_tangent = second(deformation).cauchy_stress[0, 0, 0] / strain
    base_tangent = (
        AnalyticPotential().double()(deformation).cauchy_stress[0, 0, 0]
        / strain
    )

    torch.testing.assert_close(
        first_tangent - base_tangent,
        second_tangent - base_tangent,
        rtol=1.0e-3,
        atol=2.0e-4,
    )


def test_negative_determinant_barrier_is_finite_and_selective():
    determinant = torch.tensor([1.0, 0.0, -0.2], dtype=torch.float64)
    penalty = negative_j_barrier(determinant)
    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0, 0.02], dtype=torch.float64))

    inverted = torch.diag(torch.tensor([-0.2, 1.0, 1.0], dtype=torch.float64))[None]
    response = AnalyticPotential(inversion_stiffness=10.0).double()(inverted)
    assert torch.isfinite(response.energy_density).all()
    assert torch.isfinite(response.first_piola).all()
    assert response.inversion_barrier.item() > 0.0


def test_semi_implicit_step_enforces_fixed_and_prescribed_boundaries():
    position = torch.zeros(3, 3, dtype=torch.float64)
    velocity = torch.zeros_like(position)
    force = torch.tensor([[2.0, 0.0, 0.0]] * 3, dtype=torch.float64)
    mass = torch.full((3,), 2.0, dtype=torch.float64)
    prescribed_position = position.clone()
    prescribed_position[1, 0] = 0.5
    prescribed_velocity = velocity.clone()
    prescribed_velocity[1, 0] = 5.0
    result = semi_implicit_step(
        position,
        velocity,
        force,
        mass,
        0.1,
        fixed_mask=torch.tensor([True, False, False]),
        prescribed_mask=torch.tensor([False, True, False]),
        prescribed_position=prescribed_position,
        prescribed_velocity=prescribed_velocity,
    )

    torch.testing.assert_close(result.position[0], position[0])
    torch.testing.assert_close(result.velocity[0], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(result.position[1], prescribed_position[1])
    torch.testing.assert_close(result.velocity[1], prescribed_velocity[1])
    torch.testing.assert_close(result.position[2], torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(result.velocity[2], torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(result.acceleration[2], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_mechanics_parity():
    cpu_nodes, cpu_cells = _unit_tetra(dtype=torch.float32)
    cpu_reference = precompute_tetrahedra(cpu_nodes, cpu_cells, density=2.0)
    cpu_f = deformation_gradient(cpu_nodes * torch.tensor([1.05, 0.97, 1.02]), cpu_cells, cpu_reference.dm_inv)
    cpu_potential = AnalyticPotential(
        ridge_terms=8, ridge_input_scales=[0.1, 0.1, 0.2]
    ).eval()
    cpu_response = cpu_potential(cpu_f)
    cpu_force = assemble_internal_force(
        cpu_response.first_piola,
        cpu_cells,
        cpu_reference.volume,
        cpu_reference.shape_gradients,
        4,
    )

    gpu_nodes = cpu_nodes.cuda()
    gpu_cells = cpu_cells.cuda()
    gpu_reference = precompute_tetrahedra(gpu_nodes, gpu_cells, density=2.0)
    gpu_f = deformation_gradient(
        gpu_nodes * torch.tensor([1.05, 0.97, 1.02], device="cuda"),
        gpu_cells,
        gpu_reference.dm_inv,
    )
    gpu_potential = AnalyticPotential(
        ridge_terms=8, ridge_input_scales=[0.1, 0.1, 0.2]
    ).eval().cuda()
    gpu_potential.load_state_dict(cpu_potential.state_dict())
    gpu_response = gpu_potential(gpu_f)
    gpu_force = assemble_internal_force(
        gpu_response.first_piola,
        gpu_cells,
        gpu_reference.volume,
        gpu_reference.shape_gradients,
        4,
    )

    torch.testing.assert_close(gpu_response.energy_density.cpu(), cpu_response.energy_density, atol=1.0e-6, rtol=1.0e-5)
    torch.testing.assert_close(gpu_response.first_piola.cpu(), cpu_response.first_piola, atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(gpu_force.cpu(), cpu_force, atol=2.0e-6, rtol=2.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bf16_autocast_keeps_all_analytic_mechanics_fp32():
    nodes, cells = _unit_tetra(dtype=torch.float32)
    nodes = nodes.cuda().requires_grad_(True)
    cells = cells.cuda()
    reference = precompute_tetrahedra(nodes.detach(), cells)
    potential = AnalyticPotential(
        ridge_terms=8, ridge_input_scales=[0.1, 0.1, 0.2]
    ).cuda()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        deformation = deformation_gradient(
            nodes * torch.tensor([1.05, 0.97, 1.02], device="cuda"),
            cells,
            reference.dm_inv,
        )
        state = invariants(deformation)
        response = potential(deformation)
        force = assemble_internal_force(
            response.first_piola,
            cells,
            reference.volume,
            reference.shape_gradients,
            4,
        )
        loss = response.energy_density.sum() + force.square().sum()

    assert deformation.dtype == torch.float32
    assert state.c.dtype == torch.float32
    assert state.j.dtype == torch.float32
    assert response.first_piola.dtype == torch.float32
    assert response.cauchy_stress.dtype == torch.float32
    assert force.dtype == torch.float32
    loss.backward()
    assert nodes.grad is not None
    assert torch.isfinite(nodes.grad).all()
    for parameter in (
        potential.raw_ridge,
        potential.raw_ridge_directions,
        potential.raw_ridge_centers,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_positive_material_multipliers_change_stress_but_not_reference_state():
    potential = AnalyticPotential(order=2).eval()
    identity = torch.eye(3).repeat(2, 1, 1)
    multipliers = {
        "i1": torch.tensor([[1.0, 1.0], [2.0, 1.0]]),
        "i2": torch.ones(2, 2),
        "j": torch.ones(2, 2),
        "log_j": torch.ones(2, 1),
    }
    reference = potential(identity, coefficient_multipliers=multipliers)
    torch.testing.assert_close(
        reference.cauchy_stress,
        torch.zeros_like(reference.cauchy_stress),
        atol=1.0e-6,
        rtol=0.0,
    )

    stretched = identity.clone()
    stretched[:, 0, 0] = 1.2
    response = potential(stretched, coefficient_multipliers=multipliers)
    assert not torch.allclose(response.cauchy_stress[0], response.cauchy_stress[1])
    with pytest.raises(ValueError, match="must be positive"):
        potential(
            stretched,
            coefficient_multipliers={**multipliers, "i1": -multipliers["i1"]},
        )
