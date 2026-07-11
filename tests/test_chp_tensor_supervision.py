from types import SimpleNamespace

import numpy as np
import torch

from valgraphnet.chp_model import CHPState, PhysicalStep
from valgraphnet.chp_train import (
    CHPDeviceCase,
    _cauchy_to_tensor6,
    _cell_tensor_constitutive_loss,
    _tensor6_von_mises,
    chp_step_loss,
    evaluate_teacher_forced_stress,
    fit_chp_normalizers,
)
from valgraphnet.mechanics import von_mises


def _tensor6_matrix(value: torch.Tensor) -> torch.Tensor:
    s11, s22, s33, s12, s13, s23 = value.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((s11, s12, s13), dim=-1),
            torch.stack((s12, s22, s23), dim=-1),
            torch.stack((s13, s23, s33), dim=-1),
        ),
        dim=-2,
    )


def _normalizer_case(*, with_tensor: bool = True) -> SimpleNamespace:
    displacement = np.zeros((3, 4, 3), dtype=np.float32)
    displacement[1:, :, 0] = np.asarray([0.01, 0.02], dtype=np.float32)[:, None]
    velocity = np.zeros_like(displacement)
    velocity[1:, :, 0] = np.asarray([0.01, 0.02], dtype=np.float32)[:, None]
    tensor = np.asarray([2.0, -1.0, 0.5, 0.25, -0.75, 1.25], dtype=np.float32)
    cell_stress = (
        np.tile(tensor, (3, 1, 1))
        if with_tensor
        else np.empty((3, 0, 0), dtype=np.float32)
    )
    return SimpleNamespace(
        num_steps=3,
        num_nodes=4,
        num_cells=1,
        stress_dim=1,
        times=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        displacement=displacement,
        velocity=velocity,
        stress=np.ones((3, 4, 1), dtype=np.float32),
        cell_stress=cell_stress,
        cells=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        fixed_mask=np.zeros(4, dtype=bool),
        prescribed_mask=np.zeros(4, dtype=bool),
    )


def test_tensor6_conversion_and_von_mises_use_canonical_signed_components():
    tensor6 = torch.tensor([[2.0, -1.0, 0.5, 0.25, -0.75, 1.25]])
    matrix = _tensor6_matrix(tensor6)

    torch.testing.assert_close(_cauchy_to_tensor6(matrix), tensor6)
    torch.testing.assert_close(_tensor6_von_mises(tensor6), von_mises(matrix))


def test_normalizers_fit_and_round_trip_signed_cell_tensor_statistics():
    normalizers = fit_chp_normalizers(
        [_normalizer_case()], max_cases=1, frames_per_case=2, nodes_per_frame=4
    )

    assert normalizers.cell_stress is not None
    values = torch.tensor([[-3.0, 2.0, -1.0, 0.5, -0.25, 4.0]])
    recovered = normalizers.cell_stress.inverse(
        normalizers.cell_stress.transform(values)
    )
    torch.testing.assert_close(recovered, values)
    restored = type(normalizers).from_state_dict(normalizers.state_dict())
    assert restored.cell_stress is not None
    torch.testing.assert_close(
        restored.cell_stress.reference_scale,
        normalizers.cell_stress.reference_scale,
    )

    scalar_only = fit_chp_normalizers(
        [_normalizer_case(with_tensor=False)],
        max_cases=1,
        frames_per_case=2,
        nodes_per_frame=4,
    )
    assert scalar_only.cell_stress is None


def test_signed_tensor_loss_is_zero_at_target_and_backpropagates_sign_error():
    normalizers = fit_chp_normalizers(
        [_normalizer_case()], max_cases=1, frames_per_case=2, nodes_per_frame=4
    )
    target = torch.tensor([[2.0, -1.0, 0.5, 0.25, -0.75, 1.25]])
    exact = _tensor6_matrix(target)
    exact_loss, exact_metrics = _cell_tensor_constitutive_loss(
        exact, target, normalizers, {"loss": {"cell_vm_weight": 0.25}}
    )
    torch.testing.assert_close(exact_loss, torch.zeros_like(exact_loss))
    assert exact_metrics["stress_tensor_supervision"].item() == 1.0

    wrong = (-exact).requires_grad_()
    loss, metrics = _cell_tensor_constitutive_loss(
        wrong, target, normalizers, {"loss": {"cell_vm_weight": 0.25}}
    )
    loss.backward()
    assert loss.item() > 0.0
    assert torch.isfinite(wrong.grad).all()
    assert metrics["stress_tensor_relative_rmse"].item() > 0.0

    reference = torch.zeros((1, 3, 3), requires_grad=True)
    zero_loss, _ = _cell_tensor_constitutive_loss(
        reference, torch.zeros((1, 6)), normalizers, {"loss": {}}
    )
    zero_loss.backward()
    torch.testing.assert_close(zero_loss, torch.zeros_like(zero_loss))
    assert torch.isfinite(reference.grad).all()


def test_rollout_step_loss_prefers_cell_tensor_over_inconsistent_nodal_label():
    normalizers = fit_chp_normalizers(
        [_normalizer_case()], max_cases=1, frames_per_case=2, nodes_per_frame=4
    )
    reference = torch.zeros((4, 3))
    static = SimpleNamespace(
        reference_position=reference,
        cells=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        fixed_mask=torch.zeros(4, dtype=torch.bool),
        prescribed_mask=torch.zeros(4, dtype=torch.bool),
    )
    target6 = torch.tensor([[2.0, -1.0, 0.5, 0.25, -0.75, 1.25]])
    trajectory = SimpleNamespace(
        static=static,
        times=torch.tensor([0.0, 1.0]),
        displacement=torch.zeros((2, 4, 3)),
        velocity=torch.zeros((2, 4, 3)),
        stress=torch.ones((2, 4, 1)),
        cell_stress=target6[None].repeat(2, 1, 1),
    )
    state = CHPState(reference.clone(), torch.zeros_like(reference))
    zero = torch.zeros(())
    diagnostics = {
        "kinetic": zero,
        "kinetic_after": zero,
        "potential": zero,
        "potential_after": zero,
        "external_work": zero,
        "boundary_work": zero,
        "residual_work": zero,
        "projection_dissipation": zero,
        "work_energy_balance": zero,
        "integration_update_scale": torch.ones(()),
        "integration_valid": torch.ones(()),
        "max_penetration": zero,
        "residual_norm": zero,
        "residual_reference": torch.ones(()),
        "negative_j": zero,
        "integration_domain_penalty": zero,
    }
    output = PhysicalStep(
        next_position=reference.clone(),
        next_velocity=torch.zeros_like(reference),
        acceleration=torch.zeros_like(reference),
        nodal_stress=torch.zeros((4, 1)),
        cell_stress_tensor=_tensor6_matrix(target6),
        internal_force=torch.zeros_like(reference),
        contact_force=torch.zeros_like(reference),
        damping_force=torch.zeros_like(reference),
        residual_force=torch.zeros_like(reference),
        energy_diagnostics=diagnostics,
    )

    loss, metrics = chp_step_loss(
        output, state, trajectory, 1, normalizers, {"loss": {}}
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert metrics["stress_tensor_supervision"].item() == 1.0


def _teacher_fixture(*, with_tensor: bool):
    reference = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    target6 = torch.tensor([[2.0, -1.0, 0.5, 0.25, -0.75, 1.25]])
    cell_stress = (
        target6[None].repeat(2, 1, 1)
        if with_tensor
        else torch.empty((2, 0, 0))
    )
    static = SimpleNamespace(
        reference_position=reference,
        cells=cells,
        dm_inv=torch.eye(3)[None],
        prescribed_mask=torch.zeros(4, dtype=torch.bool),
    )
    trajectory = SimpleNamespace(
        static=static,
        times=torch.tensor([0.0, 1.0]),
        displacement=torch.zeros((2, 4, 3)),
        stress=torch.ones((2, 4, 1)),
        cell_stress=cell_stress,
    )
    case = SimpleNamespace(num_steps=2)

    class ExactTensorWrongNodal:
        def eval(self):
            return self

        def nodal_stress_at(self, _static, _position):
            return torch.zeros((4, 1)), _tensor6_matrix(target6)

    cache = SimpleNamespace(device=torch.device("cpu"), get=lambda _index: trajectory)
    cfg = {
        "validation": {"teacher_stress_cases": 1, "teacher_stress_frames": 1},
        "training": {"minimum_start_j": 1.0e-2, "maximum_start_i2_bar": 1.0e5},
    }
    return ExactTensorWrongNodal(), [case], cache, cfg


def test_teacher_gate_prefers_tensor_and_explicitly_reports_scalar_fallback():
    tensor_metrics = evaluate_teacher_forced_stress(*_teacher_fixture(with_tensor=True))
    assert tensor_metrics["teacher_stress_source"] == "cell_tensor"
    assert tensor_metrics["teacher_stress_relative_rmse"] == 0.0
    assert tensor_metrics["teacher_nodal_stress_relative_rmse"] == 1.0
    assert tensor_metrics["teacher_cell_stress_tensor_coverage"] == 1.0

    scalar_metrics = evaluate_teacher_forced_stress(
        *_teacher_fixture(with_tensor=False)
    )
    assert scalar_metrics["teacher_stress_source"] == "nodal_scalar_vm_fallback"
    assert scalar_metrics["teacher_stress_relative_rmse"] == 1.0
    assert scalar_metrics["teacher_cell_stress_tensor_coverage"] == 0.0
    assert scalar_metrics["teacher_cell_stress_tensor_relative_rmse"] == float("inf")


def test_device_case_contract_includes_cell_stress():
    assert "cell_stress" in CHPDeviceCase.__dataclass_fields__
