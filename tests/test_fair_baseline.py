import torch

from valgraphnet.fair_baseline import add_position_noise, integrate_position_delta
from valgraphnet.stress_transform import AsinhStressTransform, robust_stress_loss


def test_asinh_stress_transform_round_trip_and_tail_loss():
    stress = torch.tensor([[0.0], [10.0], [100.0], [1.0e6]])
    transform = AsinhStressTransform.fit([stress])
    encoded = transform.transform(stress)
    recovered = transform.inverse(encoded)
    loss, components = robust_stress_loss(encoded + 0.1, encoded)

    assert torch.allclose(recovered, stress, rtol=1.0e-5, atol=1.0e-4)
    assert loss > 0
    assert components["stress_peak"] > 0


def test_position_delta_is_the_only_integrated_dynamics_output():
    current = torch.zeros(3, 3)
    velocity = torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1)
    delta = torch.tensor([[2.0, 0.0, 0.0]]).repeat(3, 1)
    state = integrate_position_delta(
        current,
        velocity,
        delta,
        2.0,
        fixed_mask=torch.tensor([False, True, False]),
        prescribed_mask=torch.tensor([False, False, True]),
        prescribed_position=torch.tensor([[0.0, 0.0, 0.0]] * 2 + [[4.0, 0.0, 0.0]]),
    )

    assert torch.equal(state.next_position[0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.equal(state.next_position[1], torch.zeros(3))
    assert torch.equal(state.next_position[2], torch.tensor([4.0, 0.0, 0.0]))
    assert torch.equal(state.next_velocity[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(state.next_velocity[1], torch.zeros(3))
    assert torch.equal(state.next_velocity[2], torch.tensor([2.0, 0.0, 0.0]))


def test_position_noise_corrects_target_and_only_moves_free_nodes():
    current = torch.zeros(4, 3)
    future = torch.ones(4, 3)
    generator = torch.Generator().manual_seed(7)
    noisy, corrected, noise = add_position_noise(
        current,
        future,
        torch.tensor([True, False, True, False]),
        0.003,
        generator=generator,
    )

    assert torch.equal(noise[1], torch.zeros(3))
    assert torch.equal(noise[3], torch.zeros(3))
    assert torch.allclose(noisy + corrected, future)
