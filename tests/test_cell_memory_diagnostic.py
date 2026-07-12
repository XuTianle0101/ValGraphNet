import numpy as np

from scripts.diagnose_cell_memory import (
    conditional_variance_ratio,
    objective_stress_invariants,
)


def test_deterministic_constitutive_map_has_low_local_variance():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(4000, 3))
    stress = 2.0 * features[:, 0] - features[:, 1] + 0.5 * features[:, 2]
    assert conditional_variance_ratio(features, stress, neighbors=16) < 0.10


def test_hidden_branch_variable_triggers_memory_diagnostic():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(4000, 3))
    branch = rng.choice([-5.0, 5.0], size=4000)
    stress = 0.1 * features[:, 0] + branch
    assert conditional_variance_ratio(features, stress, neighbors=16) > 0.50


def test_multicomponent_hidden_branch_triggers_memory_diagnostic():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(4000, 3))
    first = features[:, 0] + rng.choice([-4.0, 4.0], size=4000)
    second = 0.2 * features[:, 1]
    stress = np.stack([first, second], axis=1)
    assert conditional_variance_ratio(features, stress, neighbors=16) > 0.50


def test_objective_stress_invariants_ignore_coordinate_rotation():
    tensor = np.asarray([[5.0, 2.0, -1.0, 0.7, -0.2, 0.4]])
    matrix = np.asarray(
        [[[5.0, 0.7, -0.2], [0.7, 2.0, 0.4], [-0.2, 0.4, -1.0]]]
    )
    angle = 0.61
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = rotation @ matrix @ rotation.T
    rotated_tensor = np.stack(
        [
            rotated[:, 0, 0],
            rotated[:, 1, 1],
            rotated[:, 2, 2],
            rotated[:, 0, 1],
            rotated[:, 0, 2],
            rotated[:, 1, 2],
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        objective_stress_invariants(tensor),
        objective_stress_invariants(rotated_tensor),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
