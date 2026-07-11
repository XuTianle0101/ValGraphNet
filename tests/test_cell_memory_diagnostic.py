import numpy as np

from scripts.diagnose_cell_memory import conditional_variance_ratio


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
