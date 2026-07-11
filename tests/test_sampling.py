from valgraphnet.data.sampling import PerTrajectoryStepSampler


def test_per_trajectory_sampler_covers_every_group():
    groups = [range(0, 5), range(5, 10), range(10, 15)]
    sampler = PerTrajectoryStepSampler(groups, 2, shuffle=True, seed=7)

    indices = list(sampler)

    assert len(indices) == 6
    assert sum(index < 5 for index in indices) == 2
    assert sum(5 <= index < 10 for index in indices) == 2
    assert sum(index >= 10 for index in indices) == 2


def test_per_trajectory_sampler_is_epoch_reproducible():
    groups = [range(0, 20), range(20, 40)]
    sampler = PerTrajectoryStepSampler(groups, 3, shuffle=True, seed=11)

    sampler.set_epoch(4)
    first = list(sampler)
    sampler.set_epoch(4)
    second = list(sampler)
    sampler.set_epoch(5)
    third = list(sampler)

    assert first == second
    assert first != third


def test_validation_sampler_uses_evenly_spaced_steps():
    sampler = PerTrajectoryStepSampler([range(10)], 3, shuffle=False)

    assert list(sampler) == [0, 4, 9]


def test_multistep_training_sampler_excludes_incomplete_tail():
    from valgraphnet.train import _build_step_sampler

    dataset = type("Dataset", (), {"trajectory_index_groups": [range(0, 10)]})()
    cfg = {
        "seed": 3,
        "training": {
            "steps_per_trajectory_per_epoch": 20,
            "rollout_steps": 3,
        },
    }
    sampler = _build_step_sampler(dataset, cfg, training=True)

    assert sorted(sampler.groups[0]) == list(range(8))
    assert max(iter(sampler)) <= 7
