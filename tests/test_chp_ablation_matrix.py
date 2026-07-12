from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.run_chp_ablation_matrix import (
    AblationProtocolError,
    DEFAULT_MATRIX,
    _validate_checkpoint_tag,
    evaluate_validation_only,
    load_ablation_matrix,
    materialize_ablation_plan,
    validate_ablation_matrix,
)
from valgraphnet.config import load_config
from valgraphnet.chp_rollout import _resolve_rollout_split, _select_rollout_cases
from valgraphnet.chp_model import CHPGNS


def test_default_matrix_is_honest_about_supported_ablation_axes(tmp_path):
    matrix = load_ablation_matrix(DEFAULT_MATRIX)
    diagnostic = validate_ablation_matrix(matrix)
    assert diagnostic["enable_cell_memory"] is False

    plan = materialize_ablation_plan(DEFAULT_MATRIX, tmp_path / "resolved")
    assert set(value["variant_id"] for value in plan["runnable"].values()) == {
        "chp_full",
        "flat_processor",
        "unconstrained_contact",
        "one_step",
        "no_work_energy",
    }
    assert plan["test_execution_supported"] is False
    assert plan["training_execution_supported"] is False
    assert plan["comparisons"]["without_vs_with_cell_memory"]["status"] == (
        "not_triggered"
    )
    assert set(plan["incomplete_required_comparisons"]) == {
        "direct_decoder_vs_potential",
    }


def test_materialized_variants_are_single_factor_and_validation_only(tmp_path):
    plan = materialize_ablation_plan(DEFAULT_MATRIX, tmp_path / "resolved")
    by_variant = {
        entry["variant_id"]: load_config(entry["config"])
        for entry in plan["runnable"].values()
    }
    full = by_variant["chp_full"]
    one_step = by_variant["one_step"]
    no_work = by_variant["no_work_energy"]
    flat = by_variant["flat_processor"]
    unconstrained = by_variant["unconstrained_contact"]

    assert one_step["training"]["curriculum"] == [{"horizon": 1, "epochs": 16}]
    assert one_step["training"]["checkpoint_min_horizon"] == 1
    assert no_work["loss"]["work_energy"] == 0.0
    assert full["loss"]["work_energy"] == 0.1
    assert full["model"].get("use_topology_hierarchy", True) is True
    assert flat["model"]["use_topology_hierarchy"] is False
    matched_full = deepcopy(full)
    matched_flat = deepcopy(flat)
    matched_full["model"]["use_topology_hierarchy"] = False
    matched_full["training"]["output_dir"] = matched_flat["training"]["output_dir"]
    matched_full.pop("ablation")
    matched_flat.pop("ablation")
    assert matched_flat == matched_full
    assert full["contact"].get("enforce_action_reaction", True) is True
    assert unconstrained["contact"]["enforce_action_reaction"] is False
    matched_contact = deepcopy(unconstrained)
    matched_contact["contact"].pop("enforce_action_reaction")
    matched_contact["training"]["output_dir"] = full["training"]["output_dir"]
    matched_contact.pop("ablation")
    full_without_tag = deepcopy(full)
    full_without_tag.pop("ablation")
    assert matched_contact == full_without_tag
    for cfg in by_variant.values():
        assert cfg["training"]["device"] == "cuda"
        assert cfg["training"]["amp_dtype"] == "bfloat16"
        assert cfg["data"]["val_split"] == "val"
        assert cfg["validation"]["native_reference_split"] == "val"
        assert cfg["evaluation"]["case_selection"] == "even"
        assert cfg["evaluation"]["max_cases"] == 20
        assert cfg["evaluation"]["steps"] == 399
        assert cfg["ablation"]["development_only"] is True
        assert len(cfg["ablation"]["config_fingerprint"]) == 64


def test_complete_matrix_request_fails_for_unimplemented_model_switches(tmp_path):
    with pytest.raises(AblationProtocolError, match="not implemented"):
        materialize_ablation_plan(
            DEFAULT_MATRIX, tmp_path / "resolved", require_complete=True
        )


def test_test_like_evaluation_split_is_rejected():
    matrix = deepcopy(load_ablation_matrix(DEFAULT_MATRIX))
    matrix["protocol"]["evaluation_split"] = "test"
    with pytest.raises(AblationProtocolError, match="unsafe.*split"):
        validate_ablation_matrix(matrix)


def test_blocked_or_unknown_variant_cannot_be_scheduled(tmp_path):
    plan = materialize_ablation_plan(DEFAULT_MATRIX, tmp_path / "resolved")
    checkpoint_map = {
        "schema_version": 1,
        "checkpoints": {
            "chp_full": {"42": "missing.pt"},
            "flat_processor": {"42": "missing.pt"},
            "unconstrained_contact": {"42": "missing.pt"},
            "one_step": {"42": "missing.pt"},
            "no_work_energy": {"42": "missing.pt"},
        }
    }
    with pytest.raises(AblationProtocolError, match="blocked or unknown"):
        evaluate_validation_only(
            plan,
            checkpoint_map,
            tmp_path / "evaluation",
            selected_variants=["direct_decoder"],
            dry_run=True,
        )


def test_dry_run_emits_only_fixed_validation_commands(tmp_path):
    plan = materialize_ablation_plan(DEFAULT_MATRIX, tmp_path / "resolved")
    checkpoint_map = {
        "schema_version": 1,
        "checkpoints": {
            "chp_full": {"42": "full.pt"},
            "flat_processor": {"42": "flat.pt"},
            "unconstrained_contact": {"42": "contact.pt"},
            "one_step": {"42": "one.pt"},
            "no_work_energy": {"42": "energy.pt"},
        }
    }
    result = evaluate_validation_only(
        plan,
        checkpoint_map,
        tmp_path / "evaluation",
        dry_run=True,
    )
    assert len(result["records"]) == 5
    for record in result["records"]:
        command = record["command"]
        split_index = command.index("--split") + 1
        assert command[split_index] == "val"
        assert "test" not in command[split_index].lower()
        assert record["status"] == "planned"


def test_conditional_memory_ablation_becomes_blocking_only_when_triggered(tmp_path):
    matrix = load_ablation_matrix(DEFAULT_MATRIX)
    diagnostic_path = tmp_path / "diagnostic.json"
    source = json.loads(
        Path(matrix["cell_memory_diagnostic"]["result_file"]).read_text(
            encoding="utf-8"
        )
    )
    source["conditional_to_global_variance"] = 0.11
    source["enable_cell_memory"] = True
    diagnostic_path.write_text(json.dumps(source), encoding="utf-8")
    matrix["cell_memory_diagnostic"]["result_file"] = str(diagnostic_path)
    matrix["base_config"] = str(Path(matrix["base_config"]).resolve())
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")

    plan = materialize_ablation_plan(matrix_path, tmp_path / "resolved")
    assert plan["comparisons"]["without_vs_with_cell_memory"]["status"] == "blocked"
    assert "without_vs_with_cell_memory" in plan["incomplete_required_comparisons"]


def test_chp_export_uses_the_same_even_validation_subset_as_checkpointing():
    cases = [SimpleNamespace(case_id=f"case-{index:03d}") for index in range(100)]
    selected = _select_rollout_cases(cases, 20, "even")
    assert [case.case_id for case in selected] == [
        cases[round(index * 99 / 19)].case_id for index in range(20)
    ]


def test_checkpoint_cannot_be_relabelled_after_config_changes(tmp_path):
    plan = materialize_ablation_plan(DEFAULT_MATRIX, tmp_path / "resolved")
    expected = next(
        value
        for value in plan["runnable"].values()
        if value["variant_id"] == "chp_full"
    )
    cfg = load_config(expected["config"])
    checkpoint = {
        "schema_version": CHPGNS.checkpoint_schema_version,
        "dynamics_schema_version": CHPGNS.dynamics_schema_version,
        "residual_parameterization": CHPGNS.residual_parameterization,
        "residual_gate": CHPGNS.residual_gate,
        "scientific_gate_status": "passed",
        "config": cfg,
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    _validate_checkpoint_tag(path, expected)

    checkpoint["config"]["loss"]["work_energy"] = 0.0
    torch.save(checkpoint, path)
    with pytest.raises(AblationProtocolError, match="internally inconsistent"):
        _validate_checkpoint_tag(path, expected)


def test_materialized_config_rejects_test_even_outside_matrix_runner(tmp_path):
    plan = materialize_ablation_plan(DEFAULT_MATRIX, tmp_path / "resolved")
    cfg = load_config(next(iter(plan["runnable"].values()))["config"])
    assert _resolve_rollout_split(cfg, None) == "val"
    assert _resolve_rollout_split(cfg, "val") == "val"
    with pytest.raises(ValueError, match="development-only"):
        _resolve_rollout_split(cfg, "test")
