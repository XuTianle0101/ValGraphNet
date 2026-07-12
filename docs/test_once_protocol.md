# Formal test-once protocol

`scripts/test_once.py` prevents checkpoint or hyperparameter adaptation after a
formal test set is opened.  It has four states:

1. `freeze` validates every validation-selected checkpoint and writes an
   immutable lock containing raw SHA256 digests for configuration,
   checkpoint, validation evidence, evaluation entrypoints, and optional
   frozen inputs.
2. `verify` repeats those checks without reading any test trajectory and
   without consuming the test attempt.
3. `run` re-verifies the lock, atomically creates a permanent claim, and then
   executes the exact shell-free command list stored in the lock.
4. The command receipt is terminal.  A failed command also consumes the only
   attempt; changing an experiment requires a new experiment id and must be
   reported as a new protocol rather than a retry.

## Specification

The input is JSON or YAML. Paths are relative to `--workspace`.

```yaml
schema_version: 1
experiment_id: deforming-plate-frozen-2026-07-seed42

models:
  - name: native
    family: native
    config: examples/deforming_plate/config.full_400.yaml
    checkpoint: outputs/deforming_plate_full_400_sampled_stats/best.pt
    selection_evidence: outputs/deforming_plate_chp_evidence/native_val20/metrics.json
  - name: fair-mgn
    family: fair
    config: configs/deforming_plate_fair_mgn.full400.yaml
    checkpoint: outputs/deforming_plate_fair_full400_seed42/best.pt
  - name: multiscale-mgn
    family: multiscale
    config: configs/deforming_plate_multiscale_mgn.full400.yaml
    checkpoint: outputs/deforming_plate_multiscale_full400_seed42/best.pt
  - name: repository
    family: repo
    config: configs/deforming_plate_case.full400_repo.yaml
    checkpoint: outputs/deforming_plate_repo_full400_seed42/best.pt
    selection_evidence: outputs/deforming_plate_repo_full400_seed42/history.json
  - name: chp-gns
    family: chp
    config: configs/deforming_plate_chp.full400.yaml
    checkpoint: outputs/deforming_plate_chp_full400_seed42/best.pt

evaluation_entrypoints:
  - examples/deforming_plate/rollout_eval.py
  - scripts/export_fair_rollouts.py
  - scripts/export_multiscale_rollouts.py
  - scripts/export_legacy_rollouts.py
  - scripts/evaluate_chp.py

test_commands:
  - name: frozen-test-matrix
    models: [native, fair-mgn, multiscale-mgn, repository, chp-gns]
    argv: [.venv/Scripts/python.exe, scripts/run_frozen_test_matrix.py,
           --protocol, configs/deforming_plate_test_matrix.yaml]
```

The example command is illustrative: the command and every listed entrypoint
must exist before freezing. The lock intentionally does not open
`splits.json`, a case directory, or a test TFRecord.

## Family-specific validation evidence

- `native` requires an external standardized validation metrics artifact with
  `evaluation.split` equal to the configured validation split.
- `fair` and `multiscale` require the checkpoint's four rollout metrics,
  positive native references, finite minimax score, and a provenance-rich
  validation reference file. MultiScale additionally requires
  `checkpoint_split` to equal the validation split.
- `repo` requires `history.json`; the selected checkpoint epoch must have
  exactly one non-null `rollout_val` row and the row score must equal the
  checkpoint score.
- `chp` requires a passed (or explicitly not-required) scientific gate and a
  score that recomputes from all four primary rollout metrics. Native-relative
  mode also binds its validation reference; absolute-validation mode forbids
  silently substituting a native reference.

In every family, the external configuration must be semantically identical to
the configuration embedded in the checkpoint. Validation and test split names
must be distinct.

## Commands

```powershell
.venv\Scripts\python.exe scripts/test_once.py freeze `
  --spec configs/deforming_plate_test_once.yaml --workspace .

.venv\Scripts\python.exe scripts/test_once.py verify `
  --experiment-id deforming-plate-frozen-2026-07-seed42

# This is the irreversible operation. Do not run until the paper protocol is frozen.
.venv\Scripts\python.exe scripts/test_once.py run `
  --experiment-id deforming-plate-frozen-2026-07-seed42 `
  --execute-frozen-test-plan
```

An external scheduler may use `claim --consume-test-attempt` immediately before
its own frozen command plan. A claim is equally irreversible and cannot later
be followed by `run` for the same experiment id.

The default registry is `outputs/test_once_registry`. It is the governance
boundary: copying the repository or deliberately choosing a different registry
cannot be prevented by local code, so the registry directory and lock/claim
digests must be archived with the experiment artifacts. Existing evaluation
entrypoints can still be invoked directly; such outputs are not formal
test-once results unless they are covered by a lock and claim receipt.
