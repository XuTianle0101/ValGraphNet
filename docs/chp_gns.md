# CHP-GNS experiment route

CHP-GNS is the constitutive-consistent path for deforming solids. Stress,
internal force, and state evolution share one analytic tetrahedral potential;
the graph network is restricted to long-range propagation, damping, contact,
and a bounded residual force.

## Reproducible stages

All neural training and rollout commands below require CUDA. The reference
configuration targets the local RTX 4060 Ti with BF16 neural blocks and FP32
determinants, inverses, stress, and force assembly.

```powershell
# 1. Non-destructive complete DeepMind conversion (1000/100/100 x 400)
.venv\Scripts\python.exe -m examples.deforming_plate.convert_to_cases `
  --config examples\deforming_plate\config.full_400.yaml

# 2. Frozen native baseline
.venv\Scripts\python.exe -m examples.deforming_plate.train `
  --config examples\deforming_plate\config.full_400.yaml
.venv\Scripts\python.exe -m examples.deforming_plate.rollout_eval `
  --config examples\deforming_plate\config.full_400.yaml `
  --checkpoint <native-best.pt> --out <native-npz-dir>
.venv\Scripts\python.exe scripts\standardize_native_rollouts.py `
  --case-root data\deforming_plate_cases_400 `
  --split-file data\deforming_plate_cases_400\splits.json `
  --native-dir <native-npz-dir> --out <native-standard-dir>

# Frozen validation reference for checkpoint selection. This must not point at
# test artifacts; the loader verifies the split, ordered cases, and 400 frames.
.venv\Scripts\python.exe -m examples.deforming_plate.rollout_eval `
  --config examples\deforming_plate\config.full_400.yaml `
  --checkpoint <native-best.pt> --split val --max-cases 20 `
  --case-selection even --out <native-val20-npz-dir>
.venv\Scripts\python.exe scripts\standardize_native_rollouts.py `
  --case-root data\deforming_plate_cases_400 `
  --split-file data\deforming_plate_cases_400\splits.json `
  --split val --max-cases 20 --case-selection even `
  --native-dir <native-val20-npz-dir> `
  --out outputs\deforming_plate_chp_evidence\native_val20

# 3. Corrected four-output MGN: delta-x + asinh stress
.venv\Scripts\python.exe scripts\train_fair_deforming_plate.py `
  --config configs\deforming_plate_fair_mgn.full400.yaml
.venv\Scripts\python.exe scripts\export_fair_rollouts.py `
  --config configs\deforming_plate_fair_mgn.full400.yaml `
  --checkpoint <fair-best.pt> --out <fair-standard-dir>

# 4. Frozen two-level topology-bi-stride MultiScale MGN baseline
.venv\Scripts\python.exe scripts\train_multiscale_deforming_plate.py `
  --config configs\deforming_plate_multiscale_mgn.full400.yaml
.venv\Scripts\python.exe scripts\export_multiscale_rollouts.py `
  --config configs\deforming_plate_multiscale_mgn.full400.yaml `
  --checkpoint <multiscale-best.pt> --out <multiscale-standard-dir>

# 5. CHP-GNS K=1 -> 2 -> 4 -> 8 -> 16 physical rollout curriculum
.venv\Scripts\python.exe scripts\train_chp.py `
  --config configs\deforming_plate_chp.full400.yaml
.venv\Scripts\python.exe scripts\evaluate_chp.py `
  --config configs\deforming_plate_chp.full400.yaml `
  --checkpoint <chp-best.pt> --out <chp-standard-dir>

# 6. Paired physical-unit comparison and bootstrap confidence intervals
.venv\Scripts\python.exe scripts\compare_physical_rollouts.py `
  --case-root data\deforming_plate_cases_400 `
  --split-file data\deforming_plate_cases_400\splits.json `
  --experiment native=<native-standard-dir> `
  --experiment repo=<repo-standard-dir> `
  --experiment fair_mgn=<fair-standard-dir> `
  --experiment multiscale_mgn=<multiscale-standard-dir> `
  --experiment chp_gns=<chp-standard-dir> `
  --baseline fair_mgn --candidate chp_gns --out <comparison.json>
```

The common evaluator uses moving nodes for displacement, all non-prescribed
nodes for stress (including the clamped high-stress region), physical units,
truth top-5% stress support for P95 error, and pooled squared errors. The
checkpoint objective is the worst of four ratios to the frozen native result;
one improving metric cannot compensate for another regressing metric.

`standard_reference` in the comparison JSON is the ground-truth field compared
with itself. Its four errors are therefore zero by definition; it is not a
trained solver or surrogate result. Native, repository, fair MGN, MultiScale
MGN, and CHP-GNS are the rows that measure model error against that reference.

## Constitutive and benchmark gates

The exact-geometry teacher-stress gate is checked immediately after
constitutive pretraining and checked again after the four K=1 epochs. Failure
at relative RMSE 0.50 stops dynamics and rollout training before they can mask
a constitutive failure. The K=1 rollout pilot is an independent gate. Cell
memory is off by default and is enabled only when
`scripts/diagnose_cell_memory.py` reports a conditional-to-global stress
variance above 0.10.

The optimized constitutive decoder retains the analytic polynomial and
`-log(J)` barrier and adds positive convex invariant-space ridge bases. Their
softplus Bregman form has exactly zero reference energy and first derivative,
and its closed-form invariant derivative is chained to `F` in FP32. This is a
convexity statement in normalized invariant space, not a claim of global
polyconvexity or strong ellipticity.

HyperContact-3D decks and deterministic ID/material/load/mesh/combined OOD
splits are generated with:

```powershell
.venv\Scripts\python.exe scripts\generate_hypercontact3d.py `
  --config configs\hypercontact3d.yaml --output data\hypercontact3d_raw_v1_9
.venv\Scripts\python.exe scripts\run_hypercontact3d.py `
  --manifest data\hypercontact3d_raw_v1_9\manifest.json `
  --ccx wsl --ccx-arg ccx
.venv\Scripts\python.exe scripts\convert_hypercontact3d.py `
  --manifest data\hypercontact3d_raw_v1_9\manifest.json `
  --output data\hypercontact3d_cases_v1_9
.venv\Scripts\python.exe scripts\audit_hypercontact3d.py `
  --config configs\hypercontact3d_chp.v1_9.yaml `
  --data-root data\hypercontact3d_cases_v1_9 `
  --output outputs\hypercontact3d_v1_9_converted_audit.json
```

HyperContact uses quasi-static continuation semantics. Velocity and kinetic
work losses are forcibly disabled, dynamics pretraining is forbidden, and the
CalculiX fixed-support reaction is used only as an equilibrium label—never as
a model input.

The DeepMind source exposes one nodal scalar stress channel, not a six-component
cell tensor. Full tensor supervision and material/mesh OOD evidence therefore
come from HyperContact-3D and Valve/Abaqus. Valve is a hard external-data gate:
an ODB export must contain complete cell/IP stress, density, material features,
and fiber directions before it is included in a paper claim.

## Current falsifiable evidence

All numbers below are read from the cited local artifacts. `FINAL-VAL20` means
that the validation result is complete and frozen; it does **not** mean that a
final test result exists. `PARTIAL`, `DIAGNOSTIC ONLY`, `FAILED GATE`, and
`DYNAMICS/ROLLOUT INCOMPLETE` rows must not be promoted to headline
comparisons.

| Evidence | Status and scope | Result | Local artifact |
|---|---|---|---|
| Native MGN | **FINAL-VAL20**; 20 evenly selected validation trajectories, all 400 frames | moving displacement rRMSE `1.291795`; final displacement rRMSE `210.644627`; stress rRMSE `0.769733`; P95-stress rRMSE `0.686852`; `0/20` diverged | `outputs/deforming_plate_chp_evidence/native_val20/metrics.json` |
| Fair MGN | **PARTIAL (not final)**; training history currently reaches epoch 7 | The epoch-6 rollout regressed severely (moving `14.536056`; final `2177.868164`; stress `0.981129`; P95 stress `0.580530`; minimax score `11.252604`), so the best remains epoch 3: moving `2.417227`; final `283.234955`; stress `0.658701`; P95 stress `0.688011`; `0/20` diverged. Neither checkpoint is the completed 30-epoch result. | `outputs/deforming_plate_fair_mgn_full400_seed42/history.json` |
| Low-order analytic CHP | **FAILED GATE** | constitutive-pretrain teacher rRMSE `0.718619`; after four K=1 epochs, teacher rRMSE `0.692647 > 0.50` and `16/20` validation rollouts diverged. K=2--16 and test were not run. | `outputs/deforming_plate_chp_full400_seed42/constitutive_pretraining.json`; `outputs/deforming_plate_chp_full400_seed42/history.json`; `outputs/deforming_plate_chp_full400_seed42/teacher_stress_gate_failure.json` |
| Raw ridge CHP | **FAILED GATE** before dynamics/rollout | selected teacher rRMSE `0.874068 > 0.50` | `outputs/deforming_plate_chp_ridge_full400_seed42/constitutive_pretraining.json`; `outputs/deforming_plate_chp_ridge_full400_seed42/teacher_stress_gate_failure.json` |
| Robust mask-aligned base CHP | **FAILED GATE** before dynamics/rollout | selected teacher rRMSE `0.520473 > 0.50` | `outputs/deforming_plate_chp_base_mask_aligned_full400_seed42/constitutive_pretraining.json`; `outputs/deforming_plate_chp_base_mask_aligned_full400_seed42/teacher_stress_gate_failure.json` |
| Normalized mask-aligned ridge-8 CHP | **FAILED GATE** before dynamics/rollout | selected teacher rRMSE `0.521642 > 0.50` | `outputs/deforming_plate_chp_ridge8_mask_aligned_full400_seed42/constitutive_pretraining.json`; `outputs/deforming_plate_chp_ridge8_mask_aligned_full400_seed42/teacher_stress_gate_failure.json` |
| Neural-feature potential CHP | **TEACHER GATE PASSED; DYNAMICS/ROLLOUT INCOMPLETE** | selected epoch-11 teacher rRMSE/P95 `0.232305/0.172312 < 0.50`; `317/320` requested frames were mechanically admissible (`0.990625` coverage). The selected dynamics-pretrain score was `0.999806`, with active-acceleration rRMSE `0.999806` and position-increment rRMSE `0.885165`, which does not demonstrate useful learned dynamics. K=1 epoch 1 completed, but epoch 2 stopped at `train_00040`, frame `340`, when the default FP32 gradient-norm reduction overflowed; diagnostic replay found that large individual gradients can still be finite. No K=1 validation rollout, K=2--16 result, or test claim exists. | `outputs/deforming_plate_chp_neural_feature_energy_full400_seed42/constitutive_pretraining.json`; `outputs/deforming_plate_chp_neural_feature_energy_full400_seed42/dynamics_pretraining.json`; `outputs/deforming_plate_chp_neural_feature_energy_full400_seed42/history.json`; `outputs/deforming_plate_chp_neural_feature_energy_seed42.stderr.log` |
| Frozen-potential force identifiability | **DIAGNOSTIC ONLY / NO IDENTIFIABILITY**; the scalar-stress-gate checkpoint is frozen, contact-neighbour nodes are excluded, and one non-negative global inverse-inertia scale is fitted on train only | train `64 x 8`: `alpha=0`, cosine `-0.000415`, rRMSE `1.0`, `511/512` admissible; frozen validation `20 x 16`: `alpha=0`, cosine `0.000401`, rRMSE `1.0`, `317/320` admissible. Validation equals the zero-force baseline. CUDA peak allocation was `0.030548 GiB`; source revision `ece1f54` was clean and test arrays loaded were `0`. | `outputs/deforming_plate_chp_force_identifiability_seed42/frozen_train_fit.json`; `outputs/deforming_plate_chp_force_identifiability_seed42/metrics.val20.json`; `outputs/deforming_plate_chp_force_identifiability_seed42/provenance.json` |
| GPU direct-control stress decoder | **DIAGNOSTIC ONLY**; fixed final epoch selected before validation; 20 train and 20 validation cases, 16 frames/case | post-hoc train rRMSE/P95 `0.211840/0.195026`; validation rRMSE/P95 `0.223264/0.181564`; RTX 4060 Ti CUDA BF16/FP32 peak allocation `0.101559 GiB` | `outputs/deforming_plate_constitutive_identifiability_seed20260712/metrics.val20.json`; `outputs/deforming_plate_constitutive_identifiability_seed20260712/provenance.json` |
| HyperContact-3D v1.9 | **DATA AUDIT PASSED**, not a trained-model result | `168/168` 101-frame cases passed; full finite cell/IP Cauchy tensors and non-zero reactions; `min J=0.675179`; maximum cell stress `10.772040 MPa`; maximum reaction `682.881 N`; worst Neo-Hookean internal-force-plus-reaction relative residual `2.079e-5` | `outputs/hypercontact3d_v1_9_converted_audit.json` |
| Valve/Abaqus | **BLOCKED** | no real ODB is present; full cell/IP stress, material, density, and fiber fields cannot be fabricated | `outputs/deforming_plate_chp_evidence/external_benchmark_status.json` |
| Deforming-plate frozen test-once evaluation | **UNTOUCHED** for the current protocol | no frozen CHP/fair/MultiScale/repository test metric has been produced; the table contains validation or diagnostic evidence only | no result artifact by design |

The direct-control row is a non-negative **scalar stress decoder**, not a scalar
energy potential. It neither produces a full stress tensor nor generates
internal force, so its low teacher-forced error is evidence of learnable signal
in the invariant inputs, not evidence of constitutive consistency and not a
CHP result.

The neural-feature potential teacher gate also uses deforming_plate's
`nodal_scalar_vm_fallback`, because this dataset supplies only a nodal scalar
stress label. It is a valid scalar teacher-gate pass for the shared potential,
but it is **not** tensor-supervision evidence. The failed second K=1 epoch and
absence of a K=1 validation rollout mean it is not yet evidence of stable
state evolution, long-horizon accuracy, or an improvement over any baseline.

The frozen-force diagnostic shows that the internal-force direction assembled
from this particular scalar-stress-gate checkpoint is no better than a zero
prediction for the measured acceleration target. It does **not** prove that no
jointly trained potential/contact/damping model exists: contact, damping, and
residual force are deliberately excluded, and only one frozen potential plus
one non-negative global scale is tested. Consequently, deforming_plate
internal force has not been validated and must not be claimed as such.

The ground-truth stress is not zero. The label-only audit reports a non-zero
fraction of `0.946340`, absolute stress median `12.210 kPa`, P95 `86.556 kPa`,
and maximum `1.155504 MPa` across the 100 deforming-plate test trajectories
(`outputs/deforming_plate_chp_evidence/ground_truth_audit.json`). The
`standard_reference` row has zero error only because it compares each truth
array with itself. It is not a zero-valued physical solution. The native final
displacement rRMSE is especially large because the final truth-displacement
denominator is very small; its corresponding physical RMSE is `0.04434 m`.

The current protocol's frozen model test-once evaluation has not been run.
There are legacy pre-protocol test outputs in the repository and the preceding
label-only test audit, so “untouched” here means no test inference or test score
has been used for the current model/hyperparameter selection; it does not claim
that the test files have never been read.

Objective constitutive-ambiguity ratios are `0.03928` on deforming_plate and
`0.08519` on HyperContact, both below the pre-registered `0.10` cell-memory
trigger. Cell memory therefore remains disabled
(`outputs/deforming_plate_chp_evidence/cell_memory_diagnostic.json` and
`outputs/hypercontact3d_chp_evidence/cell_memory_diagnostic.json`).
