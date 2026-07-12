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

- The first low-order deforming-plate CHP pilot stopped at teacher-forced
  stress rRMSE `0.69265 > 0.50`; no later rollout or test result from that run is
  considered valid evidence.
- Objective constitutive-ambiguity ratios are `0.03928` on deforming_plate and
  `0.08519` on HyperContact, both below the pre-registered `0.10` cell-memory
  trigger. Cell memory therefore remains disabled.
- HyperContact v1.9 has 168/168 audited 101-frame cases, positive deformation
  determinants (`min J=0.67518`), non-zero full cell/IP stress and solver
  reactions. Reassembled Neo-Hookean internal force plus fixed reaction has a
  worst-case relative residual of `2.08e-5`.
- Valve/Abaqus remains blocked until real ODB files satisfying the strict
  tensor/material/density/fiber contract are supplied.
