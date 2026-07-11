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

# 3. Corrected four-output MGN: delta-x + asinh stress
.venv\Scripts\python.exe scripts\train_fair_deforming_plate.py `
  --config configs\deforming_plate_fair_mgn.full400.yaml
.venv\Scripts\python.exe scripts\export_fair_rollouts.py `
  --config configs\deforming_plate_fair_mgn.full400.yaml `
  --checkpoint <fair-best.pt> --out <fair-standard-dir>

# 4. CHP-GNS K=1 -> 2 -> 4 -> 8 -> 16 physical rollout curriculum
.venv\Scripts\python.exe scripts\train_chp.py `
  --config configs\deforming_plate_chp.full400.yaml
.venv\Scripts\python.exe scripts\evaluate_chp.py `
  --config configs\deforming_plate_chp.full400.yaml `
  --checkpoint <chp-best.pt> --out <chp-standard-dir>

# 5. Paired physical-unit comparison and bootstrap confidence intervals
.venv\Scripts\python.exe scripts\compare_physical_rollouts.py `
  --case-root data\deforming_plate_cases_400 `
  --split-file data\deforming_plate_cases_400\splits.json `
  --experiment native=<native-standard-dir> `
  --experiment repo=<repo-standard-dir> `
  --experiment fair_mgn=<fair-standard-dir> `
  --experiment chp_gns=<chp-standard-dir> `
  --baseline fair_mgn --candidate chp_gns --out <comparison.json>
```

The common evaluator uses moving nodes for displacement, all non-prescribed
nodes for stress (including the clamped high-stress region), physical units,
truth top-5% stress support for P95 error, and pooled squared errors. The
checkpoint objective is the worst of four ratios to the frozen native result;
one improving metric cannot compensate for another regressing metric.

## Constitutive and benchmark gates

The exact-geometry teacher-stress gate is checked after the four K=1 epochs.
Failure at relative RMSE 0.50 stops the rollout curriculum. Cell memory is off
by default and is enabled only when `scripts/diagnose_cell_memory.py` reports a
conditional-to-global stress variance above 0.10.

HyperContact-3D decks and deterministic ID/material/load/mesh/combined OOD
splits are generated with:

```powershell
.venv\Scripts\python.exe scripts\generate_hypercontact3d.py `
  --config configs\hypercontact3d.yaml --output data\hypercontact3d_raw
```

The DeepMind source exposes one nodal scalar stress channel, not a six-component
cell tensor. Full tensor supervision and material/mesh OOD evidence therefore
come from HyperContact-3D and Valve/Abaqus. Valve is a hard external-data gate:
an ODB export must contain complete cell/IP stress, density, material features,
and fiber directions before it is included in a paper claim.
