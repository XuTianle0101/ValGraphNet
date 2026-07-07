# Deforming Plate Example

This example adapts the PhysicsNeMo deforming-plate MeshGraphNet recipe to this
repository. It supports two paths:

- Native example: train and evaluate `HybridMeshGraphNet` directly on the
  DeepMind deforming-plate TFRecord data.
- ValGraphNet case conversion: convert the same TFRecords into this repository's
  `.npy` case format and run the existing `scripts/train.py` and
  `scripts/rollout.py` entrypoints.

## Data

Download the DeepMind MeshGraphNets deforming-plate dataset and place the files
under the path configured by `data.data_dir` in `examples/deforming_plate/config.yaml`.
The directory should contain:

```text
meta.json
train.tfrecord
eval.tfrecord
test.tfrecord
```

The original PhysicsNeMo example downloads it through DeepMind's
`deepmind-research/meshgraphnets/download_dataset.sh deforming_plate` helper.

## Native Training And Evaluation

```bash
python -m examples.deforming_plate.preprocess --config examples/deforming_plate/config.yaml
python -m examples.deforming_plate.train --config examples/deforming_plate/config.yaml
python -m examples.deforming_plate.rollout_eval \
  --config examples/deforming_plate/config.yaml \
  --checkpoint outputs/deforming_plate/best.pt
```

The default config follows the full example scale: 1000 training samples,
200 time steps, batch size 1, and 30 epochs. For a quick local sanity check,
override the sample and epoch counts in a copy of the config.

`rollout_eval.py` writes per-sequence `.npz` artifacts and a `metrics.json` with:

- `displacement_rmse`
- `rollout_rmse`
- `stress_rmse`

Set `rollout.make_gif: true` in the config to save a simple predicted/exact
scatter animation for the first test sequence.

## ValGraphNet Case Conversion

```bash
python -m examples.deforming_plate.convert_to_cases \
  --config examples/deforming_plate/config.yaml \
  --out data/deforming_plate_cases

python scripts/train.py --config configs/deforming_plate_case.yaml

python scripts/rollout.py \
  --config configs/deforming_plate_case.yaml \
  --checkpoint outputs/deforming_plate_case/best.pt \
  --case data/deforming_plate_cases/test_00000 \
  --out outputs/deforming_plate_case/rollout_test_00000
```

The converter maps `world_pos - mesh_pos` to `U.npy`, finite differences to
`V.npy` and `A.npy`, and the first stress channel to `S.npy`. DeepMind
tetrahedral cells are converted into unique two-node edge elements so the
existing ValGraphNet graph builder can preserve mesh connectivity without
changing the shell/quad element convention used by valve data.

## Dependencies

Recommended setup from the repository root:

```powershell
.\scripts\setup_env.ps1 -TorchBackend cu126 -Profile dev
.\.venv\Scripts\Activate.ps1
```

Use `-TorchBackend cpu` for a CPU-only environment. The setup script installs
PyTorch first, then the pinned text requirements in `requirements/dev.txt`, and
finally installs this repository in editable mode.

Manual dependency files:

- `requirements/base.txt`
- `requirements/deforming_plate.txt`
- `requirements/dev.txt`

The native example requires PyTorch, PyTorch Geometric, PhysicsNeMo, `tfrecord`,
SciPy, and optionally Matplotlib for gif generation.
