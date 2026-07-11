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
valid.tfrecord
test.tfrecord
```

From the repository root, download the dataset with:

```bash
bash scripts/download_deforming_plate.sh
```

This writes files to `raw_dataset/deforming_plate/deforming_plate`, matching
`data.data_dir` in `examples/deforming_plate/config.yaml`. The original
PhysicsNeMo example downloads the same DeepMind MeshGraphNets dataset.

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
use the checked-in quick config:

```bash
python -m examples.deforming_plate.preprocess \
  --config examples/deforming_plate/config.quick.yaml
python -m examples.deforming_plate.train \
  --config examples/deforming_plate/config.quick.yaml
python -m examples.deforming_plate.rollout_eval \
  --config examples/deforming_plate/config.quick.yaml \
  --checkpoint outputs/deforming_plate_quick/best.pt
```

The full native and converted-case experiment is resumable and can be launched
with one command:

```bash
python scripts/run_deforming_plate_full_experiment.py
```

It creates the case dataset when needed, trains both routes, evaluates every
configured test case, and writes the comparison to
`outputs/deforming_plate_full_experiment/comparison.json`. It also writes
`analysis.json`, `report.md`, `training_curves.png`, and
`rollout_curves.png`. The full-data config uses trajectory-stratified time-step
sampling so every case contributes once per epoch while complete 200-frame
trajectories are retained for rollout evaluation.

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

A quick converted-case check uses the corresponding small split:

```bash
python -m examples.deforming_plate.convert_to_cases \
  --config examples/deforming_plate/config.quick.yaml \
  --out data/deforming_plate_cases_quick
python scripts/train.py --config configs/deforming_plate_case.quick.yaml
python scripts/rollout.py \
  --config configs/deforming_plate_case.quick.yaml \
  --checkpoint outputs/deforming_plate_case_quick/best.pt \
  --case data/deforming_plate_cases_quick/test_00000 \
  --out outputs/deforming_plate_case_quick/rollout_test_00000
```

## GPU Memory Settings

The full configs are tuned for a local CUDA GPU with 8 GB of memory:

- `graph.max_world_neighbors: 32` prevents a dense contact frame from turning
  into an almost fully connected world graph while retaining each node's
  nearest world-space contacts.
- `training.amp: true` with `amp_dtype: bfloat16` uses tensor-core mixed
  precision without FP16's narrow numeric range.
- `model.num_processor_checkpoint_segments: 3` recomputes processor
  activations during backward to reduce the training peak.
- Native autoregressive inference uses PhysicsNeMo Warp radius search so
  dynamic world-edge search and edge-feature construction stay on CUDA.

Set `training.device: cuda` to require a GPU. The default `auto` selects CUDA
when it is available and otherwise falls back to CPU. Checkpoints include the
optimizer, scheduler where applicable, AMP scaler, epoch, and config. Set
`training.resume_from: auto` together with `training.save_latest: true` to
continue from `latest.pt` after an interrupted full run.

## Dependencies

Recommended setup from the repository root:

```bash
bash scripts/setup_env.sh --torch-backend auto --profile dev
source .venv/bin/activate
```

Use `--torch-backend cpu` for a CPU-only environment, or pass `cu118`, `cu126`,
or `cu128` when you need to override automatic CUDA detection. The setup script
installs PyTorch first, then the pinned text requirements in
`requirements/dev.txt`, and finally installs this repository in editable mode.

Manual dependency files:

- `requirements/base.txt`
- `requirements/deforming_plate.txt`
- `requirements/dev.txt`

The native example requires PyTorch, PyTorch Geometric, PhysicsNeMo, `tfrecord`,
SciPy, and optionally Matplotlib for gif generation.
