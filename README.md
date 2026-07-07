# ValGraphNet

ValGraphNet 是一个把 Abaqus 人工心脏瓣膜固体动力学结果转换为图时序数据，并用 NVIDIA PhysicsNeMo `HybridMeshGraphNet` 训练 surrogate model 的起步代码库。

目标场景：

- 瓣膜附着边固定。
- 心室侧表面施加动态跨瓣压差 `p(t)`。
- Abaqus 已有大量壳/膜单元动力学结果。
- 不同样本对应不同瓣膜设计，材料参数暂按固定处理。
- 训练逐步自回归模型：当前状态和当前边界条件 -> 下一步节点状态增量。

## 目录结构

```text
configs/valve_hybrid.yaml      # 默认训练配置
scripts/abaqus_export_odb.py   # 在 Abaqus Python 中运行，导出 case 数据
scripts/train.py               # 训练入口
scripts/rollout.py             # 自回归推理入口
valgraphnet/                   # 数据、模型、损失和训练代码
tests/                         # 不依赖 Abaqus/PhysicsNeMo 的基础测试
```

## 数据格式

每个 Abaqus case 导出为一个目录：

```text
data/processed/case_001/
  metadata.json
  nodes.npy          # [N, 3], reference coordinates
  elements.npy       # [M, K], zero-based node ids, padded by -1
  times.npy          # [T]
  pressure.npy       # [T], transvalvular pressure
  U.npy              # [T, N, 3]
  V.npy              # [T, N, 3]
  A.npy              # [T, N, 3]
  S.npy              # [T, N, C], optional nodal stress
  fixed_mask.npy     # [N], attachment boundary
  pressure_mask.npy  # [N], ventricular-side pressure surface
  leaflet_id.npy     # [N], different ids for different leaflets
  thickness.npy      # [N] or scalar
```

`elements.npy` 使用 0-based 节点索引，方便直接构图。`S.npy` 可以是 6 分量应力，也可以是 von Mises 单分量；维度由数据自动推断。

## Abaqus 导出

在 Abaqus 命令行中运行：

```bash
abaqus python scripts/abaqus_export_odb.py -- \
  --odb path/to/case.odb \
  --out data/processed/case_001 \
  --instance VALVE \
  --fixed-set ATTACHMENT \
  --pressure-surface VENTRICULAR_SURFACE \
  --leaflet-sets LEAFLET_1,LEAFLET_2,LEAFLET_3 \
  --pressure-csv path/to/pressure.csv
```

`pressure.csv` 建议两列：`time,pressure`。如果 ODB 内字段命名不是 Abaqus 默认的 `U/V/A/S`，可以用脚本参数覆盖。

## 训练

安装依赖后运行：

```bash
python scripts/train.py --config configs/valve_hybrid.yaml
```

训练代码默认使用 `HybridMeshGraphNet`，把 mesh edges 和 world/contact edges 分开编码。划分数据时建议提供 `splits.json`：

```json
{
  "train": ["case_001", "case_002"],
  "val": ["case_003"],
  "test": ["case_004"]
}
```

注意：必须按 case/design 划分，不能把同一个 case 的不同时间帧随机分到训练和验证里。

## 自回归推理

```bash
python scripts/rollout.py \
  --config configs/valve_hybrid.yaml \
  --checkpoint outputs/valve_hybrid/best.pt \
  --case data/processed/case_003 \
  --out outputs/valve_hybrid/rollout_case_003
```

输出包括 `U_pred.npy`、`V_pred.npy`、`A_pred.npy`、`S_pred.npy`，可再转换为 VTK/VTU 与 Abaqus 云图对比。

## 重要约定

- fixed nodes 在训练和 rollout 中被硬约束为零增量。
- 压力条件以 `p_k, p_{k+1}, dp/dt, phase` 和 nodal traction 形式进入节点特征。
- contact/world edges 每一帧按当前变形坐标搜索不同 leaflet 之间的近邻。
- 默认不引入材料参数输入；如果后续材料参数变化，把厚度、密度、超弹性参数加入节点或全局条件。

## Deforming Plate Example

This repository also includes a PhysicsNeMo-style deforming-plate example under
`examples/deforming_plate`.

Native path:

```bash
bash scripts/download_deforming_plate.sh
python -m examples.deforming_plate.preprocess --config examples/deforming_plate/config.yaml
python -m examples.deforming_plate.train --config examples/deforming_plate/config.yaml
python -m examples.deforming_plate.rollout_eval --config examples/deforming_plate/config.yaml --checkpoint outputs/deforming_plate/best.pt
```

ValGraphNet case conversion path:

```bash
python -m examples.deforming_plate.convert_to_cases --config examples/deforming_plate/config.yaml --out data/deforming_plate_cases
python scripts/train.py --config configs/deforming_plate_case.yaml
python scripts/rollout.py --config configs/deforming_plate_case.yaml --checkpoint outputs/deforming_plate_case/best.pt --case data/deforming_plate_cases/test_00000 --out outputs/deforming_plate_case/rollout_test_00000
```

See `examples/deforming_plate/README.md` for data layout and dependency notes.

## Environment Setup

Recommended on Linux/macOS/WSL:

```bash
cd /path/to/ValGraphNet
bash scripts/setup_env.sh --torch-backend auto --profile dev
source .venv/bin/activate
```

The default `--torch-backend auto` mode checks `nvidia-smi`, selects the highest
supported PyTorch CUDA wheel from the script's known backends, and falls back to
CPU wheels when a compatible NVIDIA GPU/CUDA version cannot be detected.

Manual overrides:

```bash
# Force CPU wheels.
bash scripts/setup_env.sh --torch-backend cpu --profile dev

# Force a specific PyTorch CUDA wheel when auto detection is not appropriate.
bash scripts/setup_env.sh --torch-backend cu118 --profile dev
bash scripts/setup_env.sh --torch-backend cu126 --profile dev
bash scripts/setup_env.sh --torch-backend cu128 --profile dev
```

Profiles:

- `base`: ValGraphNet training/rollout dependencies.
- `deforming_plate`: `base` plus TFRecord, SciPy, Matplotlib, and TensorBoard.
- `dev`: `deforming_plate` plus pytest.

Manual installation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first. Choose the command from the official PyTorch selector
# for your driver/CUDA/CPU environment, then install the remaining dependencies.
pip install -r requirements/dev.txt
pip install -e . --no-deps
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_geometric, physicsnemo; print(torch_geometric.__version__, physicsnemo.__version__)"
pytest -q
```
