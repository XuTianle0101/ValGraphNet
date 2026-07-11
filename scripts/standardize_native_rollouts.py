from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from valgraphnet.data.case import load_case, read_split_file
from valgraphnet.physical_evaluation import evaluate_prediction_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert native deforming_plate NPZ rollouts to shared artifacts."
    )
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    case_root = Path(args.case_root)
    native_root = Path(args.native_dir)
    output = Path(args.out)
    for case_id in read_split_file(args.split_file, args.split):
        source = native_root / f"{case_id}.npz"
        if not source.exists():
            continue
        case = load_case(case_root / case_id)
        artifact = np.load(source, allow_pickle=False)
        case_output = output / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        np.save(
            case_output / "U_pred.npy",
            artifact["pred_world_pos"].astype(np.float32) - case.nodes[None],
        )
        np.save(case_output / "S_pred.npy", artifact["pred_stress"].astype(np.float32))
    result = evaluate_prediction_directory(
        case_root,
        args.split_file,
        args.split,
        output,
        output_path=output / "metrics.json",
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
