"""Train the CUDA/BF16 two-level MultiScale MGN deforming-plate baseline."""

from __future__ import annotations

import argparse

from valgraphnet.config import load_config
from valgraphnet.multiscale_train import run_multiscale_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/deforming_plate_multiscale_mgn.full400.yaml",
    )
    args = parser.parse_args()
    checkpoint = run_multiscale_training(load_config(args.config))
    print(f"best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
