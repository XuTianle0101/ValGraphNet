from __future__ import annotations

import argparse

from valgraphnet.chp_train import run_chp_training
from valgraphnet.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train constitutive-consistent CHP-GNS on CUDA."
    )
    parser.add_argument(
        "--config",
        default="configs/deforming_plate_chp.full400.yaml",
        help="Path to the CHP-GNS YAML config.",
    )
    args = parser.parse_args()
    checkpoint = run_chp_training(load_config(args.config))
    print(f"best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
