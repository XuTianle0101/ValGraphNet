"""Train the CUDA/BF16 corrected deforming_plate MeshGraphNet baseline."""

from __future__ import annotations

import argparse

from valgraphnet.config import load_config
from valgraphnet.fair_train import run_fair_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the fair baseline YAML config.")
    args = parser.parse_args()
    checkpoint = run_fair_training(load_config(args.config))
    print(f"best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
