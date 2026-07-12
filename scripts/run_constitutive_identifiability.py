"""Run the CUDA/BF16 deforming_plate scalar-stress control experiment."""

from __future__ import annotations

import argparse
import json

from valgraphnet.config import load_config
from valgraphnet.constitutive_identifiability import run_control_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a train-only direct cell-scalar stress decoder and evaluate fixed "
            "even-val20 x 16. This diagnostic never evaluates test."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/deforming_plate_constitutive_identifiability.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifacts = run_control_experiment(cfg, config_path=args.config)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
